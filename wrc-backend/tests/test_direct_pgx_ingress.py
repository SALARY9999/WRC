"""Direct PGX ingress on loopback.  No PC relay, no CUDA, no Alpamayo checkpoint.

What this runs
    * configuration rejection of the retired relay
    * the real ``UdpPoseListener``, ``PlannedRouteTcpReceiver`` and
      ``CimgCameraSource`` on ports the OS assigns
    * ``TimestampManager`` on the pose and camera clocks
    * the received route through ``PlannedRouteReferenceProvider`` and
      ``TakeoverDecisionEngine``
    * ``WaypointSender`` into a socket this process bound, parsed here with an
      independent ``struct`` layout

What this does not run
    * the Alpamayo checkpoint or a GPU worker (see ``test_inference_scheduler``)
    * a real JPEG decode (the receiver is given a decoder that returns a blank
      frame, so OpenCV is not required for ingress)
    * a vehicle, or the production listen ports 30106 / 7001 / 30006 / 8090
"""

from __future__ import annotations

import json
import os
import socket
import struct
import time
import unittest
from unittest.mock import patch

import numpy as np

from camera.cimg import IMAGE_HEADER, MAGIC, PACKET_HEADER, PROTOCOL_VERSION
from camera.cimg_source import CimgCameraSource, CimgConfig
from config.loader import AppConfig, load_network_config
from time_sync import TimestampManager
from trajectory.coordinates import local_xy_to_latlon
from trajectory.takeover import (
    DeviationMetrics,
    ModelFrameOrigin,
    PlannedRouteCache,
    PlannedRouteReferenceProvider,
    ReferenceTrajectoryRequest,
    TakeoverDecisionEngine,
    TrajectoryDeviationCalculator,
)
from vehicle.history import PoseHistory
from vehicle.planned_route import PlannedRouteTcpReceiver
from vehicle.receiver import UdpPoseListener
from waypoint.sender import WaypointSender


def _free_port(kind: str) -> int:
    sock_type = socket.SOCK_DGRAM if kind == "udp" else socket.SOCK_STREAM
    with socket.socket(socket.AF_INET, sock_type) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _jpeg() -> bytes:
    sof = b"\xff\xc0" + struct.pack("!H", 11) + bytes([8])
    sof += struct.pack("!HH", 8, 8) + b"\x01\x01\x11\x00"
    return b"\xff\xd8" + sof + b"\x00" * 8 + b"\xff\xd9"


def _cimg_packet(group_ns: int, step: int) -> bytes:
    jpeg = _jpeg()
    records = b"".join(
        IMAGE_HEADER.pack(camera_id, 0, 100 + step, group_ns, len(jpeg)) + jpeg
        for camera_id in (1, 2, 3, 4)
    )
    return PACKET_HEADER.pack(
        MAGIC, PROTOCOL_VERSION, 4, group_ns, len(records)
    ) + records


def _route_frame() -> bytes:
    points = ((113.0, 28.0), (113.0001, 28.0001), (113.0002, 28.0002))
    body = b"".join(struct.pack("<ff", lon, lat) for lon, lat in points)
    return struct.pack("<BBIIB", 0xF1, 0x10, 11 + len(body), len(points), 0) + body


class _Buffer:
    def __init__(self) -> None:
        self.items: list = []

    def put(self, entry) -> None:
        self.items.append(entry)

    @property
    def latest(self):
        return self.items[-1] if self.items else None


class _CleanEnv(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (
            "ALPAMAYO_CONFIG",
            "RELAY_HOST",
            "RELAY_PORT",
            "PC_IP",
            "ALPAMAYO_RELAY_TCP_PORT",
        ):
            os.environ.pop(name, None)
            self.addCleanup(os.environ.pop, name, None)


class DirectIngressTests(_CleanEnv):
    def test_retired_relay_config_is_rejected(self) -> None:
        os.environ["PC_IP"] = "10.0.0.1"
        with self.assertRaises(ValueError) as caught:
            load_network_config()
        self.assertIn("PC_IP", str(caught.exception))

    def test_pose_route_and_camera_are_received_on_loopback(self) -> None:
        clocks = TimestampManager()
        history = PoseHistory()
        pose_port = _free_port("udp")
        pose = UdpPoseListener(
            history,
            pose_host="127.0.0.1",
            pose_port=pose_port,
            timestamp_manager=clocks,
        )
        pose.start()
        self.addCleanup(pose.stop)

        route_cache = PlannedRouteCache()
        route = PlannedRouteTcpReceiver(route_cache, "127.0.0.1", 0)
        route.start()
        self.addCleanup(route.stop)
        self.assertTrue(_wait(lambda: route.listening), route.last_error)

        buffers = [_Buffer() for _ in range(4)]
        camera_port = _free_port("tcp")
        camera = CimgCameraSource(
            buffers,
            config=CimgConfig(host="127.0.0.1", port=camera_port),
            timestamp_manager=clocks,
            decoder=lambda _jpeg, size: np.zeros((size[1], size[0], 3), np.uint8),
        )
        camera.start()
        self.addCleanup(camera.stop)
        self.addCleanup(lambda: camera.join(timeout=2))

        now_s = time.time()
        payload = json.dumps(
            {
                "type": "vehicle_state",
                "timestamp": now_s,
                "pose": {
                    "latitude": 28.0,
                    "longitude": 113.0,
                    "azimuth": 90.0,
                    "height": 10.0,
                    "vNorth": 0.0,
                    "vEast": 5.0,
                    "chassis": {"cur_spd": 5.0, "yaw_rate": 0.0},
                },
            }
        ).encode("utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(payload, ("127.0.0.1", pose_port))
        self.assertTrue(_wait(lambda: history.latest() is not None), pose.last_error)
        latest = history.latest()
        self.assertEqual(28.0, latest.latitude)
        self.assertEqual(113.0, latest.longitude)
        self.assertEqual(
            int(json.loads(payload)["timestamp"] * 1_000_000),
            latest.timestamp_us,
        )
        self.assertIsNotNone(pose.latest_chassis())

        with socket.create_connection(("127.0.0.1", route.port), timeout=2) as client:
            client.sendall(_route_frame())
        self.assertTrue(_wait(lambda: route.updates == 1), route.last_error)
        cached, _version = route_cache.snapshot()
        self.assertEqual(3, len(cached.points))

        group_ns = time.time_ns() - 50_000_000
        with socket.create_connection(("127.0.0.1", camera_port), timeout=2) as client:
            client.sendall(_cimg_packet(group_ns, 0))
        self.assertTrue(
            _wait(lambda: camera.groups_ingested == 1),
            f"ingested={camera.groups_ingested} timestamp_errors={camera.timestamp_errors} "
            f"decode_errors={camera.decode_errors}",
        )
        self.assertEqual(1, len(buffers[0].items))
        frame = buffers[0].latest
        self.assertIsNotNone(frame)
        self.assertGreater(frame.timestamp_us, 1_000_000_000_000_000)

        provider = PlannedRouteReferenceProvider(route_cache)
        reference = provider.generate(
            ReferenceTrajectoryRequest(
                np.empty((0, 2)),
                np.empty(0),
                np.array([2_000_000, 3_000_000, 4_000_000]),
                ModelFrameOrigin(28.0, 113.0, 0.0, 0.0, 0.0),
                np.array([5.0, 0.0]),
                2_000_000,
                2.0,
            )
        )
        self.assertIsNotNone(reference)
        metrics = TrajectoryDeviationCalculator().calculate(
            np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        )
        self.assertIsNotNone(metrics)
        engine = TakeoverDecisionEngine(0.8, 0.4, takeover_confirmations=2)
        first = engine.update(metrics)
        second = engine.update(metrics)
        self.assertEqual(0, first.control_flag)
        self.assertEqual(1, second.control_flag)

        capture = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        capture.bind(("127.0.0.1", 0))
        self.addCleanup(capture.close)
        host, port = capture.getsockname()[:2]
        settings = AppConfig(
            waypoint_host=host,
            waypoint_port=port,
            magic=0x514D3031,
            vehicle_id=1,
        )
        sender = WaypointSender(settings)
        self.addCleanup(sender.close)
        lat, lon = local_xy_to_latlon(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            28.0,
            113.0,
            0.0,
        )
        sender.send(second.control_flag, lat, lon)
        capture.settimeout(2)
        packet, _peer = capture.recvfrom(65535)
        magic, version, control, seq, vehicle_id, count = struct.unpack_from("<IBBIHH", packet)
        self.assertEqual(0x514D3031, magic)
        self.assertEqual(1, version)
        self.assertEqual(1, control)
        self.assertEqual(0, seq)
        self.assertEqual(1, vehicle_id)
        self.assertEqual(2, count)
        self.assertEqual(14 + 2 * 16, len(packet))
        parsed = [
            struct.unpack_from("<dd", packet, 14 + index * 16)
            for index in range(count)
        ]
        self.assertEqual(2, len(parsed))
        self.assertTrue(all(abs(item[0]) <= 90 and abs(item[1]) <= 180 for item in parsed))

    def test_four_received_groups_build_a_camera_window(self) -> None:
        import sys
        import types

        if "cv2" not in sys.modules:
            opencv = types.ModuleType("cv2")

            def _resize(image, size, interpolation=None):
                return np.zeros((size[1], size[0], image.shape[2]), dtype=np.uint8)

            def _cvt_color(image, code):
                return np.ascontiguousarray(image[:, :, ::-1])

            opencv.resize = _resize
            opencv.cvtColor = _cvt_color
            opencv.INTER_LINEAR = 1
            opencv.COLOR_BGR2RGB = 4
            opencv.imwrite = lambda *_args, **_kwargs: True
            sys.modules["cv2"] = opencv

        os.environ["ALPAMAYO_SNAPSHOT_DIR"] = ""
        os.environ["ALPAMAYO_CAMERA_STATS_INTERVAL_S"] = "0"
        from camera.sync import CameraWindowProvider

        clocks = TimestampManager()
        port = _free_port("tcp")
        provider = CameraWindowProvider(
            CimgConfig(
                host="127.0.0.1",
                port=port,
                camera_indices=(1, 2, 0, 4),
                read_timeout_s=2.0,
            ),
            frames_per_camera=4,
            timestamp_manager=clocks,
        )
        provider.readers[0]._decode = lambda _jpeg, size: np.zeros(
            (size[1], size[0], 3), np.uint8
        )
        provider.start()
        self.addCleanup(provider.stop)
        self.addCleanup(lambda: provider.readers[0].join(timeout=2))

        now_ns = time.time_ns()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            for step in range(4):
                group_ns = now_ns - (3 - step) * 100_000_000
                client.sendall(_cimg_packet(group_ns, step))
                # The production receiver deliberately collapses multiple
                # complete groups found in one socket drain to the newest one.
                # Wait for each group here so this test exercises four live
                # arrivals instead of accidentally testing backlog dropping.
                self.assertTrue(
                    _wait(
                        lambda expected=step + 1: (
                            provider.readers[0].groups_ingested >= expected
                        ),
                        timeout=2.0,
                    ),
                    f"step={step} ingested={provider.readers[0].groups_ingested}",
                )
        reader = provider.readers[0]
        self.assertTrue(
            _wait(lambda: reader.groups_ingested == 4, timeout=4.0),
            f"ingested={reader.groups_ingested} timestamp_errors={reader.timestamp_errors} "
            f"decode_errors={reader.decode_errors} refusals={dict(provider.refusals)}",
        )
        window = provider.latest_window(sync_timeout_s=2.0, catch_up_timeout_s=0.2)
        self.assertIsNotNone(window, dict(provider.refusals))
        self.assertEqual(4, len(window.camera_indices))
        self.assertEqual(4, window.frames.shape[1])
        self.assertGreater(window.t0_us, 0)


if __name__ == "__main__":
    unittest.main()
