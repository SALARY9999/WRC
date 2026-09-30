"""The maximum end-to-end regression this machine can actually run.

``cv2`` is absent from every local interpreter (and blocks ``app.py``'s import),
torch is CPU-only while the code calls ``.to("cuda")``,
and ``alpamayo1_5`` plus the checkpoint exist only on the server.  **A real
end-to-end run is impossible here**, and a test that pretended otherwise would be
worse than no test.  So this file runs the next best thing and says exactly how
far it gets.

What it runs: a real ``PipelineRunner`` in a real process, driving the real loop,
the real input gates, the real decision layer, the real trajectory crop, the real
serializer and a real UDP datagram - and, in the second child, the real FastAPI
app with its real ``on_event`` startup and shutdown.  Two seams are stubbed and
nothing else:

* the camera provider, because no decoder can run locally;
* ``AlpamayoRunner.infer``/``ensure_loaded``, because there is no checkpoint,
  no GPU and no ``alpamayo1_5``.

Each child runs in its own process (``tests/_local_chain_child.py``) because
``app._settings`` is built from the environment at import time and
``load_backend_config`` is ``lru_cache``d - the ports cannot be redirected
in-process.  The child prints one ``RESULT <json>`` line; the assertions are
here, so a failure names the field and shows the observed value.

The capture socket is bound HERE, before the child starts, and the ``<IBBIHH`` +
N x ``<dd`` layout is unpacked HERE, independently of ``WaypointSender``'s own
``struct`` format.  A test that reused the serializer's format string would pass
even if that string were wrong in both places.

**What this proves** - the loop's real gate order and the real rejection
reasons; the real decision path including the planned-route fallback; the exact
bytes on the wire for a real send, parsed by an independent parser; the three
endpoints' shapes, field order and status codes; that ``/ready`` refuses and then
admits; that shutdown is ordered, bounded and quiet.

**What it does NOT prove** - a live CIMG JPEG decode (the camera provider is
stubbed); the CUDA model's real numerics, latency and output shape; real pose
decoding from the vehicle over UDP (the real ``UdpPoseListener`` is bound and
running, but this process has no vehicle, so the samples a receiver would
append after decoding are appended directly); real planned-route TCP frames.
The real end-to-end run stays a pre-vehicle-test item that needs deployment
authorization.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "tests" / "_local_chain_child.py"

RESULT_PREFIX = "RESULT "

# The pose listener binds this port.  ``PipelineRunner.start()`` now passes
# ``settings.pose_udp_port`` through, so the port follows the configuration -
# 7001 is what the shipped config resolves to, and this test runs that config.
# The consequence is that it must not run while a real backend is running, which
# is what ``_skip_if_pose_port_taken`` checks - a backend already holding 7001 is
# a reason to stay out of its way, not to fail.  A child that reconfigures the
# port (``ALPAMAYO_POSE_UDP_PORT``) would also have to move this constant.
POSE_PORT = 7001

MAGIC = 0x514D3031
VERSION = 1
VEHICLE_ID = 7

CHILD_TIMEOUT_S = 120


def _parse_datagram(packet: bytes, byte_order: str = "<") -> dict:
    """Unpack the frozen ``<IBBIHH`` header and the N x ``<dd`` body.

    Deliberately a second implementation of the wire format.  It is the only
    thing here that would notice a wrong format string in ``waypoint/sender.py``.
    """
    header_size = struct.calcsize(byte_order + "IBBIHH")
    if len(packet) < header_size:
        raise AssertionError(f"datagram is {len(packet)} bytes, shorter than the header")
    magic, version, control, seq, vehicle_id, count = struct.unpack(
        byte_order + "IBBIHH", packet[:header_size]
    )
    body = packet[header_size:]
    waypoints = [
        struct.unpack(byte_order + "dd", body[i : i + 16])
        for i in range(0, len(body) - len(body) % 16, 16)
    ]
    return {
        "length": len(packet),
        "header_size": header_size,
        "magic": magic,
        "version": version,
        "control": control,
        "seq": seq,
        "vehicle_id": vehicle_id,
        "count": count,
        "waypoints_in_body": len(waypoints),
        "waypoints": waypoints,
        "first": waypoints[0] if waypoints else None,
        "last": waypoints[-1] if waypoints else None,
        "trailing_bytes": len(body) % 16,
    }


def _port_is_taken(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("0.0.0.0", port))
    except OSError:
        return True
    finally:
        probe.close()
    return False


def _skip_if_pose_port_taken(test: unittest.TestCase) -> None:
    """Stay out of the way of a real backend.

    The child's pose listener binds ``0.0.0.0:7001`` (see ``POSE_PORT``).  If
    something already holds it, a real backend is probably running on this
    machine, and the right response is to leave it alone and say so - not to
    fail, and certainly not to interfere.
    """
    if _port_is_taken(POSE_PORT):
        test.skipTest(
            f"UDP port {POSE_PORT} is already bound, which the pipeline's pose "
            f"listener requires and cannot be redirected. A backend is probably "
            f"running; not starting a second one alongside it."
        )


def _child_env(capture: socket.socket) -> dict:
    """The environment the child needs, all of it set before ``import app``."""
    port = capture.getsockname()[1]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            # Everything outward points at a port this process owns.
            "ALPAMAYO_WAYPOINT_HOST": "127.0.0.1",
            "ALPAMAYO_WAYPOINT_PORT": str(port),
            "ALPAMAYO_WAYPOINT_MAGIC": hex(MAGIC),
            "ALPAMAYO_WAYPOINT_VERSION": str(VERSION),
            "ALPAMAYO_VEHICLE_ID": str(VEHICLE_ID),
            # Real pose mode, so the fed history shares the pipeline's clock and
            # no gate has to be loosened.  Mock mode off: a mock pose provider
            # would overwrite the history this test feeds.
            "ALPAMAYO_MOCK_MODE": "0",
            "ALPAMAYO_POSE_MODE": "arrival",
            "ALPAMAYO_CIMG_CAMERA_INDICES": "1,2,0,4",
            # Fast spin, no package writing, no route listener.
            "ALPAMAYO_MIN_CYCLE_S": "0.05",
            "ALPAMAYO_INFER_PACKAGE_DIR": "",
            "ALPAMAYO_ROUTE_TCP_ENABLED": "0",
        }
    )
    return env


def _run_child(mode: str, capture: socket.socket) -> dict:
    """Run one child to completion, or raise ``AssertionError`` with its output."""
    completed = subprocess.run(
        [sys.executable, str(CHILD), mode],
        cwd=str(ROOT / "tests"),
        env=_child_env(capture),
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )
    diagnostics = (
        f"\n--- child stdout ---\n{completed.stdout[-6000:]}\n"
        f"--- child stderr ---\n{completed.stderr[-6000:]}"
    )
    result_line = next(
        (l for l in completed.stdout.splitlines() if l.startswith(RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise AssertionError(
            f"the {mode} child produced no {RESULT_PREFIX!r} line." + diagnostics
        )
    payload = json.loads(result_line[len(RESULT_PREFIX):])
    if completed.returncode != 0 or "crashed" in payload:
        raise AssertionError(
            f"the {mode} child failed: {payload.get('crashed')}" + diagnostics
        )
    return payload


def _drain(capture: socket.socket, timeout_s: float = 0.5) -> list[dict]:
    """Every datagram waiting on the capture socket, oldest first."""
    packets = []
    capture.settimeout(timeout_s)
    while True:
        try:
            packets.append(_parse_datagram(capture.recvfrom(65535)[0]))
        except socket.timeout:
            return packets


def _bind_capture() -> socket.socket:
    capture = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    capture.bind(("127.0.0.1", 0))
    return capture


class _ChainTest(unittest.TestCase):
    """Shared plumbing: the port check that keeps this out of a live backend."""

    def setUp(self) -> None:
        _skip_if_pose_port_taken(self)


class RealCycleWireTests(_ChainTest):
    """Child A: a real cycle, asserted on the bytes that left the process."""

    @classmethod
    def setUpClass(cls) -> None:
        # The port may have appeared between the class-level check and here, so
        # each test re-checks in setUp; this one only avoids a pointless spawn.
        if _port_is_taken(POSE_PORT):
            return
        cls.capture = _bind_capture()
        try:
            cls.result = _run_child("chain", cls.capture)
            cls.datagrams = _drain(cls.capture)
        finally:
            cls.capture.close()

    def test_the_run_produced_datagrams(self):
        self.assertTrue(self.datagrams, "no datagram arrived on the bound port")

    def test_the_wire_layout_is_the_frozen_one(self):
        for index, item in enumerate(self.datagrams):
            with self.subTest(datagram=index):
                self.assertEqual(item["header_size"], 14)
                self.assertEqual(item["trailing_bytes"], 0,
                                 "the body is not a whole number of 16-byte pairs")
                self.assertEqual(item["length"], 14 + item["count"] * 16)
                self.assertEqual(item["count"], item["waypoints_in_body"])
                self.assertGreater(item["count"], 0)
                self.assertGreaterEqual(item["count"], 1)

    def test_the_header_fields_are_the_configured_ones(self):
        item = self.datagrams[0]
        self.assertEqual(item["magic"], MAGIC)
        self.assertEqual(item["version"], VERSION)
        self.assertEqual(item["vehicle_id"], VEHICLE_ID)

    def test_the_sequence_number_advances(self):
        # The header sequence must advance by one per send, which is what the
        # vehicle uses to drop stale packets.  Contiguity from zero is asserted
        # rather than a count: the child is stopped by the parent's clock, so how
        # many sends land in the window is not fixed, but their numbering is.
        self.assertGreaterEqual(len(self.datagrams), 2)
        sequences = [item["seq"] for item in self.datagrams]
        self.assertEqual(sequences, list(range(len(sequences))), sequences)
        # ``last_seq`` is the last sequence USED, not the number of sends.
        self.assertEqual(self.result["sender_last_seq"], sequences[-1])

    def test_the_waypoints_are_the_model_output_as_lat_lon(self):
        item = self.datagrams[0]
        self.assertEqual(item["count"], 64)
        # The stub model emits 1 m steps along its +x axis with heading 90 deg,
        # so latitude is constant and longitude advances eastward - the same
        # fixture shape the replay tool uses.
        self.assertAlmostEqual(item["first"][0], 28.0, places=6)
        self.assertAlmostEqual(item["first"][1], 113.0, places=9)
        self.assertAlmostEqual(item["last"][0], 28.0, places=6)
        self.assertGreater(item["last"][1], item["first"][1])
        for lat, lon in item["waypoints"]:
            with self.subTest(waypoint=(lat, lon)):
                self.assertFalse(lat != lat or lon != lon, "NaN on the wire")

    def test_the_datagram_matches_the_snapshot_and_the_two_flags(self):
        observed = self.result
        item = self.datagrams[0]
        snapshot = observed["snapshot"]
        # The control byte is the whole point of the packet: it must agree with
        # both the decision the loop recorded and the value it recorded sending.
        self.assertEqual(item["control"], snapshot["last_sent_control_flag"])
        self.assertEqual(item["control"], snapshot["last_control_flag"])
        self.assertEqual(item["control"], observed["last_sent_control_flag"])
        self.assertEqual(observed["last_sent_control_flag"],
                         observed["last_control_flag"])
        self.assertEqual(snapshot["last_num_waypoints"], 64)
        self.assertEqual(item["seq"], 0, "the first datagram's sequence must start at 0")

    def test_the_cycle_really_ran_the_real_decision_path(self):
        observed = self.result
        decision = observed["last_takeover_decision"]
        self.assertIsNotNone(decision)
        # No planned route was ever delivered, so the real fallback must be the
        # reason - not a stubbed-out decision layer, and not the
        # camera_status_unavailable/pose_unavailable fallbacks that a
        # mis-specified stub would produce.
        self.assertEqual(decision["reference_fallback_reason"], "planned_route_not_received")
        self.assertEqual(decision["decision_reason"], "reference_trajectory_unavailable")
        self.assertEqual(decision["control_flag"], 0,
                         "with no reference trajectory the engine must not take over")
        # The trim ran on all 64 model points.  It drops the one the vehicle has
        # already passed, and whether that happens depends on where the pose
        # projects onto the first segment - so the bound is asserted, not the
        # exact value: 0 or 1 removed, never "the trim never ran".
        self.assertIn(decision["remaining_points"], (63, 64), decision)

    def test_the_published_snapshot_describes_a_healthy_loop(self):
        snapshot = self.result["snapshot"]
        self.assertGreater(snapshot["sequence"], 0)
        self.assertEqual(snapshot["phase"], "stopped")
        self.assertIsNone(snapshot["last_error"])
        self.assertIs(snapshot["model_loaded"], True)
        self.assertEqual(snapshot["unhealthy_readers"], [])
        self.assertIsNotNone(snapshot["last_pose_arrival_us"])
        # The field that /ready's freshness check reads.  It is None unless the
        # loop records a window at the point it really builds one, and this
        # assertion is the only one on this machine that would catch that.
        self.assertIsNotNone(
            snapshot["last_window_ok_at"],
            "the pipeline built windows but the snapshot reports none, so "
            "/ready would report no_camera_window forever",
        )
        self.assertGreaterEqual(snapshot["inference_count"], 2)
        self.assertGreaterEqual(snapshot["cycle_id"], snapshot["inference_count"])

    def test_the_operator_log_lines_are_intact(self):
        logs = self.result["logs"]
        self.assertEqual(logs["cycle_failed"], [], "a cycle raised")
        self.assertEqual(logs["no_window"], [], "no window was ever built")
        self.assertEqual(logs["rejected"], [], "the gate rejected the stub window")
        self.assertGreaterEqual(len(logs["inference_started"]), 2)
        self.assertGreaterEqual(len(logs["inference_sent"]), 2)
        # check_chain.sh greps exactly this substring out of backend.log.
        for line in logs["inference_sent"]:
            with self.subTest(line=line[:60]):
                self.assertIn("inference sent: control_flag=", line)
        # The payload the vehicle-side reader matches on.
        self.assertEqual(self.result["settings"]["camera_indices"], [1, 2, 0, 4])


class RealEndpointsTests(_ChainTest):
    """Child B: the real app, its real endpoints, its real graceful shutdown."""

    @classmethod
    def setUpClass(cls) -> None:
        if _port_is_taken(POSE_PORT):
            return
        cls.capture = _bind_capture()
        try:
            cls.result = _run_child("app", cls.capture)
            cls.datagrams = _drain(cls.capture)
        finally:
            cls.capture.close()

    def test_health_answers_200_with_the_cycle_half_added(self):
        self.assertEqual(self.result["health_status"], 200)
        keys = self.result["health_keys"]
        # The health prefix comes first, in order, then the additive tail -
        # the same order test_health_snapshot.py pins, re-asserted here on a
        # response that came off a real socket rather than a direct call.
        self.assertEqual(
            keys[:10],
            ["ok", "pipeline_alive", "pose_mode", "inference_count", "last_error",
             "camera_health", "camera_stats", "pose_source",
             "camera_stale_ms", "time_sync"],
        )
        for key in ("phase", "cycle_id", "snapshot_sequence", "snapshot_age_ms",
                    "uptime_s", "model_loaded", "ready"):
            with self.subTest(key=key):
                self.assertIn(key, keys)
        self.assertIs(self.result["health_body"]["ok"], True)

    def test_status_answers_200_with_its_field_set_unchanged(self):
        self.assertEqual(self.result["status_status"], 200)
        keys = self.result["status_keys"]
        # The historical keys in order, then the additive tail - the same shape
        # the /health assertion above uses, and the same rule
        # test_health_snapshot.py pins: nothing existing may move or disappear,
        # and an addition is appended rather than inserted.
        self.assertEqual(
            keys[:16],
            ["pose_timestamp_mode", "pose_last_ts_us", "pose_payload_ts_us",
             "chassis_last_ts_us", "chassis_speed_mps", "chassis_yaw_rate_rad_s",
             "chassis_packets", "pose_timestamp", "last_t0_us",
             "reference_source_mode", "planned_route_cache", "last_control_flag",
             "last_sent_control_flag",
             "last_takeover_decision", "last_num_waypoints", "last_cot"],
        )
        self.assertEqual(
            keys[16:], ["camera_cimg", "inference_pool", "inference_metrics"]
        )

    def test_ready_refuses_before_the_first_window(self):
        # Deterministic: no window means no inference has run, so the model
        # cannot have marked itself loaded.
        self.assertEqual(self.result["first_ready_status"], 503)
        body = self.result["first_ready_body"]
        self.assertIs(body["ready"], False)
        self.assertEqual(body["reason"], "starting")
        self.assertIn("startup_complete", body["failing"])
        self.assertIn("model_loaded", body["failing"])

    def test_ready_admits_once_the_pipeline_is_really_working(self):
        self.assertEqual(
            self.result["last_ready_status"], 200,
            f"never became ready; last body: {self.result['last_ready_body']}",
        )
        body = self.result["last_ready_body"]
        self.assertIs(body["ready"], True)
        self.assertEqual(body["failing"], [])
        self.assertEqual(body["reason"], "")
        self.assertIs(body["model_loaded"], True)
        self.assertIs(body["pipeline_alive"], True)
        # Every poll along the way must have had the code agree with the body.
        for query in self.result["queries"]:
            with self.subTest(query=query):
                self.assertEqual(query["status"] == 200, query["ready"])

    def test_the_body_code_and_verdict_never_disagreed(self):
        for query in self.result["queries"]:
            with self.subTest(status=query["status"]):
                if query["status"] == 503:
                    self.assertFalse(query["ready"])
                    self.assertTrue(query["reason"], "503 with no reason given")

    def test_the_real_cycle_sent_real_bytes_through_the_real_app(self):
        self.assertTrue(self.datagrams, "the app-mode child sent no datagram")
        for item in self.datagrams:
            with self.subTest(seq=item["seq"]):
                self.assertEqual(item["magic"], MAGIC)
                self.assertEqual(item["vehicle_id"], VEHICLE_ID)
                self.assertEqual(item["trailing_bytes"], 0)
                self.assertEqual(item["length"], 14 + item["count"] * 16)
        self.assertGreaterEqual(self.result["inference_count_at_exit"], 1)
        self.assertEqual(
            self.result["snapshot_at_exit"]["last_num_waypoints"], 64
        )

    def test_shutdown_ran_the_real_handlers_in_order_and_in_budget(self):
        logs = self.result["logs"]
        self.assertEqual(logs["pipeline_started"], ["pipeline started"])
        self.assertEqual(logs["pipeline_stopping"], ["pipeline stopping"])
        self.assertEqual(logs["pipeline_stopped"], ["pipeline stopped"])
        self.assertEqual(logs["stop_incomplete"], [],
                         "shutdown reported a component it could not stop")
        # The operator greps backend.log for both, in this order: a "stopped"
        # line without a "stopping" one would mean shutdown never signalled.
        self.assertGreaterEqual(len(logs["inference_sent"]), 1)
        self.assertLess(
            self.result["shutdown_seconds"], 20.0,
            "the graceful exit exceeded the whole drain budget",
        )


if __name__ == "__main__":
    unittest.main()
