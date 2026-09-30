"""Four-PGX scheduling: JPEG order, one pending slot, dispatch interval, timeouts.

No GPU, no vehicle, no remote machine.  The local slot is the existing fake
worker.  Network slots talk to a scripted transport or to a loopback HTTP
server that never loads Alpamayo.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np

from _cv2_isolation import cv2_stub
from _fake_pool import FakeContext, make_task


def _fake_resize(frame, size, interpolation=None):
    height, width = int(size[1]), int(size[0])
    return np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)


def _fake_cvt_color(frame, code):
    return frame[:, :, ::-1] if code == 4 else frame


with cv2_stub() as _CV2_STUBBED:
    if _CV2_STUBBED:
        import cv2 as _cv2_module

        _cv2_module.INTER_LINEAR = 1
        _cv2_module.COLOR_BGR2RGB = 4
        _cv2_module.resize = _fake_resize
        _cv2_module.cvtColor = _fake_cvt_color
    from camera.sync import CameraWindowProvider, FrameBuffer, SynchronizedGroup

from camera.cimg import IMAGE_HEADER, MAGIC, PACKET_HEADER
from camera.cimg_source import CimgCameraSource, CimgConfig
from config.loader import InferencePoolConfig, load_inference_pool_config
from inference.fake_worker import fake_worker_main
from inference.gpu_pool import GpuScheduler, SUBMIT_DISPATCHED, SUBMIT_QUEUED, SUBMIT_REPLACED
from inference.http_worker import InferenceService, make_server
from inference.metrics import InferenceMetrics
from inference.network_protocol import build_infer_body
from inference.result_manager import ResultManager
from inference.tasks import WORKER_BUSY, WORKER_READY, WORKER_UNREACHABLE
from time_sync import TimestampManager

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (1, 2, 3, 4)
NET_A = "http://10.0.0.2:8092"
NET_B = "http://10.0.0.3:8092"
NET_C = "http://10.0.0.4:8092"


class ManualClock:
    def __init__(self, now: float = 10_000.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def ns(self) -> int:
        return int(self.now * 1_000_000_000)


class ScriptedTransport:
    """Blocks each call until the test releases it, in call order per URL."""

    def __init__(self) -> None:
        self.infer_calls: list[tuple[str, dict]] = []
        self.health_calls: list[str] = []
        self._infer: dict[str, Queue] = defaultdict(Queue)
        self._health: dict[str, Queue] = defaultdict(Queue)
        self._lock = threading.Lock()

    def infer(self, url, payload, *, connect_timeout_s, read_timeout_s):
        del connect_timeout_s, read_timeout_s
        with self._lock:
            self.infer_calls.append((url, payload))
        outcome = self._infer[url].get()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def health(self, url, *, connect_timeout_s, read_timeout_s):
        del connect_timeout_s, read_timeout_s
        with self._lock:
            self.health_calls.append(url)
        outcome = self._health[url].get()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def release_infer(self, url: str, outcome) -> None:
        self._infer[url].put(outcome)

    def release_health(self, url: str, outcome) -> None:
        self._health[url].put(outcome)

    def wait_infer(self, count: int, timeout: float = 3.0) -> None:
        self._wait(lambda: len(self.infer_calls) >= count, timeout)

    def wait_health(self, count: int, timeout: float = 3.0) -> None:
        self._wait(lambda: len(self.health_calls) >= count, timeout)

    def _wait(self, predicate, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("scripted transport was not called in time")


def jpeg_task(task_seq: int, **kwargs):
    task = make_task(task_seq, **kwargs)
    window = task.window
    cameras = int(window.frames.shape[0])
    frames = int(window.frames.shape[1])
    jpegs = [
        [f"jpeg-c{camera}-t{frame}-s{task_seq}".encode() for frame in range(frames)]
        for camera in range(cameras)
    ]
    return replace(task, window=replace(window, jpeg_bytes=jpegs))


def ok_body(task_seq: int) -> dict:
    return {
        "ok": True,
        "task_seq": int(task_seq),
        "infer_ms": 12.0,
        "positions": [[float(index), 1.0, 0.0] for index in range(4)],
        "rotations": [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] for _ in range(4)
        ],
        "t0_us": 0,
        "cot": "",
        "ref_pose": {
            "timestamp_us": 0,
            "latitude": 1.0,
            "longitude": 2.0,
            "altitude": 3.0,
            "heading_deg": 4.0,
        },
    }


class Pool:
    def __init__(
        self,
        *,
        devices: tuple[int, ...] = (0,),
        network: tuple[str, ...] = (NET_A, NET_B, NET_C),
        dispatch_interval_ms: float = 0.0,
        infer_timeout_s: float = 30.0,
        health_interval_ms: float = 500.0,
        clock: ManualClock | None = None,
    ) -> None:
        self.clock = clock or ManualClock()
        self.gates: dict[int, threading.Event] = {}
        self.transport = ScriptedTransport()
        self.accepted: list[int] = []
        self.config = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=devices,
            latest_only=True,
            pending_capacity=1,
            minimum_ready_workers=1,
            dispatch_interval_ms=dispatch_interval_ms,
            network_workers=network,
            network_connect_timeout_s=1.0,
            network_infer_timeout_s=infer_timeout_s,
            network_health_interval_ms=health_interval_ms,
            worker_start_timeout_s=180.0,
        )
        self.scheduler = GpuScheduler(
            settings=SimpleNamespace(name="fake-settings"),
            pool_config=self.config,
            metrics=InferenceMetrics(),
            result_manager=ResultManager(),
            on_accepted=lambda result, task: self.accepted.append(int(result.task_seq)),
            context=FakeContext(),
            worker_entry=fake_worker_main,
            worker_options={"gates": self.gates},
            clock=self.clock,
            clock_ns=self.clock.ns,
            transport=self.transport,
        )

    def start(self) -> None:
        self.scheduler.start()
        self.pump(lambda: self._local_ready())
        for url in self.config.network_workers:
            self.transport.release_health(url, {"ready": True, "state": "ready"})
        expected = len(self.config.gpu_devices) + len(self.config.network_workers)
        self.pump(lambda: self.scheduler.ready_worker_count == expected)

    def submit(self, task_seq: int, **kwargs) -> str:
        self.gates.setdefault(int(task_seq), threading.Event())
        return self.scheduler.submit(jpeg_task(task_seq, **kwargs))

    def pump(self, predicate=None, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            self.scheduler.tick()
            if predicate is not None and predicate():
                return True
            if time.monotonic() >= deadline:
                return predicate is None
            time.sleep(0.002)

    def stop(self) -> None:
        for gate in self.gates.values():
            gate.set()
        for url in self.config.network_workers:
            self.transport.release_infer(url, ok_body(0))
            self.transport.release_health(url, {"ready": False, "state": "busy"})
        self.scheduler.stop(drain_s=0.5, terminate_s=0.5)

    def _local_ready(self) -> bool:
        local = [h for h in self.scheduler._handles if h.kind == "local"]
        return bool(local) and all(h.state == WORKER_READY for h in local)

    def handle(self, url: str):
        return next(h for h in self.scheduler._handles if h.endpoint == url)


class JpegOrderTests(unittest.TestCase):
    def test_cimg_jpeg_stays_aligned_with_frame_id_and_timestamp(self) -> None:
        config = CimgConfig(port=0, frame_width=2, frame_height=2)
        lock = threading.RLock()
        buffers = [
            FrameBuffer(max_size=config.queue_frames, lock=lock)
            for _ in range(config.camera_count)
        ]
        source = CimgCameraSource(
            buffers,
            config=config,
            timestamp_manager=TimestampManager(),
            decoder=_decode_stub,
            group_lock=lock,
            monotonic_ns=_stepping_clock(),
        )
        base_ns = (time.time_ns() // 1_000_000_000 - 60) * 1_000_000_000
        originals: dict[tuple[int, int], bytes] = {}
        groups = []
        for step, sequence in enumerate((10, 11)):
            jpegs = {
                camera: _jpeg(camera, sequence) for camera in CAMERAS
            }
            for camera, payload in jpegs.items():
                originals[(camera, sequence)] = payload
            capture_ns = {
                camera: base_ns + step * 200_000_000 + camera * 1000
                for camera in CAMERAS
            }
            source.ingest_bytes(
                _packet(base_ns + step * 200_000_000, sequence, jpegs, capture_ns)
            )
            packets = {}
            for slot, buffer in enumerate(buffers):
                with buffer._lock:
                    entry = buffer._queue[-1]
                packets[slot] = entry
                self.assertEqual(entry.sequence, sequence)
                self.assertEqual(entry.jpeg_bytes, originals[(entry.camera_id, sequence)])
                self.assertEqual(entry.timestamp_us, capture_ns[entry.camera_id] // 1000)
                self.assertEqual(
                    entry.frame_id,
                    f"cam{config.camera_indices[slot]}:s{sequence}",
                )
            groups.append(
                SynchronizedGroup(
                    target_timestamp_us=min(entry.timestamp_us for entry in packets.values()),
                    packets=packets,
                    max_skew_us=0,
                )
            )

        provider = object.__new__(CameraWindowProvider)
        provider.frames_per_camera = 2
        provider.camera_indices = list(config.camera_indices)
        provider._groups = groups
        provider.target_size = config.target_size
        provider.window_builds = 0
        provider.last_window_built_at = 0.0
        provider._snapshot_dir = ""
        window = provider._build()
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window.frames.shape[0], len(window.jpeg_bytes or []))
        self.assertEqual(window.frames.shape[1], len((window.jpeg_bytes or [[]])[0]))

        by_id = {}
        for group in groups:
            for entry in group.packets.values():
                by_id[entry.frame_id] = entry
        self.assertEqual(len(by_id), 8)
        for row, frame_ids in enumerate(window.frame_ids or []):
            self.assertEqual(len(frame_ids), len(window.jpeg_bytes[row]))
            self.assertEqual(len(frame_ids), len(window.frame_timestamps_us[row]))
            stamps = window.frame_timestamps_us[row]
            self.assertLess(stamps[0], stamps[1])
            for col, frame_id in enumerate(frame_ids):
                entry = by_id[frame_id]
                self.assertEqual(window.jpeg_bytes[row][col], entry.jpeg_bytes)
                self.assertEqual(window.frame_timestamps_us[row][col], entry.timestamp_us)
                self.assertEqual(
                    entry.jpeg_bytes, originals[(entry.camera_id, entry.sequence)]
                )

        frozen = replace(window, jpeg_bytes=[list(row) for row in window.jpeg_bytes])
        from inference.tasks import freeze_window

        copied = freeze_window(frozen)
        frozen.jpeg_bytes[0][0] = b"mutated"
        self.assertNotEqual(copied.jpeg_bytes[0][0], b"mutated")
        self.assertEqual(copied.jpeg_bytes[0][0], window.jpeg_bytes[0][0])


class FourSlotTests(unittest.TestCase):
    def test_four_slots_schedule_and_pending_stays_one(self) -> None:
        pool = Pool()
        try:
            pool.start()
            self.assertEqual(
                [h.kind for h in pool.scheduler._handles],
                ["local", "network", "network", "network"],
            )
            for handle in pool.scheduler._handles:
                if handle.kind == "network":
                    self.assertIsNone(handle.task_queue)
            for seq in (1, 2, 3, 4):
                self.assertEqual(pool.submit(seq), SUBMIT_DISPATCHED)
                self.assertLessEqual(pool.scheduler.pending_count, 1)
            self.assertEqual(pool.scheduler.busy_worker_count, 4)
            self.assertEqual(pool.scheduler.pending_count, 0)
            pool.transport.wait_infer(3)
            body = pool.transport.infer_calls[0][1]
            self.assertNotIn("frames", body)
            task = pool.handle(NET_A).task
            self.assertIsNotNone(task)
            assert task is not None
            self._assert_body_matches_window(body, task.window)

            self.assertEqual(pool.submit(5), SUBMIT_QUEUED)
            self.assertEqual(pool.scheduler.pending_count, 1)
            self.assertEqual(pool.submit(6), SUBMIT_REPLACED)
            self.assertEqual(pool.scheduler.pending_count, 1)
            self.assertEqual(pool.scheduler.pending_task_seq, 6)

            class _Boom:
                def is_alive(self):
                    raise AssertionError("network slot called process.is_alive()")

            for handle in pool.scheduler._handles:
                if handle.kind == "network":
                    handle.process = _Boom()
            pool.scheduler.tick()
        finally:
            for handle in pool.scheduler._handles:
                if handle.kind == "network":
                    handle.process = None
            pool.stop()

    def _assert_body_matches_window(self, body: dict, window) -> None:
        import base64

        self.assertEqual(body["frame_ids"], window.frame_ids)
        self.assertEqual(body["frame_timestamps_us"], window.frame_timestamps_us)
        for row, jpegs in enumerate(window.jpeg_bytes):
            self.assertEqual(len(body["jpeg_b64"][row]), len(jpegs))
            for col, jpeg in enumerate(jpegs):
                self.assertEqual(base64.b64decode(body["jpeg_b64"][row][col]), jpeg)
                self.assertEqual(body["frame_ids"][row][col], window.frame_ids[row][col])
                self.assertEqual(
                    body["frame_timestamps_us"][row][col],
                    window.frame_timestamps_us[row][col],
                )


class DispatchIntervalTests(unittest.TestCase):
    def test_submit_and_fill_both_wait_out_the_interval(self) -> None:
        clock = ManualClock()
        pool = Pool(
            devices=(0, 1),
            network=(),
            dispatch_interval_ms=100.0,
            clock=clock,
        )
        try:
            pool.start()
            self.assertEqual(pool.submit(1), SUBMIT_DISPATCHED)
            # A second worker is idle, and the interval still holds the task.
            self.assertEqual(pool.submit(2), SUBMIT_QUEUED)
            self.assertEqual(pool.scheduler.pending_count, 1)
            self.assertIsNone(pool.scheduler._handles[1].task)
            clock.now += 0.2
            pool.scheduler.tick()
            self.assertEqual(pool.scheduler._handles[1].task.task_seq, 2)
            self.assertEqual(pool.scheduler.pending_count, 0)
        finally:
            pool.stop()

    def test_fill_idle_does_not_dispatch_inside_the_interval(self) -> None:
        clock = ManualClock()
        pool = Pool(
            devices=(0,),
            network=(),
            dispatch_interval_ms=100.0,
            clock=clock,
        )
        try:
            pool.start()
            self.assertEqual(pool.submit(1), SUBMIT_DISPATCHED)
            self.assertEqual(pool.submit(2), SUBMIT_QUEUED)
            pool.gates[1].set()
            pool.pump(lambda: pool.scheduler._handles[0].task is None)
            self.assertEqual(pool.scheduler.pending_task_seq, 2)
            self.assertEqual(pool.scheduler._handles[0].state, WORKER_READY)
            clock.now += 0.05
            pool.scheduler.tick()
            self.assertIsNone(pool.scheduler._handles[0].task)
            self.assertEqual(pool.scheduler.pending_count, 1)
            clock.now += 0.2
            pool.scheduler.tick()
            self.assertEqual(pool.scheduler._handles[0].task.task_seq, 2)
        finally:
            pool.stop()


class NetworkFailureTests(unittest.TestCase):
    def test_timeout_does_not_redispatch_until_health_is_ready(self) -> None:
        clock = ManualClock()
        pool = Pool(
            devices=(0,),
            network=(NET_A,),
            infer_timeout_s=5.0,
            health_interval_ms=1000.0,
            clock=clock,
        )
        try:
            pool.start()
            self.assertEqual(pool.submit(1), SUBMIT_DISPATCHED)
            self.assertEqual(pool.submit(2), SUBMIT_DISPATCHED)
            self.assertEqual(pool.submit(3), SUBMIT_QUEUED)
            pool.transport.wait_infer(1)
            calls = len(pool.transport.infer_calls)
            clock.now += 5.0
            pool.scheduler.tick()
            slot = pool.handle(NET_A)
            self.assertEqual(slot.state, WORKER_UNREACHABLE)
            self.assertIsNone(slot.task)
            self.assertEqual(pool.scheduler.pending_task_seq, 3)
            self.assertEqual(len(pool.transport.infer_calls), calls)
            pool.scheduler.tick()
            self.assertEqual(slot.state, WORKER_UNREACHABLE)
            self.assertEqual(len(pool.transport.infer_calls), calls)

            pool.transport.release_health(NET_A, {"ready": True, "state": "busy"})
            pool.pump(lambda: slot.health_inflight is False)
            self.assertEqual(slot.state, WORKER_UNREACHABLE)
            self.assertEqual(len(pool.transport.infer_calls), calls)

            clock.now += 1.0
            pool.scheduler.tick()
            pool.transport.release_health(NET_A, {"ready": True, "state": "ready"})
            pool.pump(lambda: slot.state == WORKER_READY)
            pool.pump(lambda: len(pool.transport.infer_calls) == calls + 1)
            self.assertEqual(slot.state, WORKER_BUSY)
            self.assertEqual(slot.task.task_seq, 3)
        finally:
            pool.stop()

    def test_a_late_result_does_not_clear_the_new_task(self) -> None:
        clock = ManualClock()
        pool = Pool(
            devices=(0,),
            network=(NET_A,),
            infer_timeout_s=5.0,
            health_interval_ms=1000.0,
            clock=clock,
        )
        try:
            pool.start()
            pool.submit(1)
            pool.submit(2)
            pool.transport.wait_infer(1)
            clock.now += 5.0
            pool.scheduler.tick()
            slot = pool.handle(NET_A)
            self.assertEqual(slot.state, WORKER_UNREACHABLE)
            clock.now += 1.0
            pool.scheduler.tick()
            pool.transport.release_health(NET_A, {"ready": True, "state": "ready"})
            pool.pump(lambda: slot.state == WORKER_READY)
            self.assertEqual(pool.submit(3), SUBMIT_DISPATCHED)
            pool.transport.wait_infer(2)
            pool.transport.release_infer(NET_A, ok_body(2))
            pool.pump(lambda: pool.scheduler.result_manager.count("task_seq_mismatch") == 1
                      or pool.scheduler.result_manager.count("unknown_task") == 1)
            self.assertEqual(slot.state, WORKER_BUSY)
            self.assertIsNotNone(slot.task)
            self.assertEqual(slot.task.task_seq, 3)
            self.assertNotIn(2, pool.accepted)
        finally:
            pool.stop()

    def test_one_network_worker_dropping_does_not_stop_the_others(self) -> None:
        pool = Pool(devices=(0,), network=(NET_A, NET_B), infer_timeout_s=30.0)
        try:
            pool.start()
            pool.submit(1)
            pool.submit(2)
            pool.submit(3)
            pool.transport.wait_infer(2)
            pool.transport.release_infer(NET_B, ConnectionError("connection reset"))
            pool.pump(lambda: pool.handle(NET_B).state == WORKER_UNREACHABLE)
            pool.gates[1].set()
            self.assertTrue(pool.pump(lambda: 1 in pool.accepted))
            pool.transport.release_infer(NET_A, ok_body(2))
            self.assertTrue(pool.pump(lambda: 2 in pool.accepted))
            self.assertNotIn(3, pool.accepted)
            self.assertEqual(pool.handle(NET_B).state, WORKER_UNREACHABLE)
            self.assertIsNone(pool.handle(NET_B).task)
            self.assertEqual(pool.submit(4), SUBMIT_DISPATCHED)
            self.assertNotEqual(pool.scheduler._handles[0].task, None)
            self.assertIsNone(pool.handle(NET_B).task)
            self.assertEqual(pool.handle(NET_A).state, WORKER_READY)
        finally:
            pool.stop()


class HttpWorkerTests(unittest.TestCase):
    def test_health_is_not_ready_while_infer_is_running(self) -> None:
        runner = _HoldRunner()
        service = InferenceService(
            decode=lambda jpeg, size: np.zeros((size[1], size[0], 3), np.uint8),
            to_rgb=lambda frame: frame,
        )
        service.set_runner(runner)
        server = make_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base = f"http://{host}:{port}"
        task = jpeg_task(7)
        window = task.window
        ids = [[f"cam{row}:s{col}" for col in range(window.frames.shape[1])]
               for row in range(window.frames.shape[0])]
        times = [[1_700_000_000_000_000 + row * 10 + col
                  for col in range(window.frames.shape[1])]
                 for row in range(window.frames.shape[0])]
        task = replace(
            task,
            window=replace(window, frame_ids=ids, frame_timestamps_us=times),
        )
        body = build_infer_body(task)
        self.assertNotIn("frames", body)
        try:
            self.assertEqual(_request("GET", base + "/health")[1]["state"], "ready")
            holder: dict = {}

            def post() -> None:
                holder["result"] = _request("POST", base + "/infer", body)

            worker = threading.Thread(target=post)
            worker.start()
            self.assertTrue(runner.started.wait(3))
            health = _request("GET", base + "/health")[1]
            self.assertFalse(health["ready"])
            self.assertEqual(health["state"], "busy")
            status, payload = _request("POST", base + "/infer", body)
            self.assertEqual(status, 409)
            self.assertEqual(payload["state"], "busy")
            runner.release.set()
            worker.join(3)
            self.assertEqual(holder["result"][0], 200)
            self.assertTrue(holder["result"][1]["ok"])
            self.assertEqual(_request("GET", base + "/health")[1]["state"], "ready")
            seen = runner.windows[0]
            self.assertEqual(seen.jpeg_bytes, task.window.jpeg_bytes)
            self.assertEqual(seen.frame_ids, ids)
            self.assertEqual(seen.frame_timestamps_us, times)
            self.assertEqual(tuple(seen.frames.shape[:2]), (len(ids), len(ids[0])))
        finally:
            runner.release.set()
            server.shutdown()
            server.server_close()
            thread.join(3)


class DispatchConfigTests(unittest.TestCase):
    def test_environment_overrides_json_and_json_overrides_zero(self) -> None:
        document = {
            "inference_pool": {
                "mode": "inline",
                "dispatch_interval_ms": 25,
                "network_workers": ["http://10.0.0.2:8092"],
                "network_connect_timeout_s": 1.5,
                "network_infer_timeout_s": 6,
                "network_health_interval_ms": 250,
                "minimum_ready_workers": 1,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with _env(ALPAMAYO_CONFIG=str(path)):
                loaded = load_inference_pool_config()
            self.assertEqual(loaded.dispatch_interval_ms, 25)
            self.assertEqual(loaded.network_workers, ("http://10.0.0.2:8092",))
            self.assertEqual(loaded.network_connect_timeout_s, 1.5)
            self.assertEqual(loaded.network_infer_timeout_s, 6)
            self.assertEqual(loaded.network_health_interval_ms, 250)
            with _env(
                ALPAMAYO_CONFIG=str(path),
                ALPAMAYO_DISPATCH_INTERVAL_MS="40",
                ALPAMAYO_NETWORK_INFER_TIMEOUT_S="9",
            ):
                overridden = load_inference_pool_config()
            self.assertEqual(overridden.dispatch_interval_ms, 40)
            self.assertEqual(overridden.network_infer_timeout_s, 9)
            self.assertEqual(overridden.network_connect_timeout_s, 1.5)

        omitted = {
            "inference_pool": {"mode": "inline", "minimum_ready_workers": 1}
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(omitted), encoding="utf-8")
            with _env(ALPAMAYO_CONFIG=str(path)):
                defaults = load_inference_pool_config()
        self.assertEqual(defaults.dispatch_interval_ms, 0)
        self.assertEqual(defaults.network_workers, ())

    def test_repo_config_keeps_minimum_ready_workers_at_one(self) -> None:
        with _env():
            loaded = load_inference_pool_config()
        self.assertEqual(loaded.minimum_ready_workers, 1)
        self.assertEqual(loaded.dispatch_interval_ms, 0)
        text = (ROOT / "config" / "config.json").read_text(encoding="utf-8")
        self.assertIn('"minimum_ready_workers": 1', text)

    def test_one_ready_worker_is_allowed_beside_three_network_slots(self) -> None:
        config = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=(0,),
            network_workers=(NET_A, NET_B, NET_C),
            minimum_ready_workers=1,
        )
        self.assertIsNone(config.validate())
        too_many = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=(0,),
            network_workers=(NET_A, NET_B, NET_C),
            minimum_ready_workers=5,
        )
        self.assertIn("minimum_ready_workers", too_many.validate() or "")


class _Env:
    def __init__(self, **values: str) -> None:
        self.values = values
        self._previous: dict[str, str | None] = {}

    def __enter__(self):
        names = (
            "ALPAMAYO_CONFIG",
            "ALPAMAYO_INFERENCE_MODE",
            "ALPAMAYO_GPU_DEVICES",
            "ALPAMAYO_MINIMUM_READY_WORKERS",
            "ALPAMAYO_DISPATCH_INTERVAL_MS",
            "ALPAMAYO_NETWORK_WORKERS",
            "ALPAMAYO_NETWORK_CONNECT_TIMEOUT_S",
            "ALPAMAYO_NETWORK_INFER_TIMEOUT_S",
            "ALPAMAYO_NETWORK_HEALTH_INTERVAL_MS",
        )
        for name in names:
            self._previous[name] = os.environ.get(name)
            os.environ.pop(name, None)
        os.environ.update(self.values)
        return self

    def __exit__(self, *exc) -> None:
        for name, previous in self._previous.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _env(**values: str) -> _Env:
    return _Env(**values)


def _jpeg(camera_id: int, sequence: int) -> bytes:
    body = bytes([camera_id]) + struct.pack("!I", sequence)
    return b"\xff\xd8" + body + b"\x00" * 32 + b"\xff\xd9"


def _packet(group_ns: int, sequence: int, jpegs: dict[int, bytes], capture_ns: dict[int, int]) -> bytes:
    records = b"".join(
        IMAGE_HEADER.pack(camera, 0, sequence, capture_ns[camera], len(jpegs[camera]))
        + jpegs[camera]
        for camera in CAMERAS
    )
    return PACKET_HEADER.pack(MAGIC, 1, 4, group_ns, len(records)) + records


def _decode_stub(jpeg: bytes, size: tuple[int, int]):
    if jpeg[:2] != b"\xff\xd8":
        return None
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def _stepping_clock(step_ns: int = 100_000):
    counter = [0]

    def tick() -> int:
        counter[0] += step_ns
        return counter[0]

    return tick


def _request(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read().decode("utf-8"))


class _HoldRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.windows = []

    def infer(self, window, ego_xyz, ego_rot, ref_pose):
        del ego_xyz, ego_rot
        self.windows.append(window)
        self.started.set()
        self.release.wait(5)
        return SimpleNamespace(
            cot="",
            positions=np.zeros((4, 3)),
            rotations=np.zeros((4, 3, 3)),
            t0_us=int(window.t0_us),
            ref_pose=ref_pose,
        )


if __name__ == "__main__":
    unittest.main()
