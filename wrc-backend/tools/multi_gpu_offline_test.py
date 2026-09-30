#!/usr/bin/env python3
"""Offline load-balancing test for the multi-GPU pool.  No vehicle, no control link.

Three scenarios, in increasing order of how much of the real stack they touch.
Every one of them can be run on the deployment machine before the vehicle is
involved, and two of them run on a laptop with no GPU at all.

``scheduler``  (default)
    The real ``GpuScheduler`` over the real ``spawn`` context, the real
    ``ResultManager`` and the real metrics, with the stand-in worker replacing
    only the model call.  This is where the scheduling properties are proved:
    the pending slot never grows into a backlog, a newer task always displaces
    an older one, and a result that loses the race is dropped rather than sent.

    What it does *not* exercise: the model, CUDA across a process boundary, the
    trajectory crop, the coordinate transform and the UDP socket.  It measures
    ``raw_inference_hz`` and ``accepted_result_hz``; the third frequency,
    ``udp_output_hz``, is 0 here because nothing sends.  ``accepted_nonempty_hz``
    - accepted results that carry a usable trajectory - is printed as the
    scheduler-side stand-in, and the report says so rather than quietly
    presenting it as the UDP rate.

``pipeline``
    The real ``PipelineRunner`` in ``multi_process`` mode, built but never
    ``start()``ed: no cameras, no pose listener, no vehicle link.  Tasks are
    submitted to the runner's own scheduler and the runner's own
    ``on_accepted`` callback drives the real post-processing tail - takeover
    decision, latency crop, local-to-WGS84, UDP send, inference package.  A
    loopback sink decodes the datagrams off the wire and counts them.

    This is the only mode that measures ``udp_output_hz`` and the only one that
    can prove the non-empty-trajectory output rate the acceptance criteria are
    written against.  Needs ``cv2`` and the real checkpoint, so it runs on the
    deployment machine, not in CI.

``faults``
    Every way a result can be refused, injected through a real pool: the real
    ``GpuScheduler`` over real ``spawn``ed worker processes, with the faults
    pushed across the process boundary where a real model or a real vehicle
    clock would produce them.  A duplicate report, a correct result that
    arrives too late, a capture clock that rolls back or leaps or falls behind,
    a trajectory that is NaN, Inf, mis-shaped or empty, a worker that dies or
    OOMs or raises, and a newest-wins pending slot under sustained overload.
    Fake workers throughout - no model, no GPU - so it runs while a soak holds
    the devices.  ``--real`` is refused: it would mean loading a 10B checkpoint
    to ask it to return NaN.

``--real``
    With any mode, use the actual ``worker_main``: every worker loads a
    complete Alpamayo 1.5 instance on its own device.  Without it, the stand-in
    worker keeps the GPU out of the loop, which is what makes the scheduling
    properties testable at a rate the model cannot reach.

Nothing here binds a vehicle-facing port or sends to the vehicle.  The sink
binds 127.0.0.1, and ``--scenario pipeline`` points the runner's sender at it.
With ``--scenario pipeline`` and no ``--package-dir``, no inference package is
written: the JPEG set per inference is a side effect this tool opts in to, not
a surprise on a full disk.

Usage
    python tools/multi_gpu_offline_test.py
    python tools/multi_gpu_offline_test.py --scenario workers --devices 1,3,4,5
    python tools/multi_gpu_offline_test.py --scenario workers --device-sets "1;1,3;1,3,4;1,3,4,5"
    python tools/multi_gpu_offline_test.py --scenario order
    python tools/multi_gpu_offline_test.py --scenario faults --devices 1,3,4,5
    python tools/multi_gpu_offline_test.py --scenario pipeline --devices 1,3,4,5 --real

Exit status is 0 only when every assertion in the selected scenario holds.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camera.models import CameraWindow  # noqa: E402
from config.loader import InferencePoolConfig  # noqa: E402
from inference.gpu_pool import (  # noqa: E402
    GpuInventory,
    GpuScheduler,
    probe_gpu_inventory,
    worker_main as real_worker_main,
)
from inference.metrics import Frequency, InferenceMetrics  # noqa: E402
from inference.result_manager import ResultManager  # noqa: E402
from inference.tasks import InferenceTask  # noqa: E402
from vehicle.models import (  # noqa: E402
    GlobalPose,
    PoseSample,
    vehicle_azimuth_for_compass,
)

#: The four Alpamayo camera indices in physical role order.  Carried by every
#: synthetic task so the camera mapping is exercised rather than stubbed.
CAMERA_INDICES = [1, 2, 0, 4]

#: Camera frames in a window, and the ego-history length the model takes.
FRAMES_PER_CAMERA = 4
HISTORY_STEPS = 16

#: The capture interval the synthetic tasks advance by.  A step backwards
#: beyond ``max_capture_lag_ms`` is a rollback; a forward gap is not.  The
#: step still looks like a real camera: 10 Hz, the rate the vehicle's frames
#: arrive at.
CAPTURE_STEP_US = 100_000

#: The measurement window for the throughput scenarios.  Long enough that a
#: 1.5 s inference gets several rounds per worker, short enough to re-run.
MEASURE_S = 12.0

#: A worker that is loading a 10B checkpoint takes minutes; the stand-in worker
#: takes milliseconds.  Either way the scenario waits for ``ready`` rather than
#: sleeping a number guessed here.
READY_TIMEOUT_S = 900.0


# --------------------------------------------------------------------------- #
# Synthetic input
# --------------------------------------------------------------------------- #


def make_window(capture_us: int) -> CameraWindow:
    """A structurally real camera window at a size that costs nothing.

    ``(N_camera, T, 3, H, W)`` uint8 - the shape the model takes.  The stand-in
    worker never reads the pixels; a test that did would be testing the decoder,
    which has its own suite.
    """
    frames = torch.zeros(
        (len(CAMERA_INDICES), FRAMES_PER_CAMERA, 3, 4, 4), dtype=torch.uint8
    )
    return CameraWindow(
        t0_us=int(capture_us),
        frames=frames,
        camera_indices=list(CAMERA_INDICES),
        timestamps_us=[int(capture_us)] * FRAMES_PER_CAMERA,
        frame_ids=[[f"cam{i}:t"] * FRAMES_PER_CAMERA for i in CAMERA_INDICES],
        frame_timestamps_us=[
            [int(capture_us)] * FRAMES_PER_CAMERA for _ in CAMERA_INDICES
        ],
        frame_arrivals_us=[
            [int(capture_us)] * FRAMES_PER_CAMERA for _ in CAMERA_INDICES
        ],
    )


def make_task(task_seq: int, capture_us: int, *, cycle_id: int = 0) -> InferenceTask:
    """One complete frozen task, built the way the pipeline builds one.

    Goes through the real ``InferenceTask.from_build`` rather than constructing
    the dataclass directly, so the object that crosses the process boundary in
    the ``spawn`` scenarios is the same shape the loop would have frozen.
    """
    window = make_window(capture_us)
    reference = PoseSample(
        timestamp_us=int(capture_us),
        latitude=30.5,
        longitude=114.3,
        altitude=20.0,
        heading_deg=90.0,
    )
    built = SimpleNamespace(
        ego_xyz=torch.zeros((1, 1, HISTORY_STEPS, 3), dtype=torch.float32),
        ego_rot=torch.zeros((1, 1, HISTORY_STEPS, 3, 3), dtype=torch.float32),
        reference_pose=reference,
        reference_payload_timestamp_us=int(capture_us),
        ego_history=[
            {"timestamp_us": int(capture_us), "x_m": 0.0, "y_m": 0.0}
            for _ in range(HISTORY_STEPS)
        ],
        camera_frame_ids=[
            {"camera_index": i, "latest_frame_id": f"cam{i}:t"} for i in CAMERA_INDICES
        ],
        camera_frame_id={f"cam{i}": f"cam{i}:t" for i in CAMERA_INDICES},
        latest_frame_ids=[f"cam{i}:t" for i in CAMERA_INDICES],
        camera_age_ms=12.5,
        camera_pose_delta_ms=8.0,
        camera_lag_ms=3.0,
    )
    return InferenceTask.from_build(task_seq, built, window, cycle_id=cycle_id)


# --------------------------------------------------------------------------- #
# Loopback sink - the only thing here that reads bytes off a socket
# --------------------------------------------------------------------------- #


class WaypointSink:
    """A UDP receiver on 127.0.0.1 that decodes the waypoint wire format.

    Binds port 0 and reports what it got, so two runs cannot collide and nothing
    depends on a port being free in advance.

    The decoder is written out here rather than imported because its whole
    purpose is to be an *independent* reading of the bytes: a sink that reused
    the sender's own struct format would agree with a broken sender.  The layout
    is the frozen one, ``<IBBIHH`` header and ``N x <dd`` payload, and this is
    the same reading ``tests/test_protocol_golden.py`` pins.
    """

    HEADER = struct.Struct("<IBBIHH")
    POINT = struct.Struct("<dd")
    MAGIC = 0x514D3031

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.2)
        self.host, self.port = self.sock.getsockname()
        self._stop = False
        self.thread = None
        self.packets = 0
        self.with_waypoints = 0
        self.release_frames = 0
        self.decode_errors: list[str] = []
        self.bad_magic = 0
        self.seq_gaps = 0
        self.last_seq: int | None = None
        self.control_flags: dict[int, int] = {}
        self.samples: list[tuple[float, int]] = []
        self.first_point: tuple[float, float] | None = None

    def start(self) -> "WaypointSink":
        import threading

        self.thread = threading.Thread(target=self._loop, name="waypoint-sink", daemon=True)
        self.thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            self._decode(data)

    def _decode(self, data: bytes) -> None:
        if len(data) < self.HEADER.size:
            self.decode_errors.append(f"short header: {len(data)} bytes")
            return
        magic, _version, control, seq, _vehicle_id, count = self.HEADER.unpack_from(data)
        if magic != self.MAGIC:
            self.bad_magic += 1
            self.decode_errors.append(f"magic=0x{magic:08X}")
            return
        expected = self.HEADER.size + count * self.POINT.size
        if len(data) != expected:
            self.decode_errors.append(
                f"count={count} implies {expected} bytes, got {len(data)}"
            )
            return
        if count:
            self.first_point = self.POINT.unpack_from(data, self.HEADER.size)
            self.with_waypoints += 1
        else:
            self.release_frames += 1
        if self.last_seq is not None and seq != ((self.last_seq + 1) & 0xFFFFFFFF):
            # Not necessarily a bug - a failed send legitimately skips a number -
            # but it is worth reporting, which is why it is counted and printed.
            self.seq_gaps += 1
        self.last_seq = seq
        self.packets += 1
        self.control_flags[control] = self.control_flags.get(control, 0) + 1
        self.samples.append((time.monotonic(), count))

    def rate_hz(self, with_waypoints: bool = False) -> float:
        """Packets per second over the whole run, from the arrival stamps."""
        if len(self.samples) < 2:
            return 0.0
        span = self.samples[-1][0] - self.samples[0][0]
        if span <= 0:
            return 0.0
        if with_waypoints:
            return self.with_waypoints / span
        return self.packets / span

    def stop(self) -> None:
        self._stop = True
        if self.thread is not None:
            self.thread.join(1.0)
        try:
            self.sock.close()
        except OSError:
            pass

    def report(self) -> dict[str, Any]:
        return {
            "packets": self.packets,
            "with_waypoints": self.with_waypoints,
            "release_frames": self.release_frames,
            "bad_magic": self.bad_magic,
            "seq_gaps": self.seq_gaps,
            "last_seq": self.last_seq,
            "control_flags": {str(k): v for k, v in sorted(self.control_flags.items())},
            "first_point": self.first_point,
            "decode_errors": self.decode_errors[:5],
            "hz_total": round(self.rate_hz(), 3),
            "hz_with_waypoints": round(self.rate_hz(with_waypoints=True), 3),
        }


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


class Harness:
    """A real scheduler plus the counters a report needs, and no vehicle."""

    def __init__(
        self,
        *,
        devices: Sequence[int],
        result_max_age_ms: float,
        max_capture_lag_ms: float,
        worker_entry: Callable[..., None],
        worker_options: dict[str, Any],
        settings: Any,  # AppConfig: preloaded, and only needed to build the pool
    ) -> None:
        self.settings = settings
        self.config = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=tuple(int(d) for d in devices),
            latest_only=True,
            pending_capacity=1,
            result_max_age_ms=float(result_max_age_ms),
            max_capture_lag_ms=float(max_capture_lag_ms),
            scheduler_poll_interval_ms=20.0,
            worker_start_timeout_s=READY_TIMEOUT_S,
            # Every worker asked for, so "one did not come up" fails the run
            # rather than being averaged away.
            minimum_ready_workers=len(devices),
        )
        problem = self.config.validate()
        if problem is not None:
            raise SystemExit(f"pool config is not usable: {problem}")

        self.metrics = InferenceMetrics(long_s=60.0)
        self.result_manager = ResultManager(
            result_max_age_ms=float(result_max_age_ms),
            max_capture_lag_ms=float(max_capture_lag_ms),
        )
        self.scheduler = GpuScheduler(
            settings=settings,
            pool_config=self.config,
            metrics=self.metrics,
            result_manager=self.result_manager,
            on_accepted=self._on_accepted,
            worker_entry=worker_entry,
            worker_options=dict(worker_options),
        )
        self.inventory: GpuInventory | None = None
        # Accepted results, and the subset that carried a usable trajectory -
        # the scheduler-side stand-in for the UDP rate, labelled as such in the
        # report so it cannot be mistaken for a measured wire rate.
        self.accepted: list[tuple[int, int, int]] = []
        self.accepted_nonempty = Frequency()
        self.submitted = 0
        self.submitted_nonempty = 0

    # -- wiring ------------------------------------------------------------- #

    def _on_accepted(self, result: Any, task: InferenceTask) -> None:
        positions = getattr(result.model_result, "positions", None)
        points = 0 if positions is None else int(np.asarray(positions).shape[0])
        self.accepted.append((int(result.task_seq), int(result.worker_id), points))
        if points > 0:
            self.accepted_nonempty.add()

    def start(self) -> None:
        self.inventory = self.scheduler.start()

    def stop(self) -> dict[str, Any]:
        return self.scheduler.stop()

    # -- driving ------------------------------------------------------------ #

    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.scheduler.tick()
            if self.scheduler.ready_worker_count >= self.config.minimum_ready_workers:
                return True
            if self.scheduler.failed_worker_count:
                return False
            time.sleep(0.02)
        return False

    def submit(self, task: InferenceTask) -> str:
        verdict = self.scheduler.submit(task)
        self.submitted += 1
        return verdict

    def pump(self, seconds: float) -> None:
        """Tick for ``seconds``, at the configured poll interval."""
        deadline = time.monotonic() + float(seconds)
        poll = self.config.poll_interval_s
        while time.monotonic() < deadline:
            self.scheduler.tick()
            time.sleep(poll)
        self.scheduler.tick()

    def drain(self, *, seconds: float = 10.0) -> None:
        """Tick until every worker is idle, or the deadline passes."""
        deadline = time.monotonic() + float(seconds)
        while time.monotonic() < deadline:
            self.scheduler.tick()
            if self.scheduler.busy_worker_count == 0 and self.scheduler.pending_count == 0:
                return
            time.sleep(self.config.poll_interval_s)

    def submit_at(self, rate_hz: float, seconds: float) -> None:
        """Submit newest-wins tasks at ``rate_hz`` for ``seconds``, ticking between.

        The gap is computed from an absolute deadline rather than by sleeping a
        fixed interval, so a slow tick does not silently turn 10 Hz into 6 Hz -
        the input rate is the independent variable in every scenario here and it
        has to be the rate that was asked for.
        """
        period = 1.0 / float(rate_hz)
        started = time.monotonic()
        deadline = started + float(seconds)
        seq = 0
        while True:
            now = time.monotonic()
            if now >= deadline:
                return
            seq += 1
            capture = CAPTURE_STEP_US * seq
            self.submit(make_task(seq, capture, cycle_id=seq))
            self.scheduler.tick()
            target = started + seq * period
            slack = target - time.monotonic()
            if slack > 0:
                time.sleep(min(slack, period))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


class Report:
    """Collects pass/fail lines and prints them as they are decided."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        return bool(ok)

    def note(self, text: str) -> None:
        print(f"  . {text}")

    @property
    def failed(self) -> list[str]:
        return [name for name, ok, _ in self.checks if not ok]

    def summary(self) -> int:
        passed = sum(1 for _, ok, _ in self.checks if ok)
        total = len(self.checks)
        print()
        if self.failed:
            print(f"RESULT: FAILED ({total - passed}/{total} checks failed)")
            for name in self.failed:
                print(f"  - {name}")
            return 1
        print(f"RESULT: OK ({passed}/{total} checks passed)")
        return 0


def pool_report(harness: Harness) -> dict[str, Any]:
    """The scheduler-side numbers, in the terms the acceptance criteria use.

    The drop breakdown is read from two places on purpose.  ``pending_replaced``
    belongs to the scheduler - it is a task the pool chose never to run - while
    every other reason is the result manager's verdict on a result that did run.
    The published view merges them; here they are kept separate so a number can
    be traced back to whoever counted it.
    """
    snapshot = harness.metrics.snapshot()
    return {
        "raw_inference_hz": snapshot["raw_inference"]["hz_10s"],
        "accepted_result_hz": snapshot["accepted_result"]["hz_10s"],
        "accepted_nonempty_hz": round(harness.accepted_nonempty.rate(10.0), 3),
        "udp_output_hz": snapshot["udp_output"]["hz_10s"],
        "raw_total": snapshot["raw_inference"]["total"],
        "accepted_total": snapshot["accepted_result"]["total"],
        "submitted": harness.submitted,
        "pending_count": harness.scheduler.pending_count,
        "scheduler_drops": snapshot["drops"],
        "result_drops": harness.result_manager.stats(),
        "faults": snapshot["faults"],
        "infer_ms": snapshot["latency_ms"]["infer"],
        "queue_wait_ms": snapshot["latency_ms"]["queue_wait"],
        "workers": [
            {
                "worker_id": w.worker_id,
                "device": w.device,
                "state": w.state,
                "infer_count": w.infer_count,
                "last_infer_ms": None if w.last_infer_ms is None else round(w.last_infer_ms, 1),
                "avg_infer_ms": None if w.avg_infer_ms is None else round(w.avg_infer_ms, 1),
                "p50_infer_ms": None if w.p50_infer_ms is None else round(w.p50_infer_ms, 1),
                "p95_infer_ms": None if w.p95_infer_ms is None else round(w.p95_infer_ms, 1),
                "infer_samples": w.infer_samples,
                "allocated_mib": round(w.memory.allocated_bytes / (1 << 20), 1),
                "reserved_mib": round(w.memory.reserved_bytes / (1 << 20), 1),
                "peak_allocated_mib": round(w.peak_allocated_bytes / (1 << 20), 1),
                "error": w.error,
            }
            for w in harness.scheduler.worker_views()
        ],
    }


def print_pool_report(title: str, report: dict[str, Any]) -> None:
    print(f"  -- {title}")
    print(
        f"     raw_inference_hz={report['raw_inference_hz']:.2f} "
        f"accepted_result_hz={report['accepted_result_hz']:.2f} "
        f"accepted_nonempty_hz={report['accepted_nonempty_hz']:.2f} "
        f"udp_output_hz={report['udp_output_hz']:.2f}"
    )
    print(
        f"     submitted={report['submitted']} raw_total={report['raw_total']} "
        f"accepted_total={report['accepted_total']} pending={report['pending_count']}"
    )
    print(
        f"     queue_wait_ms={json.dumps(report['queue_wait_ms'])}\n"
        f"     infer_ms={json.dumps(report['infer_ms'])}"
    )
    print(
        f"     scheduler_drops={json.dumps(report['scheduler_drops'])} "
        f"faults={json.dumps(report['faults'])}"
    )
    print(f"     result_drops={json.dumps(report['result_drops'])}")
    for worker in report["workers"]:
        print(
            f"     worker {worker['worker_id']} dev {worker['device']} "
            f"{worker['state']} n={worker['infer_count']} "
            f"last={worker['last_infer_ms']} avg={worker['avg_infer_ms']} "
            f"p50={worker['p50_infer_ms']} p95={worker['p95_infer_ms']} "
            f"(samples={worker['infer_samples']}) "
            f"vram_alloc={worker['allocated_mib']}MiB "
            f"reserved={worker['reserved_mib']}MiB "
            f"peak={worker['peak_allocated_mib']}MiB"
            + (f" ERROR={worker['error']}" if worker["error"] else "")
        )


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def scenario_backlog(args: argparse.Namespace) -> int:
    """10 Hz of input against a pool that can only manage ~N/1.5 Hz.

    The literal requirement: the input arrives much faster than the pool can
    consume it, and the pool must not build a backlog.  ``pending_count`` must
    stay at or below 1 for the whole run, and the displaced tasks must be
    counted as replaced rather than queued.
    """
    print(
        f"\n== scenario backlog: {args.rate_hz:g} Hz input, "
        f"{args.infer_ms:g} ms per inference, {len(args.devices)} worker(s) =="
    )
    harness = _build(args)
    report = Report()
    try:
        harness.start()
        if not harness.wait_ready():
            return _startup_failed(harness, report)
        _startup_facts(harness, report)

        # Sampled every iteration, not at the end: the property is "never more
        # than one", and a queue that grew and drained would look clean at the
        # end of a run that spent a minute with a backlog.
        observed_max_pending = 0
        started = time.monotonic()
        stop_at = started + args.seconds
        period = 1.0 / args.rate_hz
        seq = 0
        while time.monotonic() < stop_at:
            seq += 1
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
            observed_max_pending = max(
                observed_max_pending, harness.scheduler.pending_count
            )
            target = started + seq * period
            slack = target - time.monotonic()
            if slack > 0:
                time.sleep(min(slack, period))
        harness.drain(seconds=args.infer_ms / 1000.0 + 10.0)

        measured = pool_report(harness)
        print_pool_report("backlog", measured)

        capacity_hz = len(args.devices) / (args.infer_ms / 1000.0)
        report.check(
            "pending never exceeded 1",
            observed_max_pending <= 1,
            f"max observed={observed_max_pending}",
        )
        replaced = measured["scheduler_drops"].get("pending_replaced", 0)
        report.check(
            "displaced tasks were replaced, not queued",
            replaced >= 1,
            f"pending_replaced={replaced} of {harness.submitted} submissions",
        )
        report.check(
            "the pool ran at its capacity, not at the input rate",
            abs(measured["raw_inference_hz"] - capacity_hz) <= max(0.5, capacity_hz * 0.35),
            f"raw={measured['raw_inference_hz']:.2f} Hz capacity~={capacity_hz:.2f} Hz "
            f"input={args.rate_hz:g} Hz",
        )
        report.check(
            "the backlog was discarded, not inferred",
            measured["raw_total"] <= harness.submitted,
            f"raw_total={measured['raw_total']} submitted={harness.submitted}",
        )
        report.check(
            "no worker failed",
            harness.scheduler.failed_worker_count == 0,
            f"failed={harness.scheduler.failed_worker_count}",
        )
        return report.summary()
    finally:
        _shutdown(harness)


def scenario_order(args: argparse.Namespace) -> int:
    """Results arrive 1, 3, 5, 2, 4 - and only the increasing ones drive anything.

    This is the property the whole Result Manager exists for, and it is forced
    rather than hoped for: the stand-in worker is given exact per-task delays so
    the completion order is a constant of the test.
    """
    devices = list(args.devices)
    if len(devices) != 4:
        # The delays below are chosen for exactly four workers; on a different
        # count the completions would not line up and the scenario would be
        # asserting something it did not set up.
        print(f"  ! --scenario order needs exactly 4 devices, got {devices}; skipped")
        return 0
    print(f"\n== scenario order: forced completion order 1,3,5,2,4 on {devices} ==")
    # Tasks 1..4 land on the four idle workers at once.  Task 5 goes to the
    # pending slot and is picked up by whichever worker frees first, which is the
    # one running task 1.  With these durations the completions are forced:
    #   1 -> 50 ms, 3 -> 300 ms, 5 -> 50+500 ms, 2 -> 800 ms, 4 -> 1200 ms
    # so the watermark sees 1, 3, 5 and then two results that are already stale.
    delays_by_task_seq = {1: 50.0, 2: 800.0, 3: 300.0, 4: 1200.0, 5: 500.0}
    harness = _build(args, delays_by_task_seq=delays_by_task_seq)
    report = Report()
    try:
        harness.start()
        if not harness.wait_ready():
            return _startup_failed(harness, report)
        _startup_facts(harness, report)

        for seq in (1, 2, 3, 4, 5):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
        report.note(
            "dispatch: "
            + json.dumps(
                [
                    {"worker": w.worker_id, "device": w.device, "task_seq": w.task_seq}
                    for w in harness.scheduler.worker_views()
                ]
            )
        )
        harness.drain(seconds=10.0)

        accepted = [seq for seq, _worker, _points in harness.accepted]
        drops = harness.result_manager.stats()
        report.note(f"accepted, in the order the watermark allowed them: {accepted}")
        report.check(
            "only the increasing task_seqs were accepted",
            accepted == [1, 3, 5],
            f"accepted={accepted}",
        )
        report.check(
            "the two losers were dropped as out_of_order",
            drops.get("out_of_order") == 2,
            f"out_of_order={drops.get('out_of_order')}",
        )
        report.check(
            "no result overwrote a newer one",
            harness.result_manager.last_accepted_task_seq == 5,
            f"last_accepted={harness.result_manager.last_accepted_task_seq}",
        )
        report.check(
            "the inferred-but-lost results were still counted as inferences",
            harness.metrics.raw_inference.total == 5,
            f"raw_inference.total={harness.metrics.raw_inference.total} "
            "(5 tasks ran; 3 were accepted)",
        )
        return report.summary()
    finally:
        _shutdown(harness)


def _fault_case(
    label: str, args: argparse.Namespace, report: Report, **build: Any
) -> Harness | None:
    """A freshly started pool for one fault case, or ``None`` if it never came up.

    Every case builds its own pool: a fault injected into a shared pool would
    leave its counter behind, and the next case would be asserting on a number
    it did not cause.  A case that cannot start is recorded as a failure of its
    own rather than aborting the scenario, so one unstartable pool does not hide
    the verdicts of the other five.  The caller owns the ``finally`` shutdown.
    """
    print(f"\n-- fault: {label} --")
    harness = _build(args, **build)
    harness.start()
    if not harness.wait_ready():
        report.check(
            f"{label}: the pool started", False, "workers never became ready"
        )
        _shutdown(harness)
        return None
    return harness


def _accepted_seqs(harness: Harness) -> list[int]:
    return [seq for seq, _worker, _points in harness.accepted]


def _faults_duplicate(args: argparse.Namespace, report: Report) -> None:
    """The same result delivered twice: accepted once, refused once.

    A genuine duplicate cannot be manufactured by re-submitting a task - that
    would be a new ``task_seq`` and a new inference.  It has to come from the
    worker, which is what ``duplicate_task_seqs`` does: the same message object
    put on the result queue a second time, exactly as a retry or a double-report
    would look to the Result Manager.

    The four tasks are given increasing durations so they complete in sequence.
    Without that the completion order is whatever the OS decides, and the
    duplicate would usually arrive to find the watermark already past it - a
    ``duplicate`` refusal is only reachable while the copy's own sequence is
    still the newest one accepted.
    """
    harness = _fault_case(
        "duplicate",
        args,
        report,
        delays_by_task_seq={1: 100.0, 2: 300.0, 3: 500.0, 4: 700.0},
        worker_options={"duplicate_task_seqs": [2]},
    )
    if harness is None:
        return
    try:
        for seq in (1, 2, 3, 4):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
        harness.drain(seconds=10.0)

        accepted = _accepted_seqs(harness)
        stats = harness.result_manager.stats()
        report.check(
            "the duplicated task_seq was accepted exactly once",
            accepted.count(2) == 1,
            f"accepted={accepted}",
        )
        report.check(
            "the second copy was refused as a duplicate",
            stats.get("duplicate", 0) == 1,
            f"duplicate={stats.get('duplicate')}",
        )
        report.check(
            "the duplicate did not disturb the tasks around it",
            sorted(accepted) == [1, 2, 3, 4],
            f"accepted={sorted(accepted)}",
        )
    finally:
        _shutdown(harness)


def _faults_expired(args: argparse.Namespace, report: Report) -> None:
    """A correct trajectory that arrives after its usefulness has passed.

    The slow task is the *last* one, and that ordering is load-bearing: expiry
    is checked after the watermark rule, so a slow result with a low sequence
    number would be refused as ``out_of_order`` and the age check would never
    run.  Here the three quick results take 1, 2, 3 in order and leave the
    watermark at 3, so the late result's sequence is still admissible and only
    its age can disqualify it.  Its trajectory is perfectly valid throughout.
    """
    harness = _fault_case(
        "expired",
        args,
        report,
        result_max_age_ms=300.0,
        delays_by_task_seq={1: 20.0, 2: 30.0, 3: 40.0, 4: 1500.0},
    )
    if harness is None:
        return
    try:
        for seq in (1, 2, 3, 4):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
        harness.drain(seconds=15.0)

        accepted = _accepted_seqs(harness)
        stats = harness.result_manager.stats()
        report.check(
            "the slow result was refused as expired",
            stats.get("expired", 0) == 1,
            f"expired={stats.get('expired')} "
            f"(budget {harness.result_manager.result_max_age_ms:.0f} ms, "
            "task 4 took 1500 ms)",
        )
        report.check(
            "the expired task_seq never became the watermark",
            harness.result_manager.last_accepted_task_seq == 3,
            f"last_accepted={harness.result_manager.last_accepted_task_seq}",
        )
        report.check(
            "the expired task was refused for its age, not for its sequence",
            stats.get("out_of_order", 0) == 0,
            f"out_of_order={stats.get('out_of_order')} - the ordering rule "
            "would have masked the age check",
        )
        report.check(
            "the three on-time results were accepted",
            sorted(accepted) == [1, 2, 3],
            f"accepted={sorted(accepted)}",
        )
        report.check(
            "the refused result was still counted as an inference",
            harness.metrics.snapshot()["raw_inference"]["total"] == 4,
            "expiry is a delivery verdict, not a run that never happened",
        )
    finally:
        _shutdown(harness)


def _faults_capture_clock(args: argparse.Namespace, report: Report) -> None:
    """A capture clock that steps backwards, leaps forward, then falls behind.

    A backwards step beyond the tolerance is a rollback, caught at submit and
    refused when the result comes back.  A forward leap, however large, is not
    an anomaly: it becomes the new capture reference, and that task may be
    accepted.  A lag is judged later, against a newer window.
    """
    base_us = 100_000_000
    harness = _fault_case(
        "capture clock", args, report, worker_options={"delays_ms": [10.0]}
    )
    if harness is None:
        return
    try:
        harness.submit(make_task(1, base_us, cycle_id=1))
        harness.scheduler.tick()
        # 5 s backwards: beyond any tolerance, and unlike a wrap or a reordered
        # frame it cannot be explained away as clock noise.
        harness.submit(make_task(2, base_us - 5_000_000, cycle_id=2))
        harness.scheduler.tick()
        # 10 s forwards: not an anomaly.  The stamp becomes the new reference,
        # so later windows are compared with it rather than with task 1.
        harness.submit(make_task(3, base_us + 10_000_000, cycle_id=3))
        harness.scheduler.tick()
        harness.drain(seconds=10.0)

        accepted = _accepted_seqs(harness)
        drops = harness.result_manager.stats()
        report.note(
            "capture-time refusals: "
            + json.dumps(
                {
                    k: v
                    for k, v in sorted(drops.items())
                    if "capture" in k or "pose" in k
                }
            )
        )
        report.check(
            "the backwards step was refused as a rollback",
            drops.get("capture_rollback", 0) == 1,
            f"capture_rollback={drops.get('capture_rollback')}",
        )
        report.check(
            "the forward leap was not a jump and advanced the reference",
            drops.get("capture_jump", 0) == 0
            and harness.result_manager._newest_capture_us == base_us + 10_000_000,
            f"capture_jump={drops.get('capture_jump')} "
            f"newest={harness.result_manager._newest_capture_us}",
        )
        report.check(
            "the rollback stayed out and the forward leap could be accepted",
            2 not in accepted and 3 in accepted,
            f"accepted={accepted}",
        )
    finally:
        _shutdown(harness)

    # The lag case needs a fresh pool: the watermark above has been perturbed,
    # and a lag is defined against the newest window the pool would accept.
    #
    # The lagging task must be both the oldest *and* the first to be validated,
    # which is a narrower window than it looks.  The newest window advances when
    # a task is *submitted*, not when it is accepted, so submitting two more
    # windows is enough to strand an earlier one - but if those two results came
    # back first the watermark would pass the stranded task and the ordering
    # rule, checked ahead of the camera clock, would refuse it as out_of_order.
    # Hence: task 1 quick, tasks 2 and 3 slow, all three submitted back to back.
    # Task 1's result returns while the watermark is still 0, and only then can
    # its lag - 4.8 s against a 2.5 s tolerance, from steps of 2.4 s that were
    # each individually legal - be the reason it is refused.
    harness = _fault_case(
        "capture clock lag",
        args,
        report,
        result_max_age_ms=8000.0,
        delays_by_task_seq={1: 400.0, 2: 2000.0, 3: 2000.0},
    )
    if harness is None:
        return
    try:
        step_us = 2_400_000
        harness.submit(make_task(1, base_us, cycle_id=1))
        harness.submit(make_task(2, base_us + step_us, cycle_id=2))
        harness.submit(make_task(3, base_us + 2 * step_us, cycle_id=3))
        harness.drain(seconds=20.0)

        accepted = _accepted_seqs(harness)
        drops = harness.result_manager.stats()
        report.note(
            f"newest capture window "
            f"{harness.result_manager._newest_capture_us} us: "
            f"{(harness.result_manager._newest_capture_us - base_us) / 1e6:.1f} s "
            "ahead of the refused task, from 2.4 s steps"
        )
        report.check(
            "a task that legal steps left behind was refused as a lag",
            drops.get("capture_lag", 0) == 1,
            f"capture_lag={drops.get('capture_lag')} (tolerance "
            f"{harness.result_manager.max_capture_lag_ms:.0f} ms, no single step "
            "anomalous)",
        )
        report.check(
            "no step was flagged, so the lag is the only reason it was refused",
            drops.get("capture_jump", 0) == 0 and drops.get("capture_rollback", 0) == 0,
            f"jump={drops.get('capture_jump')} rollback={drops.get('capture_rollback')}",
        )
        report.check(
            "the two windows that were still current got through",
            sorted(accepted) == [2, 3],
            f"accepted={sorted(accepted)}",
        )
    finally:
        _shutdown(harness)


def _faults_corrupt(args: argparse.Namespace, report: Report) -> None:
    """Four trajectories no downstream stage may act on.

    Each worker reports success.  Every refusal therefore has to come from the
    Result Manager's own reading of the trajectory - a stage that trusted the
    worker's ``ok`` flag would accept all four.
    """
    harness = _fault_case(
        "corrupt trajectory",
        args,
        report,
        worker_options={
            "corrupt_task_seqs": {1: "nan", 2: "inf", 3: "shape", 4: "points"},
            "delays_ms": [10.0],
        },
    )
    if harness is None:
        return
    try:
        for seq in (1, 2, 3, 4):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
        harness.drain(seconds=10.0)

        accepted = _accepted_seqs(harness)
        drops = harness.result_manager.stats()
        report.note("refusals: " + json.dumps({k: v for k, v in sorted(drops.items()) if v}))
        report.check(
            "no corrupt trajectory was accepted",
            accepted == [],
            f"accepted={accepted}",
        )
        report.check(
            "NaN and Inf were refused as non-finite",
            drops.get("non_finite_trajectory", 0) == 2,
            f"non_finite_trajectory={drops.get('non_finite_trajectory')}",
        )
        report.check(
            "the wrong number of columns was refused as a shape error",
            drops.get("invalid_trajectory_shape", 0) == 1,
            f"invalid_trajectory_shape={drops.get('invalid_trajectory_shape')}",
        )
        report.check(
            "the empty trajectory was refused as a points error",
            drops.get("invalid_trajectory_points", 0) == 1,
            f"invalid_trajectory_points={drops.get('invalid_trajectory_points')}",
        )
        report.check(
            "all four were still counted as inferences",
            harness.metrics.snapshot()["raw_inference"]["total"] == 4,
            "the refusal is of the result, not of the run",
        )
    finally:
        _shutdown(harness)


def _faults_worker_faults(args: argparse.Namespace, report: Report) -> None:
    """One worker dies, one reports OOM, one raises - and three carry on.

    The failure classes are spread across three different workers on purpose:
    a pool that lost all of them to a single cause would look the same in the
    counters, but the survivability claim is about the *remaining* workers
    continuing to produce accepted results, which only this arrangement tests.

    Worker 0 leaves *abruptly* - no ``WORKER_STOPPED``, so the parent has only
    the process table to go on.  With a polite stop the scheduler is right not
    to poll, and ``worker_exited`` would stay at zero while the pool quietly
    lost a quarter of its capacity.
    """
    harness = _fault_case(
        "worker faults",
        args,
        report,
        delays_by_task_seq={1: 50.0, 2: 60.0, 3: 70.0, 4: 80.0},
        worker_options={
            "oom_task_seqs": [2],
            "fail_task_seqs": [3],
            "abrupt_exit_after_tasks_by_worker": {0: 1},
        },
    )
    if harness is None:
        return
    try:
        for seq in (1, 2, 3, 4):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
        # Long enough for the liveness poll to notice the dead worker rather
        # than for the drain to notice that nothing is busy.
        harness.pump(2.0)
        report.note(
            "after the death: "
            + json.dumps(
                [
                    {"worker": w.worker_id, "device": w.device, "state": w.state}
                    for w in harness.scheduler.worker_views()
                ]
            )
        )
        # One at a time, draining between: with three workers left and four
        # tasks offered at once, a completion-order accident would drop one as
        # out_of_order and that drop would say nothing about survivability.
        for seq in (5, 6, 7):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
            harness.drain(seconds=10.0)
        harness.drain(seconds=10.0)

        pool = pool_report(harness)
        faults = pool["faults"]
        drops = pool["result_drops"]
        accepted = _accepted_seqs(harness)
        report.note(
            "faults: "
            + json.dumps({k: v for k, v in sorted(faults.items()) if v})
            + "  result drops: "
            + json.dumps({k: v for k, v in sorted(drops.items()) if v})
        )
        report.check(
            "the dead worker was noticed by the liveness poll",
            faults.get("worker_exited", 0) == 1,
            f"worker_exited={faults.get('worker_exited')} (the worker sent no "
            "WORKER_STOPPED, so the process table was the only evidence)",
        )
        report.check(
            "the OOM was reported as an OOM, not as a generic failure",
            drops.get("cuda_oom", 0) == 1,
            f"cuda_oom={drops.get('cuda_oom')}",
        )
        report.check(
            "the raised exception was reported as a worker error",
            drops.get("worker_exception", 0) == 1,
            f"worker_exception={drops.get('worker_exception')}",
        )
        report.check(
            "neither failed task reached the watermark",
            not ({2, 3} & set(accepted)),
            f"accepted={sorted(accepted)}",
        )
        report.check(
            "the surviving workers kept producing accepted results",
            [s for s in accepted if s in (5, 6, 7)] == [5, 6, 7],
            f"accepted={sorted(accepted)} (tasks 5-7 submitted one at a time "
            "after the worker died)",
        )
        report.check(
            "the pool is degraded by exactly one worker, not dead",
            harness.scheduler.ready_worker_count == 3
            and harness.scheduler.failed_worker_count == 1,
            f"ready={harness.scheduler.ready_worker_count} "
            f"failed={harness.scheduler.failed_worker_count}",
        )
    finally:
        _shutdown(harness)


def _faults_pending(args: argparse.Namespace, report: Report) -> None:
    """Sixteen tasks offered while all four workers are busy, none of them stale.

    The pending slot is latest-only, so the pool keeps the newest task and drops
    the seven it displaced - without ever running them.  The last assertion is
    the one that matters and the one a queue-based design would fail: the
    displaced tasks were not merely ignored downstream, they were never given to
    a worker at all.

    The four tasks occupy the four workers for at least 800 ms and finish in
    sequence, so the eleven that follow all arrive at a genuinely full pool and
    none of the four is stranded behind the watermark.
    """
    delays = {1: 800.0, 2: 1000.0, 3: 1200.0, 4: 1400.0}
    harness = _fault_case(
        "pending replacement",
        args,
        report,
        delays_by_task_seq=delays,
        # The pool is deliberately busy for seconds; the default age budget
        # would refuse the surviving task for being late rather than for
        # anything this case is testing.
        result_max_age_ms=60_000.0,
    )
    if harness is None:
        return
    try:
        for seq in (1, 2, 3, 4):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            harness.scheduler.tick()
        max_pending = harness.scheduler.pending_count
        for seq in range(5, 13):
            harness.submit(make_task(seq, CAPTURE_STEP_US * seq, cycle_id=seq))
            max_pending = max(max_pending, harness.scheduler.pending_count)
        report.check(
            "the pending slot never held more than the newest task",
            max_pending == 1,
            f"max_pending={max_pending}",
        )
        harness.drain(seconds=20.0)

        pool = pool_report(harness)
        accepted = _accepted_seqs(harness)
        replaced = pool["scheduler_drops"].get("pending_replaced", 0)
        report.note(
            "scheduler drops: "
            + json.dumps({k: v for k, v in sorted(pool["scheduler_drops"].items()) if v})
        )
        report.check(
            "the seven displaced tasks were counted as replaced",
            replaced == 7,
            f"pending_replaced={replaced} (12 offered, 1 kept, 7 displaced by a "
            "newer one)",
        )
        report.check(
            "the displaced tasks were never inferred",
            pool["raw_total"] == 5,
            f"raw_inference.total={pool['raw_total']} (tasks 1-4 plus the one "
            "that survived pending)",
        )
        report.check(
            "the newest task was the one to run",
            sorted(accepted) == [1, 2, 3, 4, 12],
            f"accepted={sorted(accepted)}",
        )
    finally:
        _shutdown(harness)


def _pool_modules_are_isolated() -> tuple[bool, str]:
    """True when no pool module can reach the decision or the send side.

    ``worker_main``'s docstring claims a worker "cannot make a control decision
    even by mistake".  That is a claim about the import graph, so this is the
    check that keeps it honest: it walks the modules the pool is made of and
    fails if any of them names the pipeline, the app, waypoints or a trajectory.
    """
    forbidden = {"app", "pipeline", "trajectory", "waypoint"}
    root = Path(__file__).resolve().parents[1]
    for name in ("gpu_pool", "result_manager", "fake_worker", "tasks"):
        path = root / "inference" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden:
                        return False, f"inference/{name}.py imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in forbidden:
                    return False, f"inference/{name}.py imports from {node.module}"
    return True, "inference/{gpu_pool,result_manager,fake_worker,tasks}.py stay clear of the control path"


def scenario_faults(args: argparse.Namespace) -> int:
    """Every way a result can be refused, injected through a real pool.

    Phase 4 of the acceptance plan asks for the Result Manager's refusals to be
    exercised end to end - the real ``GpuScheduler``, real spawned worker
    processes, the real message protocol - rather than by calling ``validate``
    directly, which is what the unit tests do and what this replaces as
    evidence.

    No model and no GPU is involved: these are fake workers, so the whole
    scenario runs on the CPU while a soak keeps the devices.  The faults are the
    ones a real model and a real vehicle clock produce: a duplicate report, a
    correct-but-late result, a capture clock that rolls back or leaps, a
    trajectory that is NaN, Inf, mis-shaped or empty, a worker that dies, OOMs
    or raises, and a newest-wins pending slot under sustained overload.
    """
    if args.real:
        print(
            "\n== scenario faults: refused together with --real ==\n"
            "  These cases inject NaN trajectories and dying workers.  Running "
            "them against the real model would mean loading a 10B checkpoint to "
            "ask it to return NaN.  Re-run without --real."
        )
        return 1

    print(f"\n== scenario faults: refusal paths on {list(args.devices)} ==")
    report = Report()
    for case in (
        _faults_duplicate,
        _faults_expired,
        _faults_capture_clock,
        _faults_corrupt,
        _faults_worker_faults,
        _faults_pending,
    ):
        case(args, report)
    ok, detail = _pool_modules_are_isolated()
    report.check("no pool module can reach the control path", ok, detail)
    return report.summary()


def scenario_workers(args: argparse.Namespace) -> int:
    """Throughput for 1, 4, 5, 6 - and any other count asked for.

    Nothing in the code may assume four devices, so this is the scenario that
    would catch it: each count is started from scratch and measured on its own.
    """
    counts: list[list[int]] = (
        [list(args.devices)]
        if args.device_sets is None
        else [[int(d) for d in chunk.split(",") if d.strip()] for chunk in args.device_sets]
    )
    print(f"\n== scenario workers: {len(counts)} configuration(s) ==")
    report = Report()
    results: list[dict[str, Any]] = []
    for devices in counts:
        print(f"\n-- {len(devices)} worker(s) on devices {devices} --")
        harness = _build(args, devices=devices)
        try:
            harness.start()
            if not harness.wait_ready():
                return _startup_failed(harness, report)
            _startup_facts(harness, report)
            harness.pump(1.0)

            harness.submit_at(args.rate_hz, args.seconds)
            harness.drain(seconds=args.infer_ms / 1000.0 + 10.0)
            measured = pool_report(harness)
            measured["device_count"] = len(devices)
            measured["devices"] = list(devices)
            results.append(measured)
            print_pool_report(f"{len(devices)} workers", measured)

            capacity_hz = len(devices) / (args.infer_ms / 1000.0)
            report.check(
                f"{len(devices)} workers: raw_inference_hz tracks capacity",
                measured["raw_inference_hz"] >= capacity_hz * 0.6,
                f"raw={measured['raw_inference_hz']:.2f} Hz "
                f"capacity~={capacity_hz:.2f} Hz",
            )
            report.check(
                f"{len(devices)} workers: every worker answered",
                all(w["infer_count"] > 0 for w in measured["workers"]),
                "counts=" + json.dumps([w["infer_count"] for w in measured["workers"]]),
            )
            report.check(
                f"{len(devices)} workers: no worker failed, no CUDA OOM",
                harness.scheduler.failed_worker_count == 0
                and not harness.metrics.faults.snapshot(),
                f"failed={harness.scheduler.failed_worker_count} "
                f"faults={json.dumps(harness.metrics.faults.snapshot())}",
            )
            # The acceptance criterion, on the numbers this mode can measure:
            # ``accepted_result_hz`` is the rate results reached the single
            # writer; the wire rate is measured by ``--mode pipeline``.
            if len(devices) >= 4:
                report.check(
                    f"{len(devices)} workers: accepted_result_hz >= 2.0",
                    measured["accepted_result_hz"] >= 2.0,
                    f"accepted={measured['accepted_result_hz']:.2f} Hz",
                )
                report.check(
                    f"{len(devices)} workers: accepted_nonempty_hz >= 2.0 "
                    "(scheduler-side stand-in for the UDP rate)",
                    measured["accepted_nonempty_hz"] >= 2.0,
                    f"accepted_nonempty={measured['accepted_nonempty_hz']:.2f} Hz",
                )
            report.check(
                f"{len(devices)} workers: pending stayed within one slot",
                measured["pending_count"] <= 1,
                f"pending={measured['pending_count']}",
            )
        finally:
            _shutdown(harness)

    if len(results) > 1:
        print("\n-- scaling summary --")
        print(f"  {'workers':>8} {'raw_hz':>8} {'accepted_hz':>12} {'accepted_nonempty_hz':>21}")
        for measured in results:
            print(
                f"  {measured['device_count']:>8} "
                f"{measured['raw_inference_hz']:>8.2f} "
                f"{measured['accepted_result_hz']:>12.2f} "
                f"{measured['accepted_nonempty_hz']:>21.2f}"
            )
        print(
            "  Rates over a window shorter than the run read a little high: a\n"
            "  Frequency times itself from its first event, so the drain tail is\n"
            "  counted while the wait for the first inference is not.  Read the\n"
            "  ratios, not the third decimal - and use --seconds 30 or more before\n"
            "  quoting any of these as a capacity figure."
        )
    return report.summary()


def scenario_pipeline(args: argparse.Namespace) -> int:
    """The real runner, the real tail, the real bytes - still no vehicle.

    Needs ``cv2`` (``pipeline.py`` imports it) and the real checkpoint when
    ``--real`` is given.  The runner is built but never started, so no camera is
    opened and no pose listener binds a port; tasks go straight to the runner's
    scheduler and the accepted results go through the runner's own
    ``_handle_result``, which is the single writer.
    """
    print(f"\n== scenario pipeline: real runner, loopback sink, {len(args.devices)} worker(s) ==")
    sink = WaypointSink().start()
    print(f"  . sink listening on {sink.host}:{sink.port}")

    # Everything the sender needs is set before the settings object is loaded.
    # ``WaypointSender.send`` refuses to transmit unless magic and vehicle_id are
    # both non-zero, and ``allow_unconfigured_udp`` is off by default - so an
    # otherwise perfect run would send nothing and look like a scheduling
    # failure.  The package directory is emptied on purpose: writing a JPEG set
    # per inference is a side effect an offline run must opt into.
    os.environ["ALPAMAYO_WAYPOINT_HOST"] = sink.host
    os.environ["ALPAMAYO_WAYPOINT_PORT"] = str(sink.port)
    os.environ.setdefault("ALPAMAYO_WAYPOINT_MAGIC", hex(WaypointSink.MAGIC))
    os.environ.setdefault("ALPAMAYO_VEHICLE_ID", "1")
    os.environ["ALPAMAYO_INFERENCE_MODE"] = "multi_process"
    os.environ["ALPAMAYO_INFER_PACKAGE_DIR"] = args.package_dir or ""
    # ``--devices`` has to be authoritative here for the same reason it is in
    # every other scenario: this mode loads the pool from the app config, and a
    # config that names no devices (the shipped default is ``gpu_devices: []``)
    # would either refuse to start or, worse, be read as "whichever GPUs are
    # free" and take a card another job is holding.  Pinned from the argument,
    # then verified below against what the loader actually produced.
    os.environ["ALPAMAYO_GPU_DEVICES"] = ",".join(str(int(d)) for d in args.devices)
    os.environ["ALPAMAYO_MINIMUM_READY_WORKERS"] = str(len(args.devices))
    os.environ["ALPAMAYO_LATEST_ONLY"] = "1"
    os.environ["ALPAMAYO_PENDING_CAPACITY"] = "1"

    report = Report()
    runner = None
    try:
        try:
            import pipeline
        except Exception as exc:  # noqa: BLE001 - most often a missing cv2
            print(f"  ! cannot import pipeline: {exc}")
            print(
                "  ! this mode needs the full dependency set (cv2, torch); "
                "it is meant for the deployment machine"
            )
            report.check("pipeline.py is importable", False, str(exc))
            return report.summary()

        import config.loader as loader

        settings = loader.load_app_config()
        # Refuse before a single process is spawned: if the loader did not take
        # the device list from the environment, this run would put workers on
        # cards nobody chose - and a benchmark on the wrong GPU is worse than no
        # benchmark, because it steals from whatever else is running there.
        pool_devices = tuple(int(d) for d in settings.inference_pool.gpu_devices)
        if pool_devices != tuple(int(d) for d in args.devices):
            report.check(
                "the pool took its device list from --devices",
                False,
                f"asked {list(args.devices)}, the loaded config gave "
                f"{list(pool_devices)} - refusing to start workers on unnamed GPUs",
            )
            return report.summary()
        report.check(
            "the pool took its device list from --devices",
            True,
            f"devices={list(pool_devices)}",
        )
        runner = pipeline.PipelineRunner(settings)

        if runner.scheduler is None:
            report.check(
                "the runner built a scheduler",
                False,
                "runner.scheduler is None - is ALPAMAYO_INFERENCE_MODE=multi_process?",
            )
            return report.summary()
        report.check(
            "the runner built a scheduler",
            True,
            f"devices={runner.scheduler.gpu_devices} "
            f"latest_only={runner.scheduler.latest_only}",
        )

        _seed_pose_history(runner)
        _install_stand_in_worker(runner, args)

        runner.scheduler.start()
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            runner.scheduler.tick()
            if runner.scheduler.ready_worker_count >= len(args.devices):
                break
            time.sleep(0.02)
        report.check(
            "all workers became ready",
            runner.scheduler.ready_worker_count >= len(args.devices),
            f"ready={runner.scheduler.ready_worker_count}/{len(args.devices)} "
            f"failed={runner.scheduler.failed_worker_count}",
        )

        # Submit at the input rate and tick.  The runner's own ``on_accepted``
        # does the crop, the transform, the UDP send and the package write, so
        # there is no second implementation of the tail to drift.
        seq = 0
        started = time.monotonic()
        period = 1.0 / args.rate_hz
        while time.monotonic() < started + args.seconds:
            seq += 1
            # Stamped from the real clock rather than the synthetic
            # ``CAPTURE_STEP_US * seq`` timeline the scheduling scenarios use.
            # The value becomes the trajectory's ``t0``, and the trim compares
            # the trajectory against the newest pose - which ``_add_pose``
            # stamps from the same clock.  A synthetic timeline starting near
            # zero makes every trajectory decades older than the pose, so the
            # trim takes its ``trajectory_expired`` path and every packet goes
            # out as an empty release frame: a closed loop that would have
            # measured the encoder and proved nothing about the waypoints.
            capture = time.time_ns() // 1000
            # A pose per cycle, roughly where a vehicle's 10 Hz stream would put
            # it: the crop trims against the newest pose, and a history seeded
            # once at startup would trim against a pose that never moves.
            _add_pose(runner, latitude=30.5 + seq * 1e-5, longitude=114.3)
            runner.scheduler.submit(make_task(seq, capture, cycle_id=seq))
            runner.scheduler.tick()
            target = started + seq * period
            slack = target - time.monotonic()
            if slack > 0:
                time.sleep(min(slack, period))
        # Drain: keep ticking, or in-flight results never reach the sender and
        # the wire rate is measured over a truncated run.
        drain_until = time.monotonic() + args.infer_ms / 1000.0 + 10.0
        while time.monotonic() < drain_until:
            runner.scheduler.tick()
            if (
                runner.scheduler.busy_worker_count == 0
                and runner.scheduler.pending_count == 0
            ):
                break
            time.sleep(0.02)
        time.sleep(0.5)

        wire = sink.report()
        print(f"  -- wire: {json.dumps(wire)}")
        snapshot = runner.snapshot
        metrics = snapshot.inference_metrics or {}
        print(
            "  -- runner: "
            f"raw_inference_hz={_hz(metrics, 'raw_inference')} "
            f"accepted_result_hz={_hz(metrics, 'accepted_result')} "
            f"udp_output_hz={_hz(metrics, 'udp_output')}"
        )
        print(f"  -- udp_packets: {json.dumps(metrics.get('udp_packets') or {})}")
        if snapshot.inference_pool is not None:
            print(f"  -- pool: {json.dumps(_pool_brief(snapshot.inference_pool.as_dict()))}")
        report.note(
            f"last inference package: {'written' if runner.last_result is not None else 'none'}"
        )

        # The three frequencies, and the one that is the acceptance criterion.
        report.check(
            "the sink decoded every packet cleanly",
            not wire["decode_errors"] and wire["bad_magic"] == 0,
            f"bad_magic={wire['bad_magic']} errors={wire['decode_errors']}",
        )
        report.check(
            "non-empty valid-trajectory UDP output >= 2.0 Hz",
            wire["hz_with_waypoints"] >= 2.0,
            f"hz_with_waypoints={wire['hz_with_waypoints']} "
            f"packets={wire['packets']} with_waypoints={wire['with_waypoints']}",
        )
        report.check(
            "the total packet rate is not what carried that",
            wire["packets"] >= wire["with_waypoints"],
            f"total={wire['packets']} with_waypoints={wire['with_waypoints']} "
            f"release={wire['release_frames']}",
        )
        report.check(
            "the UDP sequence did not go backwards",
            wire["seq_gaps"] == 0,
            f"seq_gaps={wire['seq_gaps']} last_seq={wire['last_seq']}",
        )
        report.check(
            "no worker failed, no CUDA OOM",
            runner.scheduler.failed_worker_count == 0,
            f"failed={runner.scheduler.failed_worker_count}",
        )
        report.check(
            "the published snapshot carries the pool view",
            snapshot.inference_pool is not None,
            "inference_pool is None, so /health would not report the pool",
        )
        return report.summary()
    finally:
        if runner is not None and runner.scheduler is not None:
            try:
                stopped = runner.scheduler.stop()
                if stopped.get("terminated") or stopped.get("killed"):
                    print(f"  ! pool did not stop gracefully: {json.dumps(stopped)}")
            except Exception as exc:  # noqa: BLE001 - the report matters more
                print(f"  ! pool did not stop cleanly: {exc}")
        sink.stop()


def _install_stand_in_worker(runner: Any, args: argparse.Namespace) -> None:
    """Replace the runner's scheduler with one whose workers skip the model.

    Only for the run without ``--real``.  The pool is rebuilt rather than
    patched because ``worker_entry`` is bound at construction - and the point of
    the stand-in is precisely that the real ``worker_main``, which would load a
    checkpoint, is not what runs.  Everything else about the pool is the real
    object, including the metrics and the result manager the runner already
    owns, so the numbers that come out are the runner's own.
    """
    if args.real:
        return
    from inference.fake_worker import fake_worker_main

    runner.scheduler = GpuScheduler(
        settings=runner.settings,
        pool_config=runner.settings.inference_pool,
        metrics=runner.inference_metrics,
        result_manager=runner.result_manager,
        on_accepted=runner._on_worker_result,
        worker_entry=fake_worker_main,
        worker_options={"delays_ms": [float(args.infer_ms)]},
    )


def _add_pose(
    runner: Any,
    *,
    latitude: float,
    longitude: float,
    timestamp_us: int | None = None,
) -> None:
    """Append one pose, as the pose listener would.

    ``PoseHistory`` stores ``GlobalPose``, not the ``PoseSample`` the model input
    carries - and the decision layer reads ``arrival_us`` off the newest one to
    age the pose against the monotonic clock.  A ``PoseSample`` has no such
    field, so handing the history one lets it look correct from the outside
    while the control path raises ``AttributeError`` on the first accepted
    result.  The fields are filled the way ``vehicle/receiver.py``'s mock
    provider fills them.

    The crop trims the trajectory against ``history.latest()``; if that is None
    the trim takes its ``latest_pose_unavailable`` path and every cycle falls
    back - which would make an offline run that "passed" prove nothing about the
    real tail.
    """
    payload_us = int(time.time() * 1_000_000) if timestamp_us is None else int(timestamp_us)
    runner.history.add(
        GlobalPose(
            timestamp_us=payload_us,
            payload_timestamp_us=payload_us,
            latitude=latitude,
            longitude=longitude,
            altitude=20.0,
            v_north=0.0,
            v_east=0.0,
            v_up=0.0,
            roll_deg=0.0,
            pitch_deg=0.0,
            azimuth_deg=vehicle_azimuth_for_compass(90.0),
            arrival_us=time.monotonic_ns() // 1000,
        )
    )


def _seed_pose_history(runner: Any) -> None:
    """Enough history for the trim to have a pose from the very first cycle.

    Back-dated at the 10 Hz spacing a vehicle streams at rather than taken from
    the wall clock in a tight loop: ``add`` refuses a timestamp that does not
    advance, so seeding from the clock alone would keep one sample, drop the
    other 39, and still leave ``latest()`` non-None - a seed that looks like it
    worked and is not what a 10 Hz stream would have produced.
    """
    now_us = int(time.time() * 1_000_000)
    for step in range(40):
        _add_pose(
            runner,
            latitude=30.5,
            longitude=114.3,
            timestamp_us=now_us - (40 - step) * 100_000,
        )


def _hz(metrics: dict[str, Any], key: str) -> float:
    block = metrics.get(key) or {}
    return round(float(block.get("hz_10s", 0.0)), 3)


def _pool_brief(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": view["mode"],
        "state": view["state"],
        "devices": view["devices"],
        "ready_workers": view["ready_workers"],
        "busy_workers": view["busy_workers"],
        "loading_workers": view["loading_workers"],
        "failed_workers": view["failed_workers"],
        "pending_task_seq": view["pending_task_seq"],
        "pending_count": view["pending_count"],
    }


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #


def _build(
    args: argparse.Namespace,
    *,
    delays_by_task_seq: dict[int, float] | None = None,
    devices: Sequence[int] | None = None,
    worker_options: Mapping[str, Any] | None = None,
    result_max_age_ms: float | None = None,
) -> Harness:
    chosen = list(devices if devices is not None else args.devices)
    settings = _settings_for(args, chosen)
    if args.real:
        entry, options = real_worker_main, {"warmup": True}
    else:
        from inference.fake_worker import fake_worker_main

        entry = fake_worker_main
        options = {"delays_ms": [float(args.infer_ms)]}
        if delays_by_task_seq is not None:
            options["delays_by_task_seq"] = dict(delays_by_task_seq)
        # Fault injection is merged last and deliberately not filtered: a
        # scenario that asks for a knob the worker does not implement should
        # fail loudly, not quietly run without it.
        options.update(worker_options or {})
    return Harness(
        devices=chosen,
        result_max_age_ms=(
            args.result_max_age_ms if result_max_age_ms is None else float(result_max_age_ms)
        ),
        max_capture_lag_ms=args.max_capture_lag_ms,
        worker_entry=entry,
        worker_options=options,
        settings=settings,
    )


def _settings_for(args: argparse.Namespace, devices: Sequence[int]) -> Any:
    """The real app config, for the scenario's device list.

    Loaded through the real loader, so a scenario sees the same settings object
    the pipeline would - including the camera and pose configuration the model
    input depends on.  The environment is set first because that is the override
    path the loader documents (JSON, then env, then defaults), and it is the
    only path that works for ``--device-sets``, where several device counts are
    measured in one process.
    """
    os.environ["ALPAMAYO_INFERENCE_MODE"] = "multi_process"
    os.environ["ALPAMAYO_GPU_DEVICES"] = ",".join(str(int(d)) for d in devices)
    os.environ["ALPAMAYO_MINIMUM_READY_WORKERS"] = str(len(devices))
    # Pinned rather than inherited: a config.json that turned ``latest_only`` off
    # would give the pool a short FIFO, and every scenario here is written
    # against the newest-wins policy the deployment runs.
    os.environ["ALPAMAYO_LATEST_ONLY"] = "1"
    os.environ["ALPAMAYO_PENDING_CAPACITY"] = "1"
    os.environ["ALPAMAYO_RESULT_MAX_AGE_MS"] = str(float(args.result_max_age_ms))
    os.environ["ALPAMAYO_MAX_CAPTURE_LAG_MS"] = str(float(args.max_capture_lag_ms))
    os.environ["ALPAMAYO_WORKER_START_TIMEOUT_S"] = str(READY_TIMEOUT_S)

    import config.loader as loader

    settings = loader.load_app_config()
    pool = settings.inference_pool
    if tuple(pool.gpu_devices) != tuple(int(d) for d in devices):
        # Not a silent substitution: if the environment did not take effect the
        # scenario would measure a pool nobody asked for.
        raise SystemExit(
            "pool config did not take the device list from the environment: "
            f"asked {list(devices)}, got {list(pool.gpu_devices)}"
        )
    return settings


def _startup_facts(harness: Harness, report: Report) -> None:
    """Report what the pool actually started with, before any measurement."""
    inventory = harness.inventory
    if inventory is None:
        report.note("gpu inventory: not probed")
    elif not inventory.available:
        report.note(
            f"gpu inventory unavailable ({inventory.source}: {inventory.error}); "
            "VRAM numbers below will be zeros"
        )
    else:
        # Reported even when the list is empty, and never presented as "no
        # GPUs": an empty inventory means the probe saw nothing, which is what
        # happens on a CPU-only box - not the same claim as a box that has none.
        report.note(
            f"gpu inventory via {inventory.source}: {len(inventory.devices)} device(s) visible"
        )
        for device in inventory.devices:
            free = (
                "unknown"
                if device.free_bytes is None
                else f"{device.free_bytes / (1 << 30):.1f} GiB free"
            )
            total = (
                "unknown"
                if device.total_bytes is None
                else f"{device.total_bytes / (1 << 30):.1f} GiB total"
            )
            report.note(f"device {device.index}: {device.name} {total}, {free}")
    report.note(
        f"workers ready={harness.scheduler.ready_worker_count} "
        f"loading={harness.scheduler.loading_worker_count} "
        f"devices={list(harness.scheduler.gpu_devices)}"
    )


def _startup_failed(harness: Harness, report: Report) -> int:
    report.check(
        "every worker became ready",
        False,
        f"ready={harness.scheduler.ready_worker_count} "
        f"failed={harness.scheduler.failed_worker_count} "
        f"loading={harness.scheduler.loading_worker_count}",
    )
    for worker in harness.scheduler.worker_views():
        if worker.error:
            report.note(f"worker {worker.worker_id} (device {worker.device}): {worker.error}")
    counts = report.summary()
    _shutdown(harness)
    return counts


def _shutdown(harness: Harness) -> None:
    """Stop the pool, bounded, and say so if it was not graceful."""
    try:
        stopped = harness.scheduler.stop()
    except Exception as exc:  # noqa: BLE001 - the report matters more
        print(f"  ! pool stop raised: {exc}")
        return
    if stopped.get("terminated") or stopped.get("killed"):
        print(
            "  ! pool stop was not graceful: "
            f"terminated={stopped.get('terminated')} killed={stopped.get('killed')}"
        )


def _parse_devices(text: str) -> list[int]:
    devices = [int(part) for part in text.replace(" ", "").split(",") if part]
    if not devices:
        raise argparse.ArgumentTypeError("at least one device index is required")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"device listed twice: {devices}")
    if any(d < 0 for d in devices):
        raise argparse.ArgumentTypeError(f"negative device index: {devices}")
    return devices


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Device-set syntax for --scenario workers:\n"
            "  use commas between GPU IDs in one configuration;\n"
            "  use semicolons between configurations.\n\n"
            "Example (4, then 5, then 6 workers):\n"
            "  --device-sets \"1,3,4,5;1,3,4,5,6;1,3,4,5,6,7\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=("scheduler", "backlog", "order", "workers", "faults", "pipeline"),
        default="scheduler",
        help=(
            "scheduler: all three scheduling scenarios in order (default).  "
            "backlog: 10 Hz input against a slower pool.  order: forced "
            "out-of-order completion.  workers: throughput per worker count.  "
            "faults: every way a result can be refused, injected through a real "
            "pool (fake workers, no GPU).  pipeline: the real runner plus a "
            "loopback UDP sink."
        ),
    )
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=[0, 1, 2, 3],
        help="comma-separated device indices, e.g. 1,3,4,5 (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--device-sets",
        default=None,
        help=(
            "device configurations for --scenario workers: separate GPU IDs "
            "within one configuration with commas, and separate configurations "
            "with semicolons; e.g. '1,3,4,5;1,3,4,5,6;1,3,4,5,6,7'.  "
            "Default is a single run on --devices."
        ),
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real worker_main (every worker loads a full model); "
        "default is the stand-in worker, which touches no GPU",
    )
    parser.add_argument("--rate-hz", type=float, default=10.0, help="input rate (default 10)")
    parser.add_argument("--infer-ms", type=float, default=1500.0,
                        help="stand-in inference time per task, ms (default 1500)")
    parser.add_argument("--seconds", type=float, default=MEASURE_S,
                        help=f"measurement window, s (default {MEASURE_S:g})")
    parser.add_argument("--result-max-age-ms", type=float, default=2500.0)
    parser.add_argument("--max-capture-lag-ms", type=float, default=2500.0)
    parser.add_argument(
        "--package-dir",
        default="",
        help=(
            "with --scenario pipeline, write an inference package per sent "
            "trajectory here.  Empty (the default) writes none - the package is "
            "16 JPEGs plus the record, and an offline run should not fill a "
            "disk unless it was asked to."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.device_sets is not None:
        args.device_sets = [chunk for chunk in args.device_sets.split(";") if chunk.strip()]

    print("multi-GPU offline test - no vehicle, no control link")
    if not args.real:
        print(
            "  . stand-in workers: the model call is replaced, so the numbers "
            "below are scheduling numbers, not model numbers"
        )

    if args.scenario == "pipeline":
        return scenario_pipeline(args)
    if args.scenario == "faults":
        return scenario_faults(args)

    scenarios = {
        "backlog": scenario_backlog,
        "order": scenario_order,
        "workers": scenario_workers,
    }
    if args.scenario == "scheduler":
        order = ["order", "backlog", "workers"]
    else:
        order = [args.scenario]

    codes = []
    for name in order:
        codes.append(scenarios[name](args))
    if any(code != 0 for code in codes):
        print("\nOne or more scenarios failed.  A failure here is a scheduling "
              "or capacity problem, not a flaky test.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
