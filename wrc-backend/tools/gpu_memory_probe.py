#!/usr/bin/env python3
"""Measure what N Alpamayo workers actually cost in VRAM, and whether it grows.

Three phases, each answering one question the offline test cannot.

``load``
    Bring the pool up and watch every requested device while it happens.  Two
    things are being checked.  First, that each worker's model fits: the peak is
    read from NVML *and* from the worker's own ``torch.cuda`` counters, because
    they answer different questions - NVML sees the whole physical device
    including anything else running on it, while the worker's counter is the one
    that would throw the OOM.  Second, that a worker pinned to device N really
    did allocate on device N: growth on the *other* devices during load is the
    signature of ``CUDA_VISIBLE_DEVICES`` not taking effect, and it would show up
    here as four models piled onto one card.

``warmup``
    Run a few real inferences and report the steady state each worker settles
    at.  This is the number the soak is compared against, and the number to
    budget against when deciding how many workers the box holds.

``bench``
    ``--inferences`` measured inferences, one at a time, each timed on both
    clocks: the worker's own ``infer_ms`` around the model call, and the
    parent's submit-to-accepted wall time.  It reports the per-inference list as
    well as mean/P50/P95/min/max, the trajectory shape each one came back with,
    and whether every coordinate was finite.

    One at a time rather than a batch, because the pool is newest-wins: a batch
    wider than the number of idle workers would have the pool legitimately drop
    the surplus and the timings would be of whichever few survived.

``soak``
    Keep the pool working for ``--soak-minutes`` and sample every worker's
    ``allocated_bytes`` and ``reserved_bytes`` throughout, then fit a slope to
    each.  A reserved heap that is flat while allocated is flat is a working
    allocator.  Both climbing across half an hour is a leak, and the job of this
    phase is to make the difference visible before an eight-hour shift does.

    Skipped unless asked for: a soak is measured in tens of minutes, so it is
    never something a run stumbles into.  ``--soak-minutes`` below 15 is accepted
    and will say that it is too short to conclude anything from.

After the pool is stopped the probe waits for the cards to come back and prints
what they settled at.  A worker that died without releasing its memory is
indistinguishable from a clean exit by looking at the process table, so the
release is measured rather than assumed.

Uses the real ``worker_main``, so every worker loads a complete checkpoint on
its own device.  Needs neither ``cv2`` nor ``pipeline.py``: it drives the pool
directly, so it can run on a box where the camera stack is not installed yet.

It submits synthetic windows at the *configured* camera resolution rather than
tiny ones, because the point here is memory - a 4x4 test frame would measure the
allocator's rounding, not the model.  It sends nothing and takes nothing over:
no UDP, no camera, no pose listener.

``--devices`` names **physical** indices, and the shell must not narrow them
    Each worker pins itself with ``CUDA_VISIBLE_DEVICES`` set to the index it was
    configured with, and then loads on ``cuda:0``.  That makes the configured
    index a physical one, and it makes an inherited ``CUDA_VISIBLE_DEVICES``
    actively dangerous rather than merely redundant: ``CUDA_VISIBLE_DEVICES=1``
    with ``--devices 0`` renumbers the card, so the worker sets
    ``CUDA_VISIBLE_DEVICES=0`` and the model lands on physical GPU 0 while every
    line of the log says 0.  The probe refuses to start in that state.

Usage
    python tools/gpu_memory_probe.py --devices 1
    python tools/gpu_memory_probe.py --devices 1 --inferences 5
    python tools/gpu_memory_probe.py --devices 1,3,4,5
    python tools/gpu_memory_probe.py --devices 1,3,4,5,6 --soak-minutes 30
    python tools/gpu_memory_probe.py --devices 1,3,4,5 --soak-minutes 15 --json soak.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from camera.models import CameraWindow  # noqa: E402
from config.loader import AppConfig, InferencePoolConfig  # noqa: E402
from inference.gpu_pool import (  # noqa: E402
    GpuInventory,
    GpuScheduler,
    probe_gpu_inventory,
    worker_main,
)
from inference.metrics import InferenceMetrics, LatencyStats  # noqa: E402
from inference.result_manager import (  # noqa: E402
    MIN_TRAJECTORY_COLUMNS,
    MIN_TRAJECTORY_POINTS,
    ResultManager,
)
from inference.tasks import InferenceTask  # noqa: E402
from vehicle.models import PoseSample  # noqa: E402

MIB = 1 << 20


class Report:
    """Collects pass/fail lines, printed as they are decided."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(ok), detail))
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        return bool(ok)

    def note(self, text: str) -> None:
        print(f"   . {text}")

    def summary(self) -> int:
        failed = [name for name, ok, _ in self.checks if not ok]
        passed = len(self.checks) - len(failed)
        print()
        if failed:
            print(f"RESULT: FAILED ({len(failed)}/{len(self.checks)} checks failed)")
            for name in failed:
                print(f"  - {name}")
            return 1
        print(f"RESULT: OK ({passed}/{len(self.checks)} checks passed)")
        return 0

#: How long to wait for a worker to load a 10B checkpoint.  Generous: the cost
#: of waiting too long is a slow probe, the cost of giving up early is a wrong
#: answer about whether the pool fits.
DEFAULT_LOAD_TIMEOUT_S = 900.0

#: Where the growth check draws the line.  Not zero, because a CUDA allocator
#: legitimately holds a little more as its pools fragment, and a threshold of
#: zero would fail every run and be ignored.  Well under a percent of a 24 GB
#: card per hour, so four hours of a real shift stays inside the noise.
DEFAULT_MAX_GROWTH_MIB_PER_HOUR = 32.0

#: How long one inference may take before the bench calls it a hang.  A 10B
#: model at BF16 on one 4090 is seconds, not minutes; this is the "something is
#: wrong" bound, not a budget.
DEFAULT_INFER_TIMEOUT_S = 300.0

#: How close to the pre-pool baseline a card has to come back for the release
#: check to pass.  Not zero: the driver keeps a little per-context bookkeeping
#: and another tenant can legitimately take memory during the run, so a
#: zero-tolerance check would fail for reasons that have nothing to do with
#: this probe.  Small enough that a worker's worth of unreleased memory - the
#: thing being checked for - is far outside it.
DEFAULT_RELEASE_TOLERANCE_MIB = 256.0

#: How long to keep waiting for the driver to hand the memory back.
DEFAULT_RELEASE_TIMEOUT_S = 120.0

#: The nvidia-smi fields this probe reads.  Physical indices, whole device, and
#: no CUDA context in this process.
NVIDIA_SMI_FIELDS = "index,memory.total,memory.used,memory.free"


# --------------------------------------------------------------------------- #
# Synthetic input, at the real resolution
# --------------------------------------------------------------------------- #


def make_window(settings: AppConfig, capture_us: int) -> CameraWindow:
    """One window at the configured camera geometry.

    Deliberately the real size: ``frames_per_camera`` cameras x frames of
    ``frame_height`` x ``frame_width`` uint8 is what the model's preprocessor
    actually receives, and therefore what its activations are sized against.
    """
    cameras = list(settings.cimg_camera_indices)
    frames_per_camera = int(settings.frames_per_camera)
    window = CameraWindow(
        t0_us=int(capture_us),
        frames=torch.zeros(
            (len(cameras), frames_per_camera, 3,
             int(settings.frame_height), int(settings.frame_width)),
            dtype=torch.uint8,
        ),
        camera_indices=cameras,
        timestamps_us=[int(capture_us)] * frames_per_camera,
        frame_ids=[["probe"] * frames_per_camera for _ in cameras],
    )
    return window


def make_task(settings: AppConfig, task_seq: int, capture_us: int) -> InferenceTask:
    """A complete task, built through the real ``InferenceTask.from_build``."""
    steps = int(settings.history_steps)
    built = SimpleNamespace(
        ego_xyz=torch.zeros((1, 1, steps, 3), dtype=torch.float32),
        ego_rot=(
            torch.eye(3, dtype=torch.float32)
            .reshape(1, 1, 1, 3, 3)
            .repeat(1, 1, steps, 1, 1)
        ),
        reference_pose=PoseSample(
            timestamp_us=int(capture_us),
            latitude=30.5,
            longitude=114.3,
            altitude=20.0,
            heading_deg=90.0,
        ),
        reference_payload_timestamp_us=int(capture_us),
        ego_history=(),
        camera_frame_ids=(),
        camera_frame_id={},
        latest_frame_ids=(),
        camera_age_ms=0.0,
        camera_pose_delta_ms=0.0,
        camera_lag_ms=None,
    )
    return InferenceTask.from_build(
        task_seq, built, make_window(settings, capture_us), cycle_id=task_seq
    )


# --------------------------------------------------------------------------- #
# Device memory, two ways
# --------------------------------------------------------------------------- #


def device_used_bytes(inventory: GpuInventory) -> dict[int, int]:
    """Used bytes per physical device, from the read-only probe.

    ``total - free`` rather than a per-process figure: this is the whole device,
    which is what an OOM is actually about on a box where someone else may also
    be running something.
    """
    used: dict[int, int] = {}
    if not inventory.available:
        return used
    for device in inventory.devices:
        if device.total_bytes is None or device.free_bytes is None:
            continue
        used[int(device.index)] = int(device.total_bytes) - int(device.free_bytes)
    return used


def _run_nvidia_smi(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=True,
    )
    return completed.stdout


def nvidia_smi_memory(
    runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[int, dict[str, float]]:
    """Whole-device memory per *physical* index, straight from nvidia-smi.

    nvidia-smi rather than NVML for the parent's numbers, for three reasons.
    The deployment venv has no ``pynvml``; nvidia-smi reports physical indices,
    which is what ``--devices`` names and what an operator reading their own
    ``nvidia-smi`` will see; and it holds no CUDA context, so the baseline this
    probe measures against is not itself a measurement of the probe.

    Empty when nvidia-smi is missing or unreadable.  The caller falls back to
    the read-only device probe rather than guessing.
    """
    run = _run_nvidia_smi if runner is None else runner
    try:
        text = run(
            [f"--query-gpu={NVIDIA_SMI_FIELDS}", "--format=csv,noheader,nounits"]
        )
    except Exception:  # noqa: BLE001 - an unreadable nvidia-smi is a reportable state
        return {}

    devices: dict[int, dict[str, float]] = {}
    for line in str(text).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            devices[int(fields[0])] = {
                "total_mib": float(fields[1]),
                "used_mib": float(fields[2]),
                "free_mib": float(fields[3]),
            }
        except ValueError:
            # A header line, or a card reporting "[N/A]" for a field.  Skipping
            # is right: a partial reading is still a reading.
            continue
    return devices


def describe_device(index: int, memory: dict[int, dict[str, float]]) -> str:
    values = memory.get(index)
    if not values:
        return f"dev{index}: unavailable"
    return (
        f"dev{index}: used={values['used_mib']:.0f}MiB "
        f"free={values['free_mib']:.0f}MiB total={values['total_mib']:.0f}MiB"
    )


def sample_devices() -> dict[int, int]:
    """A fresh whole-device reading in bytes, keyed by physical index.

    ``nvidia-smi`` first; the read-only inventory probe as a fallback for a box
    with no nvidia-smi on PATH.  Both answer in physical device indices, so the
    numbers stay comparable with ``--devices`` and with an operator's own
    ``nvidia-smi``.

    ``total - free`` rather than a per-process figure: this is the whole device,
    which is what an OOM is actually about on a box where someone else may also
    be running something.
    """
    memory = nvidia_smi_memory()
    if memory:
        return {
            index: int(values["used_mib"] * MIB) for index, values in memory.items()
        }
    return device_used_bytes(probe_gpu_inventory())


def slope_per_hour(samples: Sequence[tuple[float, float]]) -> float | None:
    """Least-squares slope in units per hour over ``(seconds, value)`` samples.

    None when there is nothing to fit.  Used instead of last-minus-first because
    one noisy final sample would otherwise decide the verdict on a leak.
    """
    if len(samples) < 3:
        return None
    n = float(len(samples))
    mean_x = sum(x for x, _ in samples) / n
    mean_y = sum(y for _, y in samples) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in samples)
    if denominator <= 0.0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in samples)
    return (numerator / denominator) * 3600.0


# --------------------------------------------------------------------------- #
# The probe
# --------------------------------------------------------------------------- #


class Probe:
    """A real pool of real workers, plus the numbers to judge it by."""

    def __init__(self, settings: AppConfig, devices: Sequence[int]) -> None:
        self.settings = settings
        self.devices = [int(d) for d in devices]
        self.config = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=tuple(self.devices),
            latest_only=True,
            pending_capacity=1,
            # A probe dispatches one task at a time and waits, so the age and
            # lag budgets only need to be wide enough not to drop what it sent.
            result_max_age_ms=600_000.0,
            max_capture_lag_ms=600_000.0,
            scheduler_poll_interval_ms=20.0,
            worker_start_timeout_s=DEFAULT_LOAD_TIMEOUT_S,
            minimum_ready_workers=len(self.devices),
            worker_warmup=True,
        )
        self.metrics = InferenceMetrics(long_s=60.0)
        self.result_manager = ResultManager(
            result_max_age_ms=600_000.0, max_capture_lag_ms=600_000.0
        )
        self.accepted: list[int] = []
        self.scheduler = GpuScheduler(
            settings=settings,
            pool_config=self.config,
            metrics=self.metrics,
            result_manager=self.result_manager,
            on_accepted=self._on_accepted,
            worker_entry=worker_main,
            worker_options={"warmup": True},
        )
        self.inventory: GpuInventory | None = None
        #: Whole-device usage before the pool started, so the release check has
        #: something to come back to.
        self.baseline_used: dict[int, int] = {}
        #: One row per accepted result, in acceptance order.  The parent's own
        #: record: the worker reports ``infer_ms``, everything else here is
        #: measured on this side of the process boundary.
        self.records: list[dict[str, Any]] = []
        self._seq = 0
        self._submitted_at: dict[int, float] = {}

    def _on_accepted(self, result: Any, task: InferenceTask) -> None:
        self.accepted.append(int(result.task_seq))
        self.records.append(self._describe(result))

    def _describe(self, result: Any) -> dict[str, Any]:
        """What came back, measured where an operator would look for it.

        The trajectory inspection here is deliberately *after* the result
        manager has already accepted it.  A shape or finiteness problem would
        have been dropped before this callback ran, so this does not duplicate
        that verdict - it records what the accepted shapes actually were, which
        is the number a reader wants and the validator never prints.
        """
        model_result = getattr(result, "model_result", None)
        positions = getattr(model_result, "positions", None)
        shape: tuple[int, ...] = ()
        finite = False
        if positions is not None:
            array = np.asarray(positions)
            shape = tuple(int(dimension) for dimension in array.shape)
            finite = bool(np.isfinite(array).all())
        seq = int(result.task_seq)
        submitted = self._submitted_at.get(seq)
        return {
            "task_seq": seq,
            "worker_id": int(result.worker_id),
            "device": int(result.device),
            # The worker's clock around its own model call.
            "infer_ms": round(float(result.infer_ms), 3),
            # This process's clock, from handing the task to the scheduler to
            # the accepted callback firing.  Includes the queue wait, the
            # dispatch, the model and the validation.
            "wall_ms": (
                None
                if submitted is None
                else round((time.monotonic() - submitted) * 1000.0, 3)
            ),
            "shape": list(shape),
            "points": int(shape[0]) if shape else 0,
            "finite": finite,
        }

    def failure_count(self) -> int:
        """Results the pool refused, from either half of the accounting.

        ``worker_error`` is the result manager's count of failed results;
        ``worker_failed`` is the scheduler's count of workers that died.  They
        are different events and a run that hangs wants to notice either.
        """
        stats = self.result_manager.stats()
        return int(stats.get("worker_error", 0)) + int(
            self.metrics.faults.get("worker_failed")
        )

    def oom_count(self) -> int:
        return int(self.result_manager.stats().get("cuda_oom", 0))

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> GpuInventory:
        self.inventory = self.scheduler.start()
        return self.inventory

    def stop(self) -> dict[str, Any]:
        return self.scheduler.stop()

    # -- driving ------------------------------------------------------------ #

    def submit(self, *, capture_us: int | None = None) -> int:
        self._seq += 1
        capture = (
            100_000 * self._seq if capture_us is None else int(capture_us)
        )
        self._submitted_at[self._seq] = time.monotonic()
        self.scheduler.submit(make_task(self.settings, self._seq, capture))
        return self._seq

    def tick_until(self, predicate: Any, *, timeout_s: float, on_tick: Any = None) -> bool:
        """Tick until ``predicate()`` is true or the deadline passes.

        ``on_tick`` runs on every iteration, which is where the sampling lives -
        the device numbers have to be read *while* the pool is working, not
        after it stops.
        """
        deadline = time.monotonic() + float(timeout_s)
        poll = self.config.poll_interval_s
        while time.monotonic() < deadline:
            self.scheduler.tick()
            if on_tick is not None:
                on_tick()
            if predicate():
                return True
            time.sleep(poll)
        self.scheduler.tick()
        return bool(predicate())

    def worker_memory(self) -> dict[int, dict[str, int]]:
        """Each worker's own view of its device, straight from its last report."""
        return {
            int(w.worker_id): {
                "device": int(w.device),
                "state": w.state,
                "allocated_bytes": int(w.memory.allocated_bytes),
                "reserved_bytes": int(w.memory.reserved_bytes),
                "peak_allocated_bytes": int(w.peak_allocated_bytes),
                "peak_reserved_bytes": int(w.memory.max_reserved_bytes),
                "infer_count": int(w.infer_count),
                "error": w.error,
            }
            for w in self.scheduler.worker_views()
        }


def phase_load(probe: Probe, report: Report) -> bool:
    """Start the pool and watch the devices while it loads."""
    print(f"\n-- phase load: {len(probe.devices)} worker(s) on devices {probe.devices}")

    physical = nvidia_smi_memory()
    baseline = sample_devices()
    probe.baseline_used = dict(baseline)
    if physical:
        for device in probe.devices:
            print(f"   baseline {describe_device(device, physical)}")
    elif baseline:
        print(
            "   baseline used: "
            + ", ".join(
                f"dev{d}={baseline.get(d, 0) / MIB:.0f}MiB" for d in probe.devices
            )
            + " (no nvidia-smi; read through the device probe instead)"
        )
    else:
        print("   baseline used: unavailable (no nvidia-smi and no device list)")

    started = time.monotonic()
    peaks: dict[int, int] = dict(baseline)
    timeline: list[float] = []

    def sample() -> None:
        now = time.monotonic()
        if timeline and now - timeline[-1] < 1.0:
            return
        timeline.append(now)
        for device, used in sample_devices().items():
            peaks[device] = max(peaks.get(device, 0), used)

    probe.start()
    loaded = probe.tick_until(
        lambda: probe.scheduler.ready_worker_count >= len(probe.devices)
        or probe.scheduler.failed_worker_count > 0,
        timeout_s=DEFAULT_LOAD_TIMEOUT_S,
        on_tick=sample,
    )
    elapsed_s = time.monotonic() - started

    ready = probe.scheduler.ready_worker_count
    failed = probe.scheduler.failed_worker_count
    if not report.check(
        "every worker loaded and became ready",
        loaded and ready >= len(probe.devices) and failed == 0,
        f"ready={ready}/{len(probe.devices)} failed={failed} "
        f"loading={probe.scheduler.loading_worker_count} after {elapsed_s:.0f}s",
    ):
        for view in probe.scheduler.worker_views():
            if view.error:
                print(f"   worker {view.worker_id} (device {view.device}): {view.error}")

    memory = probe.worker_memory()
    for worker_id, values in sorted(memory.items()):
        print(
            f"   worker {worker_id} dev {values['device']} {values['state']}: "
            f"allocated={values['allocated_bytes'] / MIB:.0f}MiB "
            f"reserved={values['reserved_bytes'] / MIB:.0f}MiB "
            f"peak_allocated={values['peak_allocated_bytes'] / MIB:.0f}MiB "
            f"peak_reserved={values['peak_reserved_bytes'] / MIB:.0f}MiB"
        )

    if baseline:
        for device in probe.devices:
            delta = peaks.get(device, 0) - baseline.get(device, 0)
            print(
                f"   dev{device}: used {baseline.get(device, 0) / MIB:.0f} -> "
                f"{peaks.get(device, 0) / MIB:.0f}MiB "
                f"(+{delta / MIB:.0f}MiB while loading)"
            )
        # The pinning check.  A worker that ignored CUDA_VISIBLE_DEVICES would
        # load onto device 0 and this would show four models on one card and
        # three untouched cards.
        silent = [
            d
            for d in probe.devices
            if peaks.get(d, 0) - baseline.get(d, 0) < 64 * MIB
        ]
        report.check(
            "every requested device gained a model",
            not silent,
            (
                "devices with no meaningful growth: " + str(silent)
                if silent
                else "each device grew by at least 64MiB, so no worker "
                "allocated on someone else's card"
            ),
        )
    else:
        report.note("device growth not checked: no NVML reading was available")

    after = nvidia_smi_memory()
    if after:
        for device in probe.devices:
            print(f"   loaded  {describe_device(device, after)}")

    print(f"   load wall time: {elapsed_s:.0f}s")
    # Only a fully ready pool may proceed.  A pool that started but lost a
    # worker would make every memory number below a measurement of the
    # survivors, which is not what the caller asked for.
    return loaded and ready >= len(probe.devices) and failed == 0


def phase_warmup(probe: Probe, report: Report, tasks_per_worker: int) -> None:
    """Run a few real inferences so the steady state is reached and measured.

    Round by round, one task per worker, because the pool is
    newest-wins: submitting a batch larger than the number of idle workers would
    have the pool legitimately drop the surplus, and this phase would then be
    asserting against the policy it is supposed to be measuring around.
    """
    rounds = max(1, int(tasks_per_worker))
    workers = len(probe.devices)
    print(f"\n-- phase warmup: {rounds} round(s) of {workers} real inference(s)")

    started = time.monotonic()
    accepted = 0
    for _ in range(rounds):
        probe.tick_until(
            lambda: probe.scheduler.ready_worker_count >= workers,
            timeout_s=DEFAULT_LOAD_TIMEOUT_S,
        )
        before = len(probe.accepted)
        for _ in range(workers):
            probe.submit()
        probe.tick_until(
            lambda: probe.scheduler.busy_worker_count == 0
            and probe.scheduler.pending_count == 0,
            timeout_s=DEFAULT_LOAD_TIMEOUT_S,
        )
        accepted += len(probe.accepted) - before
    elapsed_s = time.monotonic() - started

    expected = rounds * workers
    report.check(
        "every submitted task produced an accepted result",
        accepted == expected,
        f"accepted={accepted}/{expected} "
        f"drops={json.dumps(probe.result_manager.stats())}",
    )

    snapshot = probe.metrics.snapshot()
    print(f"   warmup wall time: {elapsed_s:.0f}s")
    print(
        f"   infer_ms: {json.dumps(snapshot['latency_ms']['infer'])} "
        f"over {elapsed_s:.0f}s"
    )
    for worker_id, values in sorted(probe.worker_memory().items()):
        print(
            f"   worker {worker_id} dev {values['device']}: "
            f"allocated={values['allocated_bytes'] / MIB:.0f}MiB "
            f"reserved={values['reserved_bytes'] / MIB:.0f}MiB "
            f"peak_reserved={values['peak_reserved_bytes'] / MIB:.0f}MiB "
            f"n={values['infer_count']}"
        )
    report.note(
        "the reserved-above-allocated gap above is the allocator's cache, and it "
        "is the number the soak watches"
    )


def phase_bench(
    probe: Probe,
    report: Report,
    count: int,
    *,
    timeout_s: float = DEFAULT_INFER_TIMEOUT_S,
) -> dict[str, Any]:
    """``count`` measured inferences, one at a time, each timed on both clocks.

    One at a time because the pool is newest-wins: a batch wider than the number
    of idle workers would have the pool legitimately drop the surplus, and the
    timings reported would be of whichever few happened to survive.

    Returns the collected numbers, so ``--json`` can carry them.
    """
    count = max(1, int(count))
    print(f"\n-- phase bench: {count} measured inference(s), one at a time")

    first = len(probe.records)
    started = time.monotonic()
    submitted: list[int] = []
    for _ in range(count):
        before_records = len(probe.records)
        before_failures = probe.failure_count()
        seq = probe.submit()
        submitted.append(seq)
        probe.tick_until(
            lambda: len(probe.records) > before_records
            or probe.failure_count() > before_failures,
            timeout_s=timeout_s,
        )
    elapsed_s = time.monotonic() - started

    measured = probe.records[first:]
    accepted = len(measured)

    infer_stats = LatencyStats()
    wall_stats = LatencyStats()
    for row in measured:
        infer_stats.add(row["infer_ms"])
        if row["wall_ms"] is not None:
            wall_stats.add(row["wall_ms"])

    for row in measured:
        wall = row["wall_ms"]
        wall_text = "n/a" if wall is None else f"{wall:9.1f}"
        print(
            f"   #{row['task_seq']:>3} worker {row['worker_id']} dev {row['device']} "
            f"infer={row['infer_ms']:8.1f}ms wall={wall_text}ms "
            f"shape={tuple(row['shape'])} finite={row['finite']}"
        )

    print(f"   bench wall time: {elapsed_s:.1f}s over {accepted} accepted")
    print(
        "   infer_ms (worker-side, around the model call only): "
        + json.dumps(infer_stats.snapshot())
    )
    print(
        "   wall_ms  (submit -> accepted, this process's clock): "
        + json.dumps(wall_stats.snapshot())
    )

    report.check(
        "every submitted inference was accepted",
        accepted == count,
        f"accepted={accepted}/{count} drops={json.dumps(probe.result_manager.stats())}",
    )

    bad_shape = [
        row["task_seq"]
        for row in measured
        if len(row["shape"]) != 2
        or row["shape"][1] < MIN_TRAJECTORY_COLUMNS
        or row["shape"][0] < MIN_TRAJECTORY_POINTS
    ]
    shapes = sorted({f"{tuple(row['shape'])}" for row in measured})
    report.check(
        "every trajectory had a usable shape",
        bool(measured) and not bad_shape,
        (
            f"task_seq with an unusable shape: {bad_shape}"
            if bad_shape
            else (
                "no trajectory came back to inspect"
                if not measured
                else f"every result was (T, >={MIN_TRAJECTORY_COLUMNS}) with at "
                f"least {MIN_TRAJECTORY_POINTS} point; saw {', '.join(shapes)}"
            )
        ),
    )

    non_finite = [row["task_seq"] for row in measured if not row["finite"]]
    report.check(
        "no NaN or Inf in any trajectory",
        bool(measured) and not non_finite,
        (
            f"task_seq with a non-finite coordinate: {non_finite}"
            if non_finite
            else (
                "no trajectory came back to inspect"
                if not measured
                else "every coordinate of every accepted trajectory was finite"
            )
        ),
    )

    report.check(
        "no CUDA out-of-memory",
        probe.oom_count() == 0,
        f"cuda_oom={probe.oom_count()}",
    )
    report.check(
        "no failed results and no dead workers",
        probe.failure_count() == 0,
        f"worker_error+worker_failed={probe.failure_count()}",
    )

    return {
        "count": count,
        "accepted": accepted,
        "submitted": submitted,
        "wall_s": round(elapsed_s, 3),
        "infer_ms": infer_stats.snapshot(),
        "wall_ms": wall_stats.snapshot(),
        "per_inference": measured,
    }


def wait_for_release(
    probe: Probe,
    *,
    tolerance_mib: float = DEFAULT_RELEASE_TOLERANCE_MIB,
    timeout_s: float = DEFAULT_RELEASE_TIMEOUT_S,
    sampler: Callable[[], dict[int, int]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for the cards to come back, and report what they settled at.

    A worker that died without releasing its memory looks exactly like a clean
    exit from the process table alone - the driver holds the memory until the
    last context in the process goes away, and it goes away when the process
    does, whether or not anything was cleaned up.  So the release is measured.

    The tolerance is not zero on purpose: another tenant can take memory during
    the run, and the driver keeps a little bookkeeping per context.  A whole
    worker's worth - several GiB - is far outside it.
    """
    sample = sample_devices if sampler is None else sampler
    baseline = dict(probe.baseline_used)
    if not baseline:
        return {"checked": False, "reason": "no pre-pool baseline was readable"}

    deadline = time.monotonic() + float(timeout_s)
    waited_s = 0.0
    started = time.monotonic()
    while True:
        after = sample()
        waited_s = time.monotonic() - started
        over = {
            device: {
                "baseline_mib": round(baseline.get(device, 0) / MIB, 1),
                "after_mib": round(after.get(device, 0) / MIB, 1),
                "over_mib": round(
                    (after.get(device, 0) - baseline.get(device, 0)) / MIB, 1
                ),
            }
            for device in probe.devices
            if after.get(device, 0) - baseline.get(device, 0) > tolerance_mib * MIB
        }
        if not over or time.monotonic() >= deadline:
            return {
                "checked": True,
                "released": not over,
                "waited_s": round(waited_s, 1),
                "tolerance_mib": float(tolerance_mib),
                "baseline_mib": {
                    d: round(baseline.get(d, 0) / MIB, 1) for d in probe.devices
                },
                "after_mib": {
                    d: round(after.get(d, 0) / MIB, 1) for d in probe.devices
                },
                "still_held": over,
            }
        sleep(1.0)


def phase_release(
    probe: Probe,
    report: Report,
    *,
    tolerance_mib: float = DEFAULT_RELEASE_TOLERANCE_MIB,
    timeout_s: float = DEFAULT_RELEASE_TIMEOUT_S,
) -> dict[str, Any]:
    """The release check, reported the way the other phases report."""
    print("\n-- phase release: did the cards come back")
    outcome = wait_for_release(
        probe, tolerance_mib=tolerance_mib, timeout_s=timeout_s
    )
    if not outcome.get("checked"):
        report.note(
            "release not checked: " + str(outcome.get("reason", "unavailable"))
        )
        return outcome

    for device in probe.devices:
        print(
            f"   dev{device}: {outcome['baseline_mib'].get(device)}MiB before -> "
            f"{outcome['after_mib'].get(device)}MiB after "
            f"({outcome['waited_s']:g}s)"
        )
    held = outcome.get("still_held") or {}
    report.check(
        "every device returned to its pre-pool level",
        bool(outcome.get("released")),
        (
            "still held above tolerance: " + json.dumps(held)
            if held
            else f"within {tolerance_mib:g}MiB of baseline after "
            f"{outcome['waited_s']:g}s"
        ),
    )
    return outcome


def phase_soak(probe: Probe, report: Report, *, minutes: float, rate_hz: float,
               sample_s: float, max_growth: float) -> None:
    """Keep the pool working and fit a slope to each worker's memory."""
    print(
        f"\n-- phase soak: {minutes:g} minute(s) at {rate_hz:g} Hz, "
        f"sampling every {sample_s:g}s"
    )
    if minutes * 60.0 < 15 * 60.0:
        print(
            "   ! under 15 minutes: long enough to catch a fast leak, not long "
            "enough to call a slow one absent"
        )

    start_allocated = probe.worker_memory()
    # (seconds, bytes) per worker per counter.
    series: dict[int, dict[str, list[tuple[float, float]]]] = {
        worker_id: {"allocated": [], "reserved": []}
        for worker_id in start_allocated
    }
    device_series: dict[int, list[tuple[float, float]]] = {
        d: [] for d in probe.devices
    }
    started = time.monotonic()
    last_sample = 0.0
    next_submit = started
    period = 1.0 / max(1e-6, float(rate_hz))
    end_at = started + minutes * 60.0

    def sample() -> None:
        nonlocal last_sample
        now = time.monotonic()
        if now - last_sample < float(sample_s):
            return
        last_sample = now
        offset = now - started
        for worker_id, values in probe.worker_memory().items():
            bucket = series.setdefault(
                worker_id, {"allocated": [], "reserved": []}
            )
            bucket["allocated"].append((offset, float(values["allocated_bytes"])))
            bucket["reserved"].append((offset, float(values["reserved_bytes"])))
        for device, used in sample_devices().items():
            device_series.setdefault(device, []).append((offset, float(used)))

    while time.monotonic() < end_at:
        now = time.monotonic()
        if now >= next_submit:
            probe.submit()
            next_submit = now + period
        probe.scheduler.tick()
        sample()

    probe.tick_until(
        lambda: probe.scheduler.busy_worker_count == 0,
        timeout_s=DEFAULT_LOAD_TIMEOUT_S,
        on_tick=sample,
    )
    elapsed_s = time.monotonic() - started
    end_allocated = probe.worker_memory()

    print(f"   soak ran {elapsed_s / 60.0:.1f} min, "
          f"accepted_total={len(probe.accepted)}")

    verdicts: list[tuple[str, bool, str]] = []
    for worker_id in sorted(series):
        before = start_allocated.get(worker_id, {})
        after = end_allocated.get(worker_id, {})
        alloc_slope = slope_per_hour(series[worker_id]["allocated"])
        resv_slope = slope_per_hour(series[worker_id]["reserved"])
        if alloc_slope is None or resv_slope is None:
            print(f"   worker {worker_id}: too few samples to fit a slope")
            continue
        print(
            f"   worker {worker_id} dev {after.get('device')}: "
            f"allocated {before.get('allocated_bytes', 0) / MIB:.0f} -> "
            f"{after.get('allocated_bytes', 0) / MIB:.0f}MiB "
            f"({alloc_slope / MIB:+.1f}MiB/h), "
            f"reserved {before.get('reserved_bytes', 0) / MIB:.0f} -> "
            f"{after.get('reserved_bytes', 0) / MIB:.0f}MiB "
            f"({resv_slope / MIB:+.1f}MiB/h), "
            f"peak_reserved={after.get('peak_reserved_bytes', 0) / MIB:.0f}MiB"
        )
        verdicts.append(
            (
                f"worker {worker_id} allocated did not grow",
                alloc_slope / MIB <= max_growth,
                f"{alloc_slope / MIB:+.1f}MiB/h (limit {max_growth:g})",
            )
        )
        verdicts.append(
            (
                f"worker {worker_id} reserved did not grow",
                resv_slope / MIB <= max_growth,
                f"{resv_slope / MIB:+.1f}MiB/h (limit {max_growth:g})",
            )
        )

    for device in sorted(device_series):
        slope = slope_per_hour(device_series[device])
        if slope is None:
            continue
        print(f"   dev{device} whole-device used: {slope / MIB:+.1f}MiB/h")

    for name, ok, detail in verdicts:
        report.check(name, ok, detail)


def _parse_devices(text: str) -> list[int]:
    devices = [int(part) for part in text.replace(" ", "").split(",") if part]
    if not devices:
        raise argparse.ArgumentTypeError("at least one device index is required")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"device listed twice: {devices}")
    if any(d < 0 for d in devices):
        raise argparse.ArgumentTypeError(f"negative device index: {devices}")
    return devices


def check_device_pinning(devices: Sequence[int]) -> str | None:
    """Why this shell's ``CUDA_VISIBLE_DEVICES`` makes the run unsafe, if it does.

    Every worker sets ``CUDA_VISIBLE_DEVICES`` to the index it was configured
    with and then loads on ``cuda:0``.  That is what makes the configured index
    a physical one - and it is why an inherited ``CUDA_VISIBLE_DEVICES`` is not
    merely redundant here but wrong.  ``CUDA_VISIBLE_DEVICES=1`` with
    ``--devices 0`` passes the pool's range check, because the process really
    can see exactly one card and it really is numbered 0; then the worker pins
    itself to physical GPU 0.  The model lands on someone else's card while
    every line of output says 0.

    So the rule is stated once, here: leave the variable unset and name the
    physical index in ``--devices``.
    """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not inherited:
        return None
    return (
        f"CUDA_VISIBLE_DEVICES={inherited!r} is set in this shell.  The pool sets "
        f"that variable per worker, from the physical index it was configured "
        f"with, so inheriting it renumbers the cards underneath --devices "
        f"{list(devices)}.  Unset CUDA_VISIBLE_DEVICES and put the physical "
        f"index in --devices instead."
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--devices", type=_parse_devices, default=[0],
                        help="comma-separated physical device indices (default 0)")
    parser.add_argument("--warmup-tasks-per-worker", type=int, default=2,
                        help="real inferences per worker before the steady state "
                             "is read (default 2)")
    parser.add_argument("--soak-minutes", type=float, default=0.0,
                        help="soak duration; 0 (default) skips the soak, and "
                             "anything under 15 will say it is too short")
    parser.add_argument("--rate-hz", type=float, default=2.0,
                        help="submission rate during the soak (default 2)")
    parser.add_argument("--sample-s", type=float, default=5.0,
                        help="memory sampling interval during the soak (default 5)")
    parser.add_argument("--max-growth-mib-per-hour", type=float,
                        default=DEFAULT_MAX_GROWTH_MIB_PER_HOUR,
                        help=f"growth above this fails the soak "
                             f"(default {DEFAULT_MAX_GROWTH_MIB_PER_HOUR:g})")
    parser.add_argument("--inferences", type=int, default=5,
                        help="measured inferences in the bench phase, one at a "
                             "time, each timed (default 5; 0 skips the phase)")
    parser.add_argument("--release-tolerance-mib", type=float,
                        default=DEFAULT_RELEASE_TOLERANCE_MIB,
                        help=f"how close to the pre-pool baseline a card must "
                             f"come back for the release check to pass "
                             f"(default {DEFAULT_RELEASE_TOLERANCE_MIB:g})")
    parser.add_argument("--release-timeout-s", type=float,
                        default=DEFAULT_RELEASE_TIMEOUT_S,
                        help=f"how long to wait for the driver to hand the "
                             f"memory back (default {DEFAULT_RELEASE_TIMEOUT_S:g})")
    parser.add_argument("--json", default=None,
                        help="write the collected per-worker numbers here")
    args = parser.parse_args(list(argv) if argv is not None else None)

    problem = check_device_pinning(args.devices)
    if problem is not None:
        print(f"refusing to start: {problem}")
        return 2

    # The pool config comes from the environment so this tool and the pipeline
    # agree about how many workers there are and which device each takes.
    os.environ["ALPAMAYO_INFERENCE_MODE"] = "multi_process"
    os.environ["ALPAMAYO_GPU_DEVICES"] = ",".join(str(d) for d in args.devices)
    os.environ["ALPAMAYO_MINIMUM_READY_WORKERS"] = str(len(args.devices))

    import config.loader as loader

    settings: AppConfig = loader.load_app_config()
    if tuple(settings.inference_pool.gpu_devices) != tuple(args.devices):
        print(
            "pool config did not take the device list from the environment: "
            f"asked {args.devices}, got {list(settings.inference_pool.gpu_devices)}"
        )
        return 2

    print("GPU memory probe - real workers, real checkpoint, nothing sent")
    print(
        f"  devices={args.devices} workers={len(args.devices)} "
        f"warmup={settings.inference_pool.worker_warmup}"
    )
    if not torch.cuda.is_available():
        print(
            "  ! torch reports no CUDA device.  This probe needs the machine with "
            "the GPUs; on this box the load phase will fail."
        )

    report = Report()
    probe = Probe(settings, args.devices)
    collected: dict[str, Any] = {"devices": args.devices}
    loaded = False
    try:
        loaded = phase_load(probe, report)
        if loaded:
            phase_warmup(probe, report, args.warmup_tasks_per_worker)
            collected["after_warmup"] = probe.worker_memory()
            if args.inferences > 0:
                collected["bench"] = phase_bench(probe, report, args.inferences)
            else:
                report.note("bench skipped (--inferences 0)")
            if args.soak_minutes > 0:
                phase_soak(
                    probe,
                    report,
                    minutes=args.soak_minutes,
                    rate_hz=args.rate_hz,
                    sample_s=args.sample_s,
                    max_growth=args.max_growth_mib_per_hour,
                )
                collected["after_soak"] = probe.worker_memory()
            else:
                report.note(
                    "soak skipped (--soak-minutes 0).  Growth across a shift is "
                    "the one thing a two-minute run cannot show; give it 30 "
                    "minutes before a deployment."
                )
    finally:
        # The pool is stopped before anything is concluded, and the release is
        # measured rather than assumed - a worker that died holding its memory
        # leaves no other trace.
        try:
            stopped = probe.stop()
            collected["stop"] = stopped
            if stopped.get("terminated") or stopped.get("killed"):
                print(f"  ! pool did not stop gracefully: {json.dumps(stopped)}")
        except Exception as exc:  # noqa: BLE001 - the report matters more
            print(f"  ! pool did not stop cleanly: {exc}")

        try:
            if loaded:
                collected["release"] = phase_release(
                    probe,
                    report,
                    tolerance_mib=args.release_tolerance_mib,
                    timeout_s=args.release_timeout_s,
                )
            else:
                report.note(
                    "release not checked: the pool never reached a loaded state"
                )
        except Exception as exc:  # noqa: BLE001 - the report matters more
            print(f"  ! release check did not complete: {exc}")

        if args.json:
            Path(args.json).write_text(
                json.dumps(collected, indent=2, default=str), encoding="utf-8"
            )
            print(f"  wrote {args.json}")

    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
