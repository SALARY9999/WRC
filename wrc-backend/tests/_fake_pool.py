"""Test doubles for the GPU pool: a fake spawn context, and task builders.

Why an in-process context instead of real processes
    A real ``spawn`` child takes a second to start and reports back when it
    feels like it.  Every scheduling property worth testing - "an idle worker
    takes the task immediately", "a stale pending task is replaced", "results
    arrive 1, 3, 2 and only 3 drives the vehicle" - is about *order*, and order
    chosen by a test is worth more than order observed from a sleep.

    So this module provides a context whose ``Process`` runs the worker body in
    a thread on a standard queue, and the fake worker
    (:mod:`inference.fake_worker`) can be told to block on a per-worker
    ``threading.Event`` until the test releases it.  The completion order is
    then exact and the suite does not depend on timing at all.

    What this does *not* test is that ``spawn`` works and that CUDA survives a
    process boundary - that needs a GPU and belongs to
    ``tools/multi_gpu_offline_test.py``.  The scheduler logic, the message
    protocol, the result manager and the shutdown path are all exercised here.
"""

from __future__ import annotations

import queue as queue_module
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import torch

from camera.models import CameraWindow
from inference.tasks import InferenceTask
from vehicle.models import PoseSample

#: The four Alpamayo camera indices, in physical role order - the same mapping
#: the live pipeline uses.  Tasks in tests carry the real mapping so a camera
#: labelling regression would show up here too.
CAMERA_INDICES = [1, 2, 0, 4]


class FakeQueue:
    """A ``multiprocessing.Queue`` stand-in that can be closed from outside.

    ``close()`` is what lets the fake context's ``terminate()`` actually
    unblock a worker sitting in ``get()``: a Python thread cannot be killed, so
    the queue has to be the thing that ends the wait.  ``get`` therefore polls
    in short slices rather than blocking forever.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._queue: queue_module.Queue = queue_module.Queue()
        self._closed = False

    def put(self, item: Any) -> None:
        self._queue.put(item)

    def put_nowait(self, item: Any) -> None:
        self._queue.put_nowait(item)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._closed:
                raise EOFError(f"{self.name or 'queue'} closed")
            try:
                return self._queue.get(timeout=0.01)
            except queue_module.Empty:
                if not block:
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    raise

    def get_nowait(self) -> Any:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._closed = True

    def join_thread(self) -> None:
        pass


class FakeProcess:
    """A ``multiprocessing.Process`` stand-in backed by a daemon thread."""

    def __init__(
        self,
        target: Callable[..., None] | None = None,
        args: tuple = (),
        name: str | None = None,
        daemon: bool | None = None,
        context: "FakeContext | None" = None,
    ) -> None:
        self._target = target
        self._args = args
        self.name = name or "fake-process"
        self.daemon = daemon
        self._context = context
        self._thread: threading.Thread | None = None
        self._alive = False
        self.exitcode: int | None = None
        #: Set by ``terminate()``/``kill()`` so a test can assert the scheduler
        #: reached for the hard stop rather than waiting forever.
        self.terminated = False
        self.killed = False
        self.started = False

    def start(self) -> None:
        self.started = True
        self._alive = True
        self._thread = threading.Thread(
            target=self._run, name=self.name, daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            assert self._target is not None
            self._target(*self._args)
            self.exitcode = 0
        except BaseException:  # noqa: BLE001 - mirrors a child that died
            self.exitcode = 1
        finally:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def terminate(self) -> None:
        self.terminated = True
        if self._context is not None:
            self._context.close_queues()
        if self._thread is not None:
            self._thread.join(1.0)
        self._alive = False
        if self.exitcode is None:
            self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.terminate()
        self.exitcode = -9


class FakeContext:
    """A ``multiprocessing`` context whose processes are threads."""

    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []
        self.queues: list[FakeQueue] = []

    def Queue(self, *args: Any, **kwargs: Any) -> FakeQueue:
        queue = FakeQueue(name=f"fake-queue-{len(self.queues)}")
        self.queues.append(queue)
        return queue

    def Process(self, target=None, args=(), name=None, daemon=None, **_) -> FakeProcess:
        process = FakeProcess(
            target=target, args=args, name=name, daemon=daemon, context=self
        )
        self.processes.append(process)
        return process

    def close_queues(self) -> None:
        for queue in self.queues:
            queue.close()

    def live_processes(self) -> list[FakeProcess]:
        return [p for p in self.processes if p.is_alive()]


def make_window(
    capture_us: int,
    *,
    cameras: list[int] | None = None,
    frames_per_camera: int = 4,
    height: int = 4,
    width: int = 4,
) -> CameraWindow:
    """A tiny but structurally real camera window.

    ``(N_camera, T, 3, H, W)`` uint8, exactly the shape the model takes, at a
    size that costs nothing - the fake worker never reads the pixels, and a test
    that did would be testing the decoder, which has its own suite.
    """
    indices = list(CAMERA_INDICES if cameras is None else cameras)
    frames = torch.zeros(
        (len(indices), frames_per_camera, 3, height, width), dtype=torch.uint8
    )
    return CameraWindow(
        t0_us=int(capture_us),
        frames=frames,
        camera_indices=indices,
        timestamps_us=[int(capture_us)] * frames_per_camera,
        frame_ids=[["cam%d:t" % i] * frames_per_camera for i in indices],
        frame_timestamps_us=[[int(capture_us)] * frames_per_camera for _ in indices],
        frame_arrivals_us=[[int(capture_us)] * frames_per_camera for _ in indices],
    )


def make_built(window: CameraWindow, *, steps: int = 16) -> SimpleNamespace:
    """The accepted ``InputBuildResult`` fields a task is frozen from."""
    reference = PoseSample(
        timestamp_us=int(window.t0_us),
        latitude=30.5,
        longitude=114.3,
        altitude=20.0,
        heading_deg=90.0,
    )
    return SimpleNamespace(
        ego_xyz=torch.zeros((1, 1, steps, 3), dtype=torch.float32),
        ego_rot=torch.zeros((1, 1, steps, 3, 3), dtype=torch.float32),
        reference_pose=reference,
        reference_payload_timestamp_us=int(window.t0_us),
        ego_history=[
            {"timestamp_us": int(window.t0_us), "x_m": 0.0, "y_m": 0.0}
            for _ in range(steps)
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


def make_task(
    task_seq: int,
    *,
    capture_us: int | None = None,
    created_ns: int | None = None,
    cycle_id: int = 0,
    steps: int = 16,
) -> InferenceTask:
    """One complete, frozen task - the scheduling unit, built cheaply."""
    capture = int(task_seq * 100_000 if capture_us is None else capture_us)
    window = make_window(capture)
    return InferenceTask.from_build(
        task_seq,
        make_built(window, steps=steps),
        window,
        cycle_id=cycle_id,
        now_ns=(
            time.monotonic_ns() if created_ns is None else int(created_ns)
        ),
    )


__all__ = [
    "CAMERA_INDICES",
    "FakeContext",
    "FakeProcess",
    "FakeQueue",
    "make_built",
    "make_task",
    "make_window",
]
