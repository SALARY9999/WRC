"""A worker that runs no model, for testing the pool without a GPU.

Same process body signature as :func:`inference.gpu_pool.worker_main`, so the
scheduler cannot tell the two apart - which is the point: every scheduling,
ordering and shutdown property can then be tested deterministically on a CPU
box, with the real ``GpuScheduler``, the real ``ResultManager`` and the real
message protocol.

Two more knobs inject the failures a real model or a real worker can produce, so
the Result Manager's refusals are exercised across a real process boundary
rather than by calling ``validate`` directly:

    ``corrupt_task_seqs``    ``task_seq`` -> one of ``CORRUPT_KINDS``.  The task
                             still succeeds at the worker; what comes back is a
                             trajectory no downstream stage may act on.
    ``duplicate_task_seqs``  tasks whose result is reported twice, which is the
                             only way to produce a genuine duplicate: the
                             Result Manager sees the same ``task_seq`` accepted
                             once and must refuse the second copy.
    ``exit_after_tasks_by_worker``
                             ``worker_id`` -> how many inferences that one
                             worker runs before it stops answering.  Same
                             effect as ``exit_after_tasks``, but aimed at a
                             single worker, so "one worker dies and the rest
                             carry on" is expressible.
    ``abrupt_exit_after_tasks_by_worker``
                             the same map and the same count, but the worker
                             leaves without announcing ``WORKER_STOPPED``, so
                             the parent has only the process table to go on.
                             The polite form above is a shutdown the scheduler
                             is right not to poll for; only this one exercises
                             the liveness poll and its ``worker_exited`` fault.
                             Set on its own it is enough - the count comes with
                             it.

Three ways to control completion order
    ``delays_ms``            a list of per-inference delays, cycled.  A blunt
                             instrument: every worker's first inference gets the
                             same delay, so it cannot reorder tasks.
    ``delays_by_task_seq``   ``task_seq`` -> delay in ms, falling back to
                             ``delays_ms`` for a sequence with no entry.  This
                             is what forces a *specific* completion order, and
                             like ``fail_task_seqs`` it is a plain dict of
                             numbers, so it survives a pickled trip to a spawned
                             child - which is what makes the ordering property
                             testable across a real process boundary.
    ``gates``                ``worker_id`` (or ``task_seq``) -> a
                             ``threading.Event``.  The worker blocks until the
                             test releases it, so the completion order is chosen
                             exactly rather than hoped for from a sleep.  The
                             sharpest of the three, and in-process only: an
                             ``Event`` cannot be pickled to a spawned child, and
                             it does not need to be.

Nothing here imports torch.  The result it fabricates is the real
:class:`~inference.alpamayo.ModelResult` when that module can be imported and a
plain stand-in otherwise, so a CPU-only box with no torch can still exercise
the whole scheduler.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping

from inference.tasks import (
    ERROR_CUDA_OOM,
    ERROR_WORKER_EXCEPTION,
    GpuMemory,
    InferenceResult,
    InferenceTask,
    WORKER_FAILED,
    WORKER_LOADING,
    WORKER_READY,
    WORKER_STOPPED,
    WorkerStatus,
)

logger = logging.getLogger(__name__)

#: Points in the trajectory a fake worker emits.  The real model emits 64.
DEFAULT_TRAJECTORY_POINTS = 64

#: How long a fake worker waits on its gate before giving up, so a test that
#: forgets to release one fails rather than hanging the suite.
DEFAULT_GATE_TIMEOUT_S = 30.0

#: Trajectory corruptions ``corrupt_task_seqs`` can ask for.  Each one is a
#: distinct refusal in the Result Manager, so they are named for the verdict
#: rather than for the arithmetic that produces them:
#: ``nan``/``inf`` -> ``non_finite_trajectory``, ``shape`` ->
#: ``invalid_trajectory_shape``, ``points`` -> ``invalid_trajectory_points``.
CORRUPT_NAN = "nan"
CORRUPT_INF = "inf"
CORRUPT_SHAPE = "shape"
CORRUPT_POINTS = "points"
CORRUPT_KINDS = (CORRUPT_NAN, CORRUPT_INF, CORRUPT_SHAPE, CORRUPT_POINTS)


@dataclass
class _StandInResult:
    """The subset of ``ModelResult`` the result manager reads.

    Used only when ``inference.alpamayo`` cannot be imported (no torch).  It
    carries the same ``positions`` contract so the shape/point/finite checks
    exercise the same code path.
    """

    positions: Any
    rotations: Any
    t0_us: int
    ref_pose: Any
    cot: str = ""
    input_ids_sha256: str = ""
    input_ids_shape: tuple[int, ...] = ()
    prompt_text: str = ""


def _wrap_result(task: InferenceTask, positions, rotations, cot: str):
    """The real ``ModelResult`` when it can be imported, a stand-in otherwise."""
    t0_us = int(task.capture_timestamp_us)
    try:
        from inference.alpamayo import ModelResult

        return ModelResult(
            cot=cot,
            positions=positions,
            rotations=rotations,
            t0_us=t0_us,
            ref_pose=task.reference_pose,
        )
    except Exception:  # noqa: BLE001 - torch absent: a stand-in is enough here
        return _StandInResult(
            positions=positions,
            rotations=rotations,
            t0_us=t0_us,
            ref_pose=task.reference_pose,
            cot=cot,
        )


def make_model_result(task: InferenceTask, *, points: int = DEFAULT_TRAJECTORY_POINTS):
    """A plausible, finite trajectory in the task's own reference frame.

    Emitted forward along +x at 0.1 s spacing, which is what the real model
    produces in shape: ``(T, 3)`` positions and ``(T, 3, 3)`` rotations.  The
    values are synthetic and deliberately boring - a test that needs a specific
    trajectory should build its own.
    """
    import numpy as np

    count = max(1, int(points))
    steps = np.arange(count, dtype=np.float64) * 0.1
    positions = np.stack(
        [steps * 5.0, np.zeros(count), np.zeros(count)], axis=1
    ).astype(np.float64)
    rotations = np.repeat(
        np.eye(3, dtype=np.float64)[None, :, :], count, axis=0
    )
    return _wrap_result(task, positions, rotations, f"fake worker {task.task_seq}")


def make_corrupt_result(
    task: InferenceTask,
    kind: str,
    *,
    points: int = DEFAULT_TRAJECTORY_POINTS,
):
    """A trajectory the Result Manager must refuse, of the named ``kind``.

    Built from the same base trajectory as :func:`make_model_result` so the only
    difference between an accepted and a refused result is the corruption
    itself - otherwise a refusal could be attributed to the wrong check.
    """
    import numpy as np

    good = make_model_result(task, points=points)
    positions = np.asarray(good.positions, dtype=np.float64)
    rotations = np.asarray(good.rotations, dtype=np.float64)

    if kind == CORRUPT_NAN:
        positions = positions.copy()
        positions[len(positions) // 2, 1] = np.nan
    elif kind == CORRUPT_INF:
        positions = positions.copy()
        positions[len(positions) // 2, 1] = np.inf
    elif kind == CORRUPT_SHAPE:
        # One column: ``positions[:, 1]`` does not exist, so the crop and the
        # coordinate transform have nothing to read.
        positions = positions[:, :1]
    elif kind == CORRUPT_POINTS:
        positions = positions[:0]
        rotations = rotations[:0]
    else:
        raise ValueError(
            f"unknown corruption {kind!r}; expected one of {CORRUPT_KINDS}"
        )
    return _wrap_result(task, positions, rotations, f"fake worker {task.task_seq} {kind}")


def fake_worker_main(
    worker_id: int,
    device: int,
    settings: Any,
    task_queue: Any,
    result_queue: Any,
    options: Mapping[str, Any] | None = None,
) -> None:
    """The fake process body.  Same call shape as the real worker."""
    options = dict(options or {})
    delays = [float(x) for x in (options.get("delays_ms") or [0.0])] or [0.0]
    # A dict keyed by ``task_seq``, so a task's duration does not depend on how
    # many inferences this particular worker has already run.  The cycled list
    # above cannot express "task 4 is the slow one" when four workers each take
    # their own path through it.
    delays_by_seq = {
        int(k): float(v)
        for k, v in (options.get("delays_by_task_seq") or {}).items()
    }
    # ``or {}`` would be wrong here: the caller passes its gate dict by
    # reference and fills it with an entry per task, so at worker start it is
    # legitimately *empty*.  Substituting a new dict at that moment would leave
    # every later gate unset and make the suite pass for the wrong reason.
    gates = options.get("gates")
    if gates is None:
        gates = {}
    gate_timeout_s = float(options.get("gate_timeout_s", DEFAULT_GATE_TIMEOUT_S))
    load_ms = float(options.get("load_ms", 0.0))
    load_failure = options.get("load_failure")
    fail_task_seqs = {int(x) for x in (options.get("fail_task_seqs") or ())}
    oom_task_seqs = {int(x) for x in (options.get("oom_task_seqs") or ())}
    points = int(options.get("trajectory_points", DEFAULT_TRAJECTORY_POINTS))
    corrupt_by_seq = {
        int(k): str(v) for k, v in (options.get("corrupt_task_seqs") or {}).items()
    }
    duplicate_seqs = {int(x) for x in (options.get("duplicate_task_seqs") or ())}
    # ``exit_after_tasks`` is worker-wide, which cannot express "one worker dies
    # and the other three carry on" - the case the scheduler's liveness poll and
    # its failure isolation exist for.  This is the same knob keyed by
    # ``worker_id``, and it is a plain dict of numbers for the same reason
    # ``delays_by_task_seq`` is: it has to survive a pickled trip to a child.
    exit_by_worker = {
        int(k): int(v)
        for k, v in (options.get("exit_after_tasks_by_worker") or {}).items()
    }
    # ``exit_after_tasks_by_worker`` stops the worker *politely*: it leaves the
    # loop and sends ``WORKER_STOPPED``, which the parent reads as an orderly
    # shutdown and deliberately does not poll for.  A real crash sends nothing,
    # and the only way the parent can learn of it is the process table - so the
    # abrupt variant returns without that final status, which is what makes the
    # scheduler's liveness poll reachable from a test at all.
    #
    # Either map carries its own count, so ``abrupt_exit_after_tasks_by_worker``
    # alone is enough to say "this worker runs one inference and then dies".
    # Requiring the polite map as well would leave the count unset and the
    # worker running forever - a fault that silently never fires.
    abrupt_by_worker = {
        int(k): int(v)
        for k, v in (options.get("abrupt_exit_after_tasks_by_worker") or {}).items()
    }
    finish_after = exit_by_worker.get(int(worker_id))
    if finish_after is None:
        finish_after = abrupt_by_worker.get(int(worker_id))
    if finish_after is None:
        finish_after = options.get("exit_after_tasks")
    finish_abruptly = int(worker_id) in abrupt_by_worker or bool(
        options.get("exit_abruptly", False)
    )

    def report(message: Any) -> None:
        try:
            result_queue.put_nowait(message)
        except Exception:  # noqa: BLE001
            pass

    def status(state: str, **extra: Any) -> None:
        report(
            WorkerStatus(
                worker_id=int(worker_id),
                device=int(device),
                state=state,
                at_monotonic_ns=time.monotonic_ns(),
                **extra,
            )
        )

    status(WORKER_LOADING)
    if load_ms:
        time.sleep(load_ms / 1000.0)
    if load_failure:
        status(
            WORKER_FAILED,
            error=str(load_failure),
            error_kind=ERROR_WORKER_EXCEPTION,
        )
        return
    status(WORKER_READY, memory=GpuMemory(allocated_bytes=1, reserved_bytes=2))

    infer_count = 0
    last_infer_ms: float | None = None
    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):  # pragma: no cover - queue torn down
            break
        if task is None:
            break
        if not isinstance(task, InferenceTask):  # pragma: no cover
            continue

        gate = gates.get(task.task_seq, gates.get(int(worker_id)))
        if gate is not None and not gate.wait(timeout=gate_timeout_s):
            logger.error(
                "fake worker %d timed out waiting for its gate on task_seq=%d",
                worker_id,
                task.task_seq,
            )

        delay_ms = delays_by_seq.get(
            int(task.task_seq), delays[infer_count % len(delays)]
        )
        started = time.monotonic()
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        infer_count += 1
        last_infer_ms = elapsed_ms

        if int(task.task_seq) in oom_task_seqs:
            result = InferenceResult(
                task_seq=int(task.task_seq),
                worker_id=int(worker_id),
                device=int(device),
                ok=False,
                error="CUDA out of memory (fake)",
                error_kind=ERROR_CUDA_OOM,
                infer_ms=elapsed_ms,
                finished_monotonic_ns=time.monotonic_ns(),
                memory=GpuMemory(allocated_bytes=1, reserved_bytes=2),
            )
        elif int(task.task_seq) in fail_task_seqs:
            result = InferenceResult(
                task_seq=int(task.task_seq),
                worker_id=int(worker_id),
                device=int(device),
                ok=False,
                error="fake worker exception",
                error_kind=ERROR_WORKER_EXCEPTION,
                infer_ms=elapsed_ms,
                finished_monotonic_ns=time.monotonic_ns(),
                memory=GpuMemory(allocated_bytes=1, reserved_bytes=2),
            )
        else:
            # A corrupt task still *succeeds* at the worker: the corruption is
            # in the trajectory, which is exactly the shape of failure the
            # Result Manager exists to catch.  Reported as a worker error
            # instead it would exercise a different check.
            corruption = corrupt_by_seq.get(int(task.task_seq))
            model_result = (
                make_corrupt_result(task, corruption, points=points)
                if corruption
                else make_model_result(task, points=points)
            )
            result = InferenceResult(
                task_seq=int(task.task_seq),
                worker_id=int(worker_id),
                device=int(device),
                ok=True,
                model_result=model_result,
                infer_ms=elapsed_ms,
                finished_monotonic_ns=time.monotonic_ns(),
                memory=GpuMemory(allocated_bytes=1, reserved_bytes=2),
            )
        report(result)
        if int(task.task_seq) in duplicate_seqs:
            # The same result object, a second time.  The Result Manager must
            # accept the first copy and refuse this one as a duplicate.
            report(result)
        status(
            WORKER_READY,
            task_seq=int(task.task_seq),
            infer_count=infer_count,
            last_infer_ms=last_infer_ms,
            memory=GpuMemory(allocated_bytes=1, reserved_bytes=2),
        )
        if finish_after is not None and infer_count >= int(finish_after):
            if finish_abruptly:
                # Straight out, with no ``WORKER_STOPPED``: the parent's only
                # evidence is that the process is gone.
                return
            # A polite stop - it leaves the loop and announces itself on the
            # way out, so the parent reads a shutdown rather than a death.
            break
    status(WORKER_STOPPED, infer_count=infer_count, last_infer_ms=last_infer_ms)


__all__ = [
    "CORRUPT_INF",
    "CORRUPT_KINDS",
    "CORRUPT_NAN",
    "CORRUPT_POINTS",
    "CORRUPT_SHAPE",
    "DEFAULT_GATE_TIMEOUT_S",
    "DEFAULT_TRAJECTORY_POINTS",
    "fake_worker_main",
    "make_corrupt_result",
    "make_model_result",
]
