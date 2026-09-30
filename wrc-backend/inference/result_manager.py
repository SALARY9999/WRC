"""The gateway between a GPU worker's answer and the vehicle.

Why a worker's result cannot go straight to the takeover engine
    Worker results come back in completion order, and completion order is not
    task order.  With several workers running complete models side by side, two
    results can arrive as ``1, 3, 2`` - and a trajectory is only meaningful
    relative to the *newest* state the vehicle has.  Accepting task 2 after task
    3 would rewind the takeover state machine, overwrite a newer trajectory and
    send a datagram computed from an older camera window.  So every result
    passes through here first, and only a result that clears every check below
    is handed to the single writer that owns the takeover state and the socket.

The ordering rule is one line
    ``if result.task_seq <= last_accepted_task_seq: drop_as_out_of_order()``
    ``task_seq`` is the only ordering key.  ``capture_timestamp_us`` is a
    *remote* clock - it can step backwards or arrive in a different domain
    after a config change - so it is used to decide whether a result is fresh
    and whether the camera clock stepped backwards, never to decide which of
    two tasks is newer.  A forward gap, however large, is not an anomaly: it
    becomes the new capture reference.

Every check, in the order they run
    1. the worker reported success (a failed result is a result: it travels the
       same path so the worker's slot is released and the failure counted);
    2. the echoed ``task_seq`` matches the task this worker was actually given;
    3. ``task_seq`` is strictly greater than the last accepted one;
    4. the task has not exceeded ``result_max_age_ms`` since it was frozen;
    5. the camera clock has not misbehaved for this task, and the task is not
       more than ``max_capture_lag_ms`` behind the newest capture window seen;
    6. the trajectory has a usable shape;
    7. it has at least one point;
    8. every coordinate is finite.

    A drop at any step is counted under its own name and the result is
    discarded.  Nothing here raises: a malformed result is ordinary control
    flow, and an exception in the validation path would take down the one
    thread that is allowed to send.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from inference.tasks import InferenceResult, InferenceTask

# --------------------------------------------------------------------------- #
# Drop reasons.  Each is a distinct counter in /health: "we are dropping" is
# not actionable, "we are dropping because every result is 4 s old" is.
# --------------------------------------------------------------------------- #

DROP_WORKER_ERROR = "worker_error"
DROP_UNKNOWN_TASK = "unknown_task"
DROP_TASK_SEQ_MISMATCH = "task_seq_mismatch"
DROP_OUT_OF_ORDER = "out_of_order"
DROP_DUPLICATE = "duplicate"
DROP_EXPIRED = "expired"
DROP_CAPTURE_LAG = "capture_lag"
DROP_CAPTURE_ANOMALY = "capture_timestamp_anomaly"
DROP_INVALID_SHAPE = "invalid_trajectory_shape"
DROP_INVALID_POINTS = "invalid_trajectory_points"
DROP_NON_FINITE = "non_finite_trajectory"

#: Camera-clock anomalies, detected when a task is submitted rather than when
#: its result comes back: the clock is a property of the *input*, and flagging
#: it early is what lets the drop be explained instead of mysterious.
CAPTURE_ROLLBACK = "rollback"
#: Kept so ``stats()["capture_jump"]`` stays in the health view.  A forward
#: gap, however large, is not an anomaly and does not increment this counter.
CAPTURE_JUMP = "jump"

#: Minimal trajectory shape the downstream crop and coordinate transform need:
#: ``(T, >=2)`` so ``positions[:, 0]`` and ``positions[:, 1]`` exist.
MIN_TRAJECTORY_COLUMNS = 2
MIN_TRAJECTORY_POINTS = 1


@dataclass(frozen=True)
class ResultVerdict:
    """The result manager's answer about one result.

    ``reason`` is empty exactly when ``accepted`` is True, so a caller can
    branch on either without the two ever disagreeing.
    """

    accepted: bool
    reason: str = ""
    detail: str = ""


ACCEPTED = ResultVerdict(accepted=True)


class ResultManager:
    """Validates worker results and keeps the accepted-sequence watermark.

    Single-threaded by contract: every method is called from the pipeline
    thread, which is also the only thread that touches the takeover engine and
    the waypoint socket.  ``last_accepted_task_seq`` is the watermark that
    implements the out-of-order rule, and it only ever moves forward.
    """

    def __init__(
        self,
        *,
        result_max_age_ms: float = 2500.0,
        max_capture_lag_ms: float = 2500.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.result_max_age_ms = float(result_max_age_ms)
        self.max_capture_lag_ms = float(max_capture_lag_ms)
        self._clock_ns = clock_ns
        self.last_accepted_task_seq: int = -1
        #: worker_id -> the task dispatched to it and not yet answered.
        self._inflight_by_worker: dict[int, InferenceTask] = {}
        #: task_seq -> the task, for results whose worker no longer matches.
        self._dispatched: dict[int, InferenceTask] = {}
        #: task_seq of tasks whose capture timestamp was flagged at submit.
        self._anomalous: set[int] = set()
        #: Newest camera capture timestamp that was *not* flagged as anomalous.
        #: This is the "newest capture window" the lag check compares against.
        self._newest_capture_us: int | None = None
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Submit side
    # ------------------------------------------------------------------ #

    def note_task(self, task: InferenceTask) -> str | None:
        """Register a freshly frozen task; returns a camera anomaly if any.

        Called once per task that ``InputBuilder`` accepted, whether or not a
        worker is free for it - a task that is immediately replaced still moved
        the camera clock forward, and the clock is checked here rather than at
        validation so the anomaly is attributed to the input that caused it.

        A step backwards beyond ``max_capture_lag_ms`` is a rollback: the task
        is flagged and the capture watermark stays put, so one reversed stamp
        does not pull the lag reference into the past.  A forward gap of any
        size is not an anomaly and does advance the watermark.  Refusing a
        large forward gap and then keeping the old reference made every later
        window look anomalous too.
        """
        capture_us = int(task.capture_timestamp_us)
        previous = self._newest_capture_us
        if previous is not None:
            delta_ms = (capture_us - previous) / 1000.0
            if delta_ms < -self.max_capture_lag_ms:
                self._anomalous.add(task.task_seq)
                self._bump(CAPTURE_ROLLBACK)
                return CAPTURE_ROLLBACK
        self._newest_capture_us = capture_us
        return None

    def note_dispatch(self, task: InferenceTask, worker_id: int) -> None:
        """Record that ``worker_id`` is now running ``task``."""
        self._inflight_by_worker[int(worker_id)] = task
        self._dispatched[task.task_seq] = task

    def discard_task(self, task: InferenceTask) -> None:
        """Forget a task that will never produce a result.

        Used for a pending task replaced by a newer one, and for the in-flight
        task of a worker that died.  The anomaly flag is cleared here too: the
        task is gone, so keeping its flag would leak one entry per discarded
        task for the life of the process.
        """
        self._anomalous.discard(task.task_seq)
        self._dispatched.pop(task.task_seq, None)

    def discard_worker(self, worker_id: int) -> InferenceTask | None:
        """Forget whatever ``worker_id`` was running; returns it if anything."""
        task = self._inflight_by_worker.pop(int(worker_id), None)
        if task is not None:
            self.discard_task(task)
        return task

    # ------------------------------------------------------------------ #
    # Result side
    # ------------------------------------------------------------------ #

    def validate(self, result: InferenceResult) -> ResultVerdict:
        """Run every check in order.  Never raises; never mutates on a drop.

        The task is looked up by its *sequence*, not by the worker that reported
        it.  A worker that has already answered is free again and may be holding
        a newer task; if a delayed second copy of the old result consumed that
        newer record, the newer task's real result would arrive to find nothing
        waiting for it and be dropped as unknown.  So a result only ever
        releases the slot belonging to its own sequence.
        """
        now_ns = int(self._clock_ns())
        seq = int(result.task_seq)
        worker_id = int(result.worker_id)
        task = self._dispatched.get(seq)
        inflight = self._inflight_by_worker.get(worker_id)

        def reject(reason: str, detail: str = "") -> ResultVerdict:
            """Count a drop and leave every in-flight record alone.

            Used when the two records disagree: with no way to tell which of
            them is wrong, releasing either would be a guess.
            """
            self._bump(reason)
            return ResultVerdict(accepted=False, reason=reason, detail=detail)

        def drop(reason: str, detail: str = "") -> ResultVerdict:
            """Count a drop and release this task's records - worker slot too."""
            self._bump(reason)
            self._release(worker_id, seq)
            return ResultVerdict(accepted=False, reason=reason, detail=detail)

        # 1. the worker said it failed.  Every failure counts as a worker
        #    error; a CUDA OOM additionally counts under its own name, so the
        #    subset relation is explicit and an OOM rate can be read directly.
        if result.failed:
            self._bump(DROP_WORKER_ERROR)
            if result.error_kind:
                self._bump(result.error_kind)
            self._release(worker_id, seq)
            return ResultVerdict(
                accepted=False,
                reason=DROP_WORKER_ERROR,
                detail=(
                    f"worker={result.worker_id} device={result.device} "
                    f"kind={result.error_kind or 'unknown'} error={result.error}"
                ),
            )

        # 2. the echo matches what this worker was actually given.
        if task is None:
            # Nothing is in flight under this sequence at all.  Either the
            # worker named a task it was never handed, or this is a second copy
            # of a result that has already been accepted.
            if inflight is not None:
                return reject(
                    DROP_TASK_SEQ_MISMATCH,
                    f"worker={worker_id} was running task_seq="
                    f"{inflight.task_seq} but reported task_seq={seq}",
                )
            if seq == self.last_accepted_task_seq:
                return reject(
                    DROP_DUPLICATE, f"task_seq={seq} was already accepted"
                )
            return reject(
                DROP_UNKNOWN_TASK,
                f"task_seq={seq} worker={worker_id} was never dispatched a task",
            )
        if inflight is not None and int(inflight.task_seq) != seq:
            return reject(
                DROP_TASK_SEQ_MISMATCH,
                f"worker={worker_id} was running task_seq={inflight.task_seq} "
                f"but reported task_seq={seq}",
            )

        # 3. the ordering rule.  Everything newer than the watermark is
        #    admissible; anything at or below it has already been overtaken.
        if seq <= self.last_accepted_task_seq:
            # ``==`` is a repeat of an accepted sequence, ``<`` is a result that
            # lost the race.  Both are dropped by the rule above; the split is
            # only so the two show up under different counters.
            if seq == self.last_accepted_task_seq:
                return drop(
                    DROP_DUPLICATE, f"task_seq={seq} was already accepted"
                )
            return drop(
                DROP_OUT_OF_ORDER,
                f"task_seq={seq} <= last_accepted={self.last_accepted_task_seq}",
            )

        # 4. the task is too old to drive the vehicle now.
        age_ms = task.age_ms(now_ns)
        if age_ms > self.result_max_age_ms:
            return drop(
                DROP_EXPIRED,
                f"task_seq={result.task_seq} age_ms={age_ms:.1f} "
                f"limit_ms={self.result_max_age_ms:.1f}",
            )

        # 5. the camera clock was misbehaving, or this window is far behind the
        #    newest one already taken.
        if result.task_seq in self._anomalous:
            return drop(
                DROP_CAPTURE_ANOMALY,
                f"task_seq={result.task_seq} "
                f"capture_timestamp_us={task.capture_timestamp_us}",
            )
        if self._newest_capture_us is not None:
            lag_ms = (
                self._newest_capture_us - task.capture_timestamp_us
            ) / 1000.0
            if lag_ms > self.max_capture_lag_ms:
                return drop(
                    DROP_CAPTURE_LAG,
                    f"task_seq={result.task_seq} capture_lag_ms={lag_ms:.1f} "
                    f"limit_ms={self.max_capture_lag_ms:.1f}",
                )

        positions = getattr(result.model_result, "positions", None)
        # 6. shape.
        if positions is None:
            return drop(DROP_INVALID_SHAPE, "result carried no trajectory")
        array = np.asarray(positions)
        if array.ndim != 2 or array.shape[1] < MIN_TRAJECTORY_COLUMNS:
            return drop(
                DROP_INVALID_SHAPE,
                f"task_seq={result.task_seq} shape={tuple(array.shape)}",
            )
        # 7. at least one usable point.
        if array.shape[0] < MIN_TRAJECTORY_POINTS:
            return drop(
                DROP_INVALID_POINTS,
                f"task_seq={result.task_seq} points={int(array.shape[0])}",
            )
        # 8. finite coordinates.
        if not np.isfinite(array).all():
            return drop(
                DROP_NON_FINITE,
                f"task_seq={result.task_seq} has a non-finite coordinate",
            )

        self.last_accepted_task_seq = int(result.task_seq)
        self._release(worker_id, seq)
        return ACCEPTED

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _forget(self, task_seq: int) -> None:
        self._anomalous.discard(int(task_seq))
        self._dispatched.pop(int(task_seq), None)

    def _release(self, worker_id: int, task_seq: int) -> None:
        """Free the reporting worker's slot and forget the task.

        The slot is only freed when the worker is still recorded as running
        *this* task.  A duplicate of an old result must not empty the slot of
        whatever that worker is running now.
        """
        holder = self._inflight_by_worker.get(int(worker_id))
        if holder is not None and int(holder.task_seq) == int(task_seq):
            self._inflight_by_worker.pop(int(worker_id), None)
        self._forget(task_seq)

    def _bump(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1

    def count(self, name: str) -> int:
        return self._counts.get(name, 0)

    @property
    def inflight_task_seqs(self) -> tuple[int, ...]:
        return tuple(sorted(self._dispatched))

    def stats(self) -> dict[str, int]:
        """The drop breakdown, for the published pool view."""
        counts = dict(self._counts)
        return {
            "out_of_order": counts.get(DROP_OUT_OF_ORDER, 0),
            "duplicate": counts.get(DROP_DUPLICATE, 0),
            "expired": counts.get(DROP_EXPIRED, 0),
            "capture_lag": counts.get(DROP_CAPTURE_LAG, 0),
            "capture_anomaly": counts.get(DROP_CAPTURE_ANOMALY, 0),
            "capture_rollback": counts.get(CAPTURE_ROLLBACK, 0),
            "capture_jump": counts.get(CAPTURE_JUMP, 0),
            "invalid_trajectory_shape": counts.get(DROP_INVALID_SHAPE, 0),
            "invalid_trajectory_points": counts.get(DROP_INVALID_POINTS, 0),
            "non_finite_trajectory": counts.get(DROP_NON_FINITE, 0),
            # ``worker_error`` counts every failed result; ``cuda_oom`` and
            # ``worker_exception`` are its disjoint parts, so this is a subset
            # relation rather than three independent tallies.
            "worker_error": counts.get(DROP_WORKER_ERROR, 0),
            "cuda_oom": counts.get("cuda_oom", 0),
            "worker_exception": counts.get("worker_exception", 0),
            "unknown_task": counts.get(DROP_UNKNOWN_TASK, 0),
            "task_seq_mismatch": counts.get(DROP_TASK_SEQ_MISMATCH, 0),
        }


__all__ = [
    "ACCEPTED",
    "CAPTURE_JUMP",
    "CAPTURE_ROLLBACK",
    "DROP_CAPTURE_ANOMALY",
    "DROP_CAPTURE_LAG",
    "DROP_DUPLICATE",
    "DROP_EXPIRED",
    "DROP_INVALID_POINTS",
    "DROP_INVALID_SHAPE",
    "DROP_NON_FINITE",
    "DROP_OUT_OF_ORDER",
    "DROP_TASK_SEQ_MISMATCH",
    "DROP_UNKNOWN_TASK",
    "DROP_WORKER_ERROR",
    "MIN_TRAJECTORY_COLUMNS",
    "MIN_TRAJECTORY_POINTS",
    "ResultManager",
    "ResultVerdict",
]
