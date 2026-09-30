"""What the GPU pool schedules, and what a worker hands back.

The scheduling unit is a **complete model input window**, never a single
image.  :class:`InferenceTask` freezes everything one ``AlpamayoRunner.infer``
call needs - the camera window, the two ego tensors, the reference pose and the
input diagnostics the inference package records - so that a worker can run the
model without reading the pipeline's live pose history, camera buffers or
planned-route cache.  Those are all owned by the pipeline thread, and a worker
process that reached into them would be a second reader of state that only has
one writer.

``task_seq`` is the identity
    It is assigned by the pipeline, once, at the moment ``InputBuilder``
    produced an accepted input - never for a rejected cycle, because a rejected
    cycle has no task.  It is strictly increasing for the life of the process,
    and it is the *only* thing that orders two tasks.  ``capture_timestamp_us``
    is carried alongside it for freshness, end-to-end data age and camera-clock
    anomaly detection; it is never used to decide which task is newer, because
    the camera clock is a remote clock that can step backwards.

Immutability
    Both dataclasses are ``frozen``.  :func:`freeze_window` additionally copies
    the window's list fields so the task cannot be mutated by the provider
    reusing a buffer - ``CameraWindow`` is a plain dataclass built fresh per
    ``latest_window()`` call, and this makes that an invariant rather than an
    assumption.

Nothing here imports torch at runtime.  That is deliberate: the scheduler and
the result manager are tested with fake workers and no CUDA, so this module has
to be importable on a box without a GPU stack.  ``ego_xyz``/``ego_rot`` are
annotated as ``Any`` for that reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from camera.models import CameraWindow
    from inference.alpamayo import ModelResult
    from vehicle.models import PoseSample

# --------------------------------------------------------------------------- #
# Worker lifecycle
# --------------------------------------------------------------------------- #

#: Loading the checkpoint.  Nothing is dispatched to a worker in this state.
WORKER_LOADING = "loading"
#: Idle and able to accept a task.
WORKER_READY = "ready"
#: Running a task right now.
WORKER_BUSY = "busy"
#: Crashed, or failed to load.  Never recovered automatically - a failed worker
#: stays visible in the pool view so an operator sees a pool running degraded
#: rather than a pool silently short of capacity.
WORKER_FAILED = "failed"
#: Stopped on request.
WORKER_STOPPED = "stopped"
#: A network worker missed its deadline or could not be reached.  Distinct from
#: ``failed``: a failed local process stays failed, while an unreachable network
#: worker may accept work again only after ``GET /health`` explicitly reports
#: ``ready``.  The timeout path itself must not make that transition.
WORKER_UNREACHABLE = "unreachable"

#: The error kinds a worker can report.  Distinct strings, because the counter
#: an operator watches is different for each: an OOM is a memory budget problem
#: and a generic exception is a code problem.
ERROR_CUDA_OOM = "cuda_oom"
ERROR_WORKER_EXCEPTION = "worker_exception"


@dataclass(frozen=True)
class GpuMemory:
    """The worker's own view of its device, as of the message that carries it.

    Reported in bytes and read from the worker process, never from the parent:
    ``torch.cuda.memory_allocated`` is per-process, so the parent's value would
    say nothing about the model that worker loaded.
    """

    allocated_bytes: int = 0
    reserved_bytes: int = 0
    max_allocated_bytes: int = 0
    max_reserved_bytes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "allocated_bytes": int(self.allocated_bytes),
            "reserved_bytes": int(self.reserved_bytes),
            "max_allocated_bytes": int(self.max_allocated_bytes),
            "max_reserved_bytes": int(self.max_reserved_bytes),
        }


@dataclass(frozen=True)
class WorkerStatus:
    """One worker's lifecycle state, published to the parent.

    Sent on every transition (load started, ready, failed, stopped) and on
    every result, so the pool view in ``/health`` is never more than one
    inference stale.
    """

    worker_id: int
    device: int
    state: str
    task_seq: int | None = None
    infer_count: int = 0
    last_infer_ms: float | None = None
    error: str | None = None
    error_kind: str = ""
    memory: GpuMemory = field(default_factory=GpuMemory)
    at_monotonic_ns: int = 0

    @property
    def failed(self) -> bool:
        return self.state == WORKER_FAILED


# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #


def freeze_window(window: "CameraWindow") -> "CameraWindow":
    """A copy of ``window`` whose list fields cannot be mutated by the provider.

    ``dataclasses.replace`` alone copies the dataclass but re-uses the same list
    objects, so a later ``append`` on the provider's buffer would be visible
    inside a task that was supposed to be frozen.  Every list-valued field is
    rebuilt here; the frame tensor is shared on purpose (it is written once, at
    build time, and never mutated).
    """
    def _copy_rows(rows: Any) -> Any:
        if rows is None:
            return None
        return [list(row) if isinstance(row, (list, tuple)) else row for row in rows]

    return replace(
        window,
        camera_indices=list(window.camera_indices),
        timestamps_us=list(window.timestamps_us),
        frame_ids=_copy_rows(window.frame_ids),
        frame_timestamps_us=_copy_rows(window.frame_timestamps_us),
        frame_arrivals_us=_copy_rows(window.frame_arrivals_us),
        jpeg_bytes=_copy_rows(getattr(window, "jpeg_bytes", None)),
    )


@dataclass(frozen=True)
class InferenceTask:
    """One frozen, complete model input.

    Built by the pipeline thread after ``InputBuilder`` accepted a cycle, and
    handed to exactly one worker.  Everything a worker needs is here; a worker
    that reads anything else is a bug, because the alternative is a worker
    reading state another thread is writing.
    """

    # -- identity and ordering --------------------------------------------- #
    #: The only ordering key.  Strictly increasing, unique for the process.
    task_seq: int
    #: ``window.t0_us`` - the camera clock instant this window is aligned on.
    #: Used for freshness, data age and camera-clock anomaly detection ONLY.
    capture_timestamp_us: int
    #: When the pipeline froze this task, on the monotonic clock.  The result
    #: manager ages results against this, not against the camera clock.
    created_monotonic_ns: int
    #: The cycle that produced it, for log correlation only.
    cycle_id: int = 0

    # -- the model call's arguments ---------------------------------------- #
    window: "CameraWindow" = None  # type: ignore[assignment]
    ego_xyz: Any = None
    ego_rot: Any = None
    reference_pose: "PoseSample" = None  # type: ignore[assignment]

    # -- input diagnostics, for the inference package ----------------------- #
    reference_payload_timestamp_us: int | None = None
    ego_history: tuple[Mapping[str, Any], ...] = ()
    camera_frame_ids: tuple[Mapping[str, Any], ...] = ()
    camera_frame_id: Mapping[str, Any] = field(default_factory=dict)
    latest_frame_ids: tuple[Any, ...] = ()
    camera_age_ms: float = 0.0
    camera_pose_delta_ms: float = 0.0
    camera_lag_ms: float | None = None

    @classmethod
    def from_build(
        cls,
        task_seq: int,
        built: Any,
        window: "CameraWindow",
        *,
        cycle_id: int = 0,
        now_ns: int | None = None,
    ) -> "InferenceTask":
        """Freeze one accepted :class:`InputBuildResult` into a task.

        ``built`` is duck-typed rather than imported: this module stays free of
        ``inference.input_builder`` so the scheduler can be exercised with a
        five-line stand-in for the builder.  Only the caller's ``accepted``
        check decides whether a task exists at all.
        """
        return cls(
            task_seq=int(task_seq),
            capture_timestamp_us=int(window.t0_us),
            created_monotonic_ns=(
                time.monotonic_ns() if now_ns is None else int(now_ns)
            ),
            cycle_id=int(cycle_id),
            window=freeze_window(window),
            ego_xyz=built.ego_xyz,
            ego_rot=built.ego_rot,
            reference_pose=built.reference_pose,
            reference_payload_timestamp_us=built.reference_payload_timestamp_us,
            ego_history=tuple(built.ego_history or ()),
            camera_frame_ids=tuple(built.camera_frame_ids or ()),
            camera_frame_id=dict(built.camera_frame_id or {}),
            latest_frame_ids=tuple(built.latest_frame_ids or ()),
            camera_age_ms=float(built.camera_age_ms),
            camera_pose_delta_ms=float(built.camera_pose_delta_ms),
            camera_lag_ms=built.camera_lag_ms,
        )

    def age_ms(self, now_ns: int) -> float:
        """How long this task has existed, on the monotonic clock."""
        return (int(now_ns) - self.created_monotonic_ns) / 1e6


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InferenceResult:
    """What a worker sends back.  One message per task, success or failure.

    A failure is a *result*, not an exception: it travels the same queue so the
    scheduler can release the worker's slot and count the failure, instead of
    discovering a dead worker by watching a process exit code.

    ``task_seq`` and ``worker_id`` are echoed rather than inferred.  The
    scheduler matches the echo against the task it dispatched; a mismatch is a
    protocol bug and the result is dropped rather than guessed at.
    """

    task_seq: int
    worker_id: int
    device: int
    ok: bool
    #: ``None`` when ``ok`` is False.  A :class:`ModelResult` is a plain
    #: dataclass of numpy arrays and a ``PoseSample``, so it pickles.
    model_result: "ModelResult | None" = None
    error: str | None = None
    error_kind: str = ""
    #: Pure model time inside the worker, measured around ``infer`` only.
    infer_ms: float = 0.0
    #: When the worker finished, on the monotonic clock of... the worker.  Both
    #: processes share the same monotonic origin (fork/exec inherits it on
    #: Linux), so this is comparable with the parent's clock; it is used only
    #: for diagnostics, never for ordering - that is ``task_seq``.
    finished_monotonic_ns: int = 0
    memory: GpuMemory = field(default_factory=GpuMemory)

    @property
    def failed(self) -> bool:
        return not self.ok

    @property
    def is_cuda_oom(self) -> bool:
        return self.error_kind == ERROR_CUDA_OOM


__all__ = [
    "ERROR_CUDA_OOM",
    "ERROR_WORKER_EXCEPTION",
    "GpuMemory",
    "InferenceResult",
    "InferenceTask",
    "WORKER_BUSY",
    "WORKER_FAILED",
    "WORKER_LOADING",
    "WORKER_READY",
    "WORKER_STOPPED",
    "WORKER_UNREACHABLE",
    "WorkerStatus",
    "freeze_window",
]
