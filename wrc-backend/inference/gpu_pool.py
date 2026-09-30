"""Many GPUs, one complete model instance each, and the scheduler between them.

The shape of this module
    The pipeline thread freezes a complete :class:`InferenceTask` and calls
    :meth:`GpuScheduler.submit`.  The scheduler owns one *process* per GPU; each
    process loads one whole Alpamayo checkpoint, warms up, and runs one task at
    a time.  Results come back on a shared queue and are validated by
    :class:`~inference.result_manager.ResultManager` before anything downstream
    sees them.

Why processes and not threads
    Two threads sharing one CUDA model is one model with two callers, not two
    models: they serialise on the GIL and on the CUDA stream, so the throughput
    is the single-GPU throughput plus contention.  A second *process* with its
    own device is the only thing that actually doubles the inference rate.

Why ``spawn`` and not ``fork``
    ``fork`` after CUDA has been initialised is undefined behaviour - the child
    inherits a half-copied CUDA context that no driver call can be made
    against, and the failure surfaces later as an opaque ``cudaError`` in an
    unrelated call.  ``spawn`` starts a fresh interpreter, so a worker's CUDA
    state is created inside the worker.  Nothing in this module forks.

Device selection
    ``gpu_devices`` are *physical* indices, the numbers ``nvidia-smi`` shows.
    Each worker sets ``CUDA_VISIBLE_DEVICES`` to its own index before it touches
    CUDA, so inside the worker the model always lands on ``cuda:0`` and cannot
    accidentally allocate on a neighbour's card.  A ``CUDA_VISIBLE_DEVICES``
    already set in the parent environment is overridden, and that is logged -
    silently ignoring the operator's value would be worse than either choice.

Scheduling policy: newest task wins
    When a worker is free, a task is dispatched to it at once.  When none is,
    the incoming task replaces the pending one (see ``latest_only``): the
    vehicle cares about the newest camera window, not about every window in
    order, and a FIFO backlog would make every queued result stale before it
    ran.  A task already running on a GPU is never cancelled - the cost of a
    half-finished inference is already paid, and the result manager is what
    decides whether its answer is still worth using.

Nothing here imports torch at module import time
    The worker body imports it, inside the child, after ``CUDA_VISIBLE_DEVICES``
    has been set.  The parent needs no CUDA at all in ``multi_process`` mode,
    which is what lets the whole scheduler be tested with no GPU present.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue as queue_module
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

from inference.metrics import InferenceMetrics, LatencyStats
from inference.network_client import UrllibTransport
from inference.network_protocol import (
    NetworkFault,
    NetworkHealth,
    build_infer_body,
    health_is_ready,
    to_inference_result,
)
from inference.result_manager import ResultManager
from inference.tasks import (
    ERROR_CUDA_OOM,
    ERROR_WORKER_EXCEPTION,
    GpuMemory,
    InferenceResult,
    InferenceTask,
    WORKER_BUSY,
    WORKER_FAILED,
    WORKER_LOADING,
    WORKER_READY,
    WORKER_STOPPED,
    WORKER_UNREACHABLE,
    WorkerStatus,
)

from config.loader import DEFAULT_HF_ENDPOINT

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from config.loader import AppConfig

logger = logging.getLogger("app")

#: Where the model lives inside a worker, after ``CUDA_VISIBLE_DEVICES`` has
#: pinned the process to one physical device.
WORKER_DEVICE = "cuda:0"

#: The scheduler's own bookkeeping, not a drop reason: a pending task replaced
#: by a newer one is the latest-only policy working, and it is counted here so
#: "how much work are we throwing away to stay current" has an answer.
DROP_PENDING_REPLACED = "pending_replaced"

#: Reason strings ``submit`` returns, so a caller can log the policy's decision.
SUBMIT_DISPATCHED = "dispatched"
SUBMIT_QUEUED = "queued"
SUBMIT_REPLACED = "replaced"
SUBMIT_DROPPED = "dropped"


# --------------------------------------------------------------------------- #
# GPU inventory
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str = ""
    total_bytes: int | None = None
    free_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
        }


@dataclass(frozen=True)
class GpuInventory:
    """What the box has, as far as we could tell without disturbing it.

    ``available`` is False when no inventory could be read at all - which is
    reported, never treated as "no GPUs".  The distinction matters: a missing
    ``pynvml`` on a working eight-GPU box must not look like an empty box, so
    the pool bounds its device check by what it could actually observe.
    """

    available: bool
    source: str
    devices: tuple[GpuInfo, ...] = ()
    error: str | None = None

    @property
    def count(self) -> int | None:
        return len(self.devices) if self.available else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "source": self.source,
            "devices": [d.as_dict() for d in self.devices],
            "error": self.error,
        }


def probe_gpu_inventory() -> GpuInventory:
    """Read the device list without creating a CUDA context or allocating.

    NVML first, because it is read-only and allocates nothing: creating a CUDA
    context on a 24 GB card that is about to host a 10B checkpoint costs real
    memory, and the whole point of this probe is to run *before* that.

    The torch fallback is kept because a box can have a working CUDA runtime and
    no ``pynvml``.  It reports names and total memory and **not** free memory,
    because every way of asking torch for free memory needs an active context -
    and on a shared box that is worse than not knowing.  Measured on the
    eight-GPU deployment machine, ``torch.cuda.mem_get_info`` for each index
    took ~390MiB *on every card in the machine*, including a card another user
    was running a job on.  A startup path that must be safe to call before
    anything is committed cannot be the thing that commits it.
    """
    delta = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            devices = []
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                devices.append(
                    GpuInfo(
                        index=index,
                        name=str(name),
                        total_bytes=int(memory.total),
                        free_bytes=int(memory.free),
                    )
                )
            return GpuInventory(available=True, source="pynvml", devices=tuple(devices))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - shutdown must not mask the result
                pass
    except Exception as exc:  # noqa: BLE001 - fall through to torch
        nvml_error = f"{type(exc).__name__}: {exc}"
    else:  # pragma: no cover - unreachable, the try block always returns
        nvml_error = ""

    try:
        import torch

        if not torch.cuda.is_available():
            return GpuInventory(
                available=True,
                source="torch",
                devices=(),
                error=(
                    "torch reports no CUDA device"
                    + (f"; CUDA_VISIBLE_DEVICES={delta!r}" if delta else "")
                ),
            )
        devices = []
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            # Total comes from the static device properties, which - measured on
            # the deployment box - allocates nothing and creates no context.
            # ``torch.cuda.mem_get_info`` does the opposite: it needs an active
            # context, so asking it about each card creates one on each card, a
            # few hundred MiB apiece, machine-wide and including cards belonging
            # to other people.  Free memory is therefore left unknown here;
            # ``free_bytes is None`` already means "not established" to every
            # caller, and the NVML branch above and nvidia-smi both answer it
            # without touching the device.
            try:
                total: int | None = int(
                    torch.cuda.get_device_properties(index).total_memory
                )
            except Exception:  # noqa: BLE001 - a name alone is still useful
                total = None
            devices.append(
                GpuInfo(
                    index=index,
                    name=str(name),
                    total_bytes=total,
                    free_bytes=None,
                )
            )
        return GpuInventory(available=True, source="torch", devices=tuple(devices))
    except Exception as exc:  # noqa: BLE001 - an unreadable inventory is a reportable state
        return GpuInventory(
            available=False,
            source="none",
            error=f"pynvml: {nvml_error}; torch: {type(exc).__name__}: {exc}",
        )


def validate_pool_startup(
    pool_config: Any, inventory: GpuInventory | None = None
) -> str | None:
    """Everything that must hold before a worker is spawned; ``None`` if fine.

    Two halves.  The first is the config's own structural check
    (:meth:`InferencePoolConfig.validate`), which the loader already ran - it is
    repeated here because a caller can build a config object directly, and a
    scheduler that starts on an invalid config is a worse outcome than a
    duplicated line.

    The second is the one check the loader *cannot* do: whether the named
    devices exist on this box.  ``gpu_devices=[1,3,4,9]`` on a four-GPU machine
    should be a startup error naming device 9, not a pool where one worker dies
    on a CUDA call nobody reads.  Only asserted when an inventory could actually
    be read - "no inventory" must not be reported as "no GPUs".
    """
    validator = getattr(pool_config, "validate", None)
    if callable(validator):
        problem = validator()
        if problem is not None:
            return problem
    devices = [int(d) for d in getattr(pool_config, "gpu_devices", ())]
    if not devices:
        return "inference_pool.gpu_devices must name at least one GPU"
    # ``devices`` is required as well as ``available``: an inventory that read
    # successfully and found *no* devices ("torch reports no CUDA device") is
    # not evidence that the configured indices are out of range, and treating it
    # as such would turn a missing driver into a device-number error.
    if inventory is not None and inventory.available and inventory.devices:
        count = len(inventory.devices)
        out_of_range = sorted({d for d in devices if d >= count})
        if out_of_range:
            return (
                f"inference_pool.gpu_devices names {out_of_range}, but this box "
                f"reports {count} visible device(s) "
                f"({inventory.source}).  Check CUDA_VISIBLE_DEVICES."
            )
    return None


# --------------------------------------------------------------------------- #
# The worker process
# --------------------------------------------------------------------------- #


def _report(result_queue: Any, message: Any) -> None:
    """Best-effort publish from the child.  Must never take the worker down."""
    try:
        result_queue.put_nowait(message)
    except Exception:  # noqa: BLE001 - a failed status report is not a failed run
        pass


def _gpu_memory(torch_module: Any) -> GpuMemory:
    try:
        return GpuMemory(
            allocated_bytes=int(torch_module.cuda.memory_allocated()),
            reserved_bytes=int(torch_module.cuda.memory_reserved()),
            max_allocated_bytes=int(torch_module.cuda.max_memory_allocated()),
            max_reserved_bytes=int(torch_module.cuda.max_memory_reserved()),
        )
    except Exception:  # noqa: BLE001 - diagnostics must not fail an inference
        return GpuMemory()


def _warm_up(runner: Any, settings: "AppConfig", torch_module: Any) -> None:
    """One throwaway inference so the first real result is not the slow one.

    The first call through the processor and the sampling loop pays for CUDA
    kernel selection, autotuning and allocator growth.  On a 10B model that is
    seconds, and paying it on the first *real* task means that task's result can
    exceed ``result_max_age_ms`` and be dropped - the pool would look broken for
    its first cycle.  Failures are logged and swallowed: a warm-up that does not
    work is a slow first task, not a wrong answer.
    """
    from camera.models import CameraWindow

    cameras = list(settings.cimg_camera_indices)
    frames_per_camera = int(settings.frames_per_camera)
    height = int(settings.frame_height)
    width = int(settings.frame_width)
    frames = torch_module.zeros(
        (len(cameras), frames_per_camera, 3, height, width), dtype=torch_module.uint8
    )
    steps = int(settings.history_steps)
    window = CameraWindow(
        t0_us=0,
        frames=frames,
        camera_indices=cameras,
        timestamps_us=[0] * frames_per_camera,
        frame_ids=[["warm-up"] * frames_per_camera for _ in cameras],
    )
    ego_xyz = torch_module.zeros((1, 1, steps, 3), dtype=torch_module.float32)
    ego_rot = (
        torch_module.eye(3, dtype=torch_module.float32)
        .reshape(1, 1, 1, 3, 3)
        .repeat(1, 1, steps, 1, 1)
    )
    from vehicle.models import PoseSample

    runner.infer(
        window,
        ego_xyz,
        ego_rot,
        PoseSample(
            timestamp_us=0, latitude=0.0, longitude=0.0, altitude=0.0, heading_deg=0.0
        ),
    )


def worker_main(
    worker_id: int,
    device: int,
    settings: "AppConfig",
    task_queue: Any,
    result_queue: Any,
    options: Mapping[str, Any] | None = None,
) -> None:
    """The worker process body: load one whole model, then serve tasks.

    Deliberately narrow.  It loads, warms up, runs ``infer`` and reports.  It
    does not decide a takeover, crop a trajectory, convert coordinates or touch
    the waypoint socket: those are the pipeline thread's, and keeping them there
    is what makes them single-writer.  A worker has no route cache, no pose
    history and no socket, so it *cannot* make a control decision even by
    mistake.
    """
    options = dict(options or {})
    # Before anything touches CUDA: pin this process to its own physical device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(device))
    # The HF mirror defaults the model loader relies on.  Set here rather than
    # in the parent so a worker's environment is readable in one place.  The
    # value arrives in ``options`` from the scheduler's settings, with the
    # loader's default as the fallback for a directly-spawned worker.
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault(
        "HF_ENDPOINT", options.get("hf_endpoint", DEFAULT_HF_ENDPOINT)
    )

    def status(state: str, **extra: Any) -> None:
        _report(
            result_queue,
            WorkerStatus(
                worker_id=int(worker_id),
                device=int(device),
                state=state,
                at_monotonic_ns=time.monotonic_ns(),
                **extra,
            ),
        )

    status(WORKER_LOADING)
    try:
        import torch

        from inference.alpamayo import AlpamayoRunner
    except Exception as exc:  # noqa: BLE001 - report, never traceback into the void
        status(
            WORKER_FAILED,
            error=f"import failed: {type(exc).__name__}: {exc}",
            error_kind=ERROR_WORKER_EXCEPTION,
        )
        return

    try:
        runner = AlpamayoRunner(settings, device=WORKER_DEVICE)
        runner.ensure_loaded()
        if options.get("warmup", True):
            try:
                _warm_up(runner, settings, torch)
            except Exception as exc:  # noqa: BLE001 - a slow first task, not a fault
                logger.warning(
                    "worker %d warm-up inference failed (%s: %s); the first real "
                    "task will pay the warm-up cost",
                    worker_id,
                    type(exc).__name__,
                    exc,
                )
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        status(
            WORKER_FAILED,
            error=f"load failed: {type(exc).__name__}: {exc}",
            error_kind=ERROR_WORKER_EXCEPTION,
            memory=_gpu_memory(torch),
        )
        return

    status(WORKER_READY, memory=_gpu_memory(torch))

    infer_count = 0
    last_infer_ms: float | None = None
    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):
            break
        if task is None:
            break
        if not isinstance(task, InferenceTask):  # pragma: no cover - protocol guard
            logger.error("worker %d received %r; ignoring", worker_id, type(task))
            continue
        started = time.monotonic()
        try:
            model_result = runner.infer(
                task.window, task.ego_xyz, task.ego_rot, task.reference_pose
            )
            elapsed_ms = (time.monotonic() - started) * 1000.0
            infer_count += 1
            last_infer_ms = elapsed_ms
            result = InferenceResult(
                task_seq=int(task.task_seq),
                worker_id=int(worker_id),
                device=int(device),
                ok=True,
                model_result=model_result,
                infer_ms=elapsed_ms,
                finished_monotonic_ns=time.monotonic_ns(),
                memory=_gpu_memory(torch),
            )
        except Exception as exc:  # noqa: BLE001 - every failure is a *result*
            elapsed_ms = (time.monotonic() - started) * 1000.0
            oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(
                exc
            ).lower()
            # The failed allocation is still held by the caching allocator.
            # Releasing it here is what lets the worker survive one OOM and take
            # the next task instead of dying on the one after.
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            result = InferenceResult(
                task_seq=int(task.task_seq),
                worker_id=int(worker_id),
                device=int(device),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                error_kind=ERROR_CUDA_OOM if oom else ERROR_WORKER_EXCEPTION,
                infer_ms=elapsed_ms,
                finished_monotonic_ns=time.monotonic_ns(),
                memory=_gpu_memory(torch),
            )
        _report(result_queue, result)
        status(
            WORKER_READY,
            task_seq=int(task.task_seq),
            infer_count=infer_count,
            last_infer_ms=last_infer_ms,
            memory=result.memory,
        )

    status(
        WORKER_STOPPED,
        infer_count=infer_count,
        last_infer_ms=last_infer_ms,
        memory=_gpu_memory(torch),
    )


# --------------------------------------------------------------------------- #
# The scheduler
# --------------------------------------------------------------------------- #


@dataclass
class _WorkerHandle:
    """Parent-side bookkeeping for one worker process."""

    worker_id: int
    device: int
    process: Any
    task_queue: Any
    #: ``local`` uses ``process`` and ``task_queue``.  ``network`` has neither:
    #: its liveness is ``GET /health``, and it must not be asked
    #: ``process.is_alive()``.
    kind: str = "local"
    endpoint: str = ""
    #: Monotonic deadline for the request this slot is running.  Network only.
    deadline: float | None = None
    last_health_at: float = 0.0
    health_inflight: bool = False
    state: str = WORKER_LOADING
    task: InferenceTask | None = None
    infer_count: int = 0
    last_infer_ms: float | None = None
    error: str | None = None
    error_kind: str = ""
    memory: GpuMemory = field(default_factory=GpuMemory)
    spawned_at: float = 0.0
    #: This worker's own inference-time summary.  One ``LatencyStats`` per
    #: handle rather than a pooled one: the whole point of a per-GPU view is to
    #: see one device running hot while the others are fine, and an average
    #: across four workers hides exactly that.  Bounded, so a soak does not
    #: accumulate samples.  Fed from ``_handle_result`` - the same single place
    #: ``note_raw_inference`` is called, so the sample count and the raw
    #: inference count move together.
    infer_stats: LatencyStats = field(default_factory=LatencyStats)

    @property
    def idle(self) -> bool:
        return self.state == WORKER_READY and self.task is None


@dataclass(frozen=True)
class WorkerView:
    """One worker as ``/health`` reports it."""

    worker_id: int
    device: int
    state: str
    task_seq: int | None = None
    infer_count: int = 0
    last_infer_ms: float | None = None
    error: str | None = None
    memory: GpuMemory = field(default_factory=GpuMemory)
    #: Inference-time summary over this worker's own results.  ``None`` until the
    #: first result, so "no data yet" is never rendered as "fast"; and the
    #: percentiles are over ``infer_samples``, which is reported next to them so
    #: a P95 over four samples cannot be mistaken for a P95 over four hundred.
    avg_infer_ms: float | None = None
    p50_infer_ms: float | None = None
    p95_infer_ms: float | None = None
    infer_samples: int = 0
    endpoint: str = ""

    @property
    def failed(self) -> bool:
        return self.state == WORKER_FAILED

    @property
    def ready(self) -> bool:
        return self.state == WORKER_READY

    @property
    def busy(self) -> bool:
        return self.state == WORKER_BUSY

    @property
    def loading(self) -> bool:
        return self.state == WORKER_LOADING

    @property
    def peak_allocated_bytes(self) -> int:
        """Peak since this worker started; ``allocated_bytes`` is right now.

        A ``reserved_bytes`` that climbs while ``allocated_bytes`` is flat is
        normal caching.  Both climbing across a soak is the growth the memory
        probe exists to catch, and this pair is what says which one moved.
        """
        return int(self.memory.max_allocated_bytes or self.memory.allocated_bytes)


class GpuScheduler:
    """Owns the worker processes, the pending slot and the result drain.

    Every method is called from the pipeline thread.  That is not a limitation
    to work around: it is what makes ``submit`` and ``tick`` atomic with respect
    to each other, so "was a worker idle at the moment this task arrived" has
    one answer.
    """

    def __init__(
        self,
        *,
        settings: "AppConfig",
        pool_config: Any,
        metrics: InferenceMetrics,
        result_manager: ResultManager,
        on_accepted: Callable[[InferenceResult, InferenceTask], None],
        context: Any | None = None,
        worker_entry: Callable[..., None] | None = None,
        worker_options: Mapping[str, Any] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        clock: Callable[[], float] = time.monotonic,
        transport: Any | None = None,
    ) -> None:
        self.settings = settings
        self.config = pool_config
        self.metrics = metrics
        self.result_manager = result_manager
        self.on_accepted = on_accepted
        self._clock_ns = clock_ns
        self._clock = clock
        self._transport = transport or UrllibTransport()
        #: Shared by ``submit`` and ``_fill_idle``.  ``None`` until the first
        #: dispatch, so a pool that has never sent a task is not waiting out
        #: an interval that never started.
        self._last_dispatch_at: float | None = None
        self._context = context or multiprocessing.get_context("spawn")
        self._worker_entry = worker_entry or worker_main
        #: Handed to every worker as its ``options`` mapping.  The pipeline
        #: passes ``{"warmup": ...}``; tests pass gates and delays, which is how
        #: a completion order is chosen rather than waited for.
        self._worker_options: dict[str, Any] = dict(worker_options or {})
        self._handles: list[_WorkerHandle] = []
        self._by_id: dict[int, _WorkerHandle] = {}
        self._result_queue: Any | None = None
        # The pending slot.  ``latest_only`` keeps exactly the newest task here;
        # otherwise it is a short FIFO bounded by ``pending_capacity``.  Either
        # way ``deque(maxlen=...)`` does the eviction, so there is no unbounded
        # backlog to grow.
        self._pending: Any = None
        self._task_seq = 0
        self._started = False
        self._stopping = False
        self._stop_report: dict[str, Any] = {}
        self._last_exit_check = 0.0

    # ------------------------------------------------------------------ #
    # Configuration surface
    # ------------------------------------------------------------------ #

    @property
    def gpu_devices(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.config.gpu_devices)

    @property
    def latest_only(self) -> bool:
        return bool(self.config.latest_only)

    @property
    def pending_capacity(self) -> int:
        return max(1, int(self.config.pending_capacity))

    def next_task_seq(self) -> int:
        """Allocate the next strictly increasing task sequence number.

        The scheduler owns the counter so that allocation, dispatch, worker
        echo and result validation all read the same sequence - there is no
        second place a number could come from.
        """
        self._task_seq += 1
        return self._task_seq

    @property
    def last_task_seq(self) -> int:
        return self._task_seq

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> GpuInventory:
        """Spawn one process per device and return the GPU inventory.

        Non-blocking on purpose: the checkpoint takes a minute or two to load
        and blocking here would stall the ASGI startup handler, keeping
        ``/health`` from answering at all during exactly the window an operator
        polls it.  Readiness is reported by ``ready_worker_count`` instead, and
        ``/ready`` gates on it.
        """
        if self._started:
            raise RuntimeError("GpuScheduler.start() called twice")
        import collections

        inventory = probe_gpu_inventory()
        problem = validate_pool_startup(self.config, inventory)
        if problem is not None:
            raise ValueError(problem)
        delta = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if delta:
            logger.info(
                "inference pool overrides the inherited CUDA_VISIBLE_DEVICES=%r "
                "with one device per worker: %s",
                delta,
                list(self.gpu_devices),
            )
        if inventory.available:
            logger.info(
                "gpu inventory (%s): %s",
                inventory.source,
                ", ".join(
                    f"{d.index}:{d.name}"
                    + (
                        f" free={d.free_bytes / (1 << 30):.1f}GiB"
                        if d.free_bytes is not None
                        else ""
                    )
                    for d in inventory.devices
                )
                or "(none)",
            )
        else:
            logger.warning(
                "gpu inventory unavailable (%s); device bounds are not checked",
                inventory.error,
            )

        self._result_queue = self._context.Queue()
        # ``latest_only`` is expressed as a one-slot deque rather than a special
        # case in the code: "keep the newest" and "keep a one-deep FIFO" then
        # share every line below, and the config validator guarantees the two
        # cannot disagree.
        self._pending = collections.deque(
            maxlen=1 if self.latest_only else self.pending_capacity
        )
        worker_id = 0
        for device in self.gpu_devices:
            task_queue = self._context.Queue()
            process = self._context.Process(
                target=self._worker_entry,
                args=(
                    worker_id,
                    device,
                    self.settings,
                    task_queue,
                    self._result_queue,
                    self._worker_options,
                ),
                name=f"alpamayo-gpu-worker-{worker_id}-dev{device}",
                daemon=True,
            )
            process.start()
            handle = _WorkerHandle(
                worker_id=worker_id,
                device=int(device),
                process=process,
                task_queue=task_queue,
                kind="local",
                spawned_at=self._clock(),
            )
            self._handles.append(handle)
            self._by_id[worker_id] = handle
            worker_id += 1
        health_gap_s = float(self.config.network_health_interval_ms) / 1000.0
        for endpoint in self.config.network_workers:
            handle = _WorkerHandle(
                worker_id=worker_id,
                device=-1,
                process=None,
                task_queue=None,
                kind="network",
                endpoint=str(endpoint),
                state=WORKER_LOADING,
                # The first tick should ask /health immediately.  A zero
                # timestamp would wait out the interval when the clock itself
                # is near zero.
                spawned_at=self._clock(),
                last_health_at=self._clock() - health_gap_s,
            )
            self._handles.append(handle)
            self._by_id[worker_id] = handle
            worker_id += 1
        self._started = True
        logger.info(
            "inference pool started: workers=%d devices=%s network=%s "
            "mode=multi_process latest_only=%s pending_capacity=%d poll_ms=%.1f "
            "dispatch_interval_ms=%.1f",
            len(self._handles),
            list(self.gpu_devices),
            list(self.config.network_workers),
            self.latest_only,
            self.pending_capacity,
            float(self.config.scheduler_poll_interval_ms),
            float(self.config.dispatch_interval_ms),
        )
        return inventory

    def stop(self, *, drain_s: float = 5.0, terminate_s: float = 3.0) -> dict[str, Any]:
        """Stop every worker within a bounded budget.

        Four steps, and the bound is the point: signal, drain, terminate, kill.
        A worker part-way through a 10B inference will not see the sentinel
        until it finishes, so ``drain_s`` is a real deadline and not a
        formality - and a shutdown that cannot be bounded is a shutdown that
        leaves a GPU allocated for the next process to trip over.
        """
        if not self._started:
            return {"workers": 0, "stopped": [], "terminated": [], "killed": []}
        self._stopping = True
        started_at = self._clock()
        stopped: list[int] = []
        terminated: list[int] = []
        killed: list[int] = []

        local = [
            h for h in self._handles if h.kind != "network" and h.process is not None
        ]
        network = [
            h for h in self._handles if h.kind == "network" or h.process is None
        ]

        # -- signal ---------------------------------------------------------- #
        for handle in local:
            try:
                handle.task_queue.put_nowait(None)
            except Exception:  # noqa: BLE001 - a dead worker needs no sentinel
                pass

        # -- drain ----------------------------------------------------------- #
        # Network slots have no process.  They are stopped here without
        # calling ``is_alive``; an in-flight HTTP thread is a daemon and is
        # abandoned with the slot.
        deadline = started_at + max(0.0, drain_s)
        for handle in local:
            remaining = max(0.0, deadline - self._clock())
            try:
                handle.process.join(remaining)
            except Exception:  # noqa: BLE001
                pass
        alive = [h for h in local if h.process.is_alive()]
        done = [h for h in local if not h.process.is_alive()]
        stopped = [h.worker_id for h in done]
        stopped.extend(h.worker_id for h in network)

        # -- terminate ------------------------------------------------------- #
        for handle in alive:
            try:
                handle.process.terminate()
            except Exception:  # noqa: BLE001
                pass
        deadline = self._clock() + max(0.0, terminate_s)
        for handle in alive:
            remaining = max(0.0, deadline - self._clock())
            try:
                handle.process.join(remaining)
            except Exception:  # noqa: BLE001
                pass
        still = [h for h in alive if h.process.is_alive()]
        terminated = [h.worker_id for h in alive if not h.process.is_alive()]

        # -- kill ------------------------------------------------------------ #
        for handle in still:
            try:
                handle.process.kill()
                handle.process.join(1.0)
                killed.append(handle.worker_id)
            except Exception:  # noqa: BLE001
                logger.error("worker %d could not be killed", handle.worker_id)

        for handle in self._handles:
            handle.state = WORKER_STOPPED
            if handle.task_queue is None:
                continue
            try:
                handle.task_queue.close()
                handle.task_queue.join_thread()
            except Exception:  # noqa: BLE001
                pass
        if self._pending is not None:
            for task in self._pending:
                self.result_manager.discard_task(task)
            self._pending.clear()
        if self._result_queue is not None:
            try:
                self._result_queue.close()
                self._result_queue.join_thread()
            except Exception:  # noqa: BLE001
                pass

        self._stop_report = {
            "workers": len(self._handles),
            "stopped": stopped,
            "terminated": terminated,
            "killed": killed,
            "elapsed_ms": round((self._clock() - started_at) * 1000.0, 3),
            "drain_s": float(drain_s),
            "terminate_s": float(terminate_s),
        }
        if terminated or killed:
            logger.warning(
                "inference pool stop was not graceful: stopped=%s terminated=%s "
                "killed=%s elapsed_ms=%.0f",
                stopped,
                terminated,
                killed,
                self._stop_report["elapsed_ms"],
            )
        else:
            logger.info(
                "inference pool stopped: workers=%d elapsed_ms=%.0f",
                len(self._handles),
                self._stop_report["elapsed_ms"],
            )
        return dict(self._stop_report)

    # ------------------------------------------------------------------ #
    # Submit side
    # ------------------------------------------------------------------ #

    def submit(self, task: InferenceTask) -> str:
        """Hand a task to a free worker, or park it as the pending one.

        Returns one of ``SUBMIT_*``.  ``on_accepted`` is *not* called from here:
        a submitted task has not run yet.
        """
        self.result_manager.note_task(task)
        if self._may_dispatch():
            for handle in self._handles:
                if handle.idle:
                    self._dispatch(handle, task)
                    return SUBMIT_DISPATCHED
        displaced = self._offer_pending(task)
        if displaced is None:
            return SUBMIT_QUEUED
        self.metrics.drops.bump(DROP_PENDING_REPLACED)
        self.result_manager.discard_task(displaced)
        logger.debug(
            "pending task displaced: displaced_task_seq=%d new_task_seq=%d "
            "workers_busy=%d latest_only=%s",
            displaced.task_seq,
            task.task_seq,
            sum(1 for h in self._handles if h.task is not None),
            self.latest_only,
        )
        # Under ``latest_only`` the newcomer took the slot; under a short FIFO
        # it is the newcomer that was refused.  Both mean one task will never
        # be inferred, which is what the counter records.
        return SUBMIT_REPLACED if self.latest_only else SUBMIT_DROPPED

    def _offer_pending(self, task: InferenceTask) -> InferenceTask | None:
        """Put ``task`` in the pending slot; return the task left behind.

        With ``latest_only`` the slot holds one task and the newcomer always
        wins.  Without it the slot is a short FIFO and the newcomer is dropped
        when the deque is full - the opposite trade, chosen deliberately by
        whoever turned the flag off.
        """
        pending = self._pending
        if pending is None:  # pragma: no cover - start() always builds the deque
            return None
        if self.latest_only:
            displaced = pending[0] if pending else None
            # ``maxlen == 1``, so the append evicts the old task for us.
            pending.append(task)
            return displaced
        if len(pending) >= pending.maxlen:
            return task  # the newcomer is the one that gets dropped
        pending.append(task)
        return None

    def _may_dispatch(self) -> bool:
        """Whether the shared dispatch interval currently allows a new task.

        ``submit`` and ``_fill_idle`` both ask this.  Zero means the interval
        is off.  A task that is refused here stays in the pending slot.
        """
        interval_ms = float(self.config.dispatch_interval_ms)
        if interval_ms <= 0:
            return True
        last = self._last_dispatch_at
        if last is None:
            return True
        return (self._clock() - last) * 1000.0 >= interval_ms

    def _remember_dispatch(self) -> None:
        self._last_dispatch_at = self._clock()

    def _dispatch(self, handle: _WorkerHandle, task: InferenceTask) -> None:
        """Send one task to one idle worker, and record that it is running."""
        now_ns = int(self._clock_ns())
        self.metrics.queue_wait_ms.add(task.age_ms(now_ns))
        self.result_manager.note_dispatch(task, handle.worker_id)
        handle.task = task
        handle.state = WORKER_BUSY
        if handle.kind == "network":
            self._dispatch_network(handle, task)
            return
        try:
            handle.task_queue.put_nowait(task)
        except Exception as exc:  # noqa: BLE001 - a broken pipe means a dead worker
            handle.state = WORKER_FAILED
            handle.error = f"dispatch failed: {type(exc).__name__}: {exc}"
            handle.error_kind = ERROR_WORKER_EXCEPTION
            handle.task = None
            self.result_manager.discard_worker(handle.worker_id)
            self.metrics.faults.bump("worker_dispatch_failed")
            logger.error(
                "worker %d could not be given task_seq=%d: %s",
                handle.worker_id,
                task.task_seq,
                exc,
            )
            return
        self._remember_dispatch()

    def _dispatch_network(self, handle: _WorkerHandle, task: InferenceTask) -> None:
        """Start one HTTP inference.  The pipeline thread does not wait on it."""
        try:
            body = build_infer_body(task)
        except Exception as exc:  # noqa: BLE001 - a bad payload must not send tensors
            self._mark_unreachable(handle, f"payload rejected: {exc}")
            return
        handle.deadline = self._clock() + float(self.config.network_infer_timeout_s)
        worker_id = int(handle.worker_id)
        device = int(handle.device)
        task_seq = int(task.task_seq)
        endpoint = handle.endpoint
        connect_timeout_s = float(self.config.network_connect_timeout_s)
        infer_timeout_s = float(self.config.network_infer_timeout_s)
        transport = self._transport
        result_queue = self._result_queue

        def run() -> None:
            try:
                response = transport.infer(
                    endpoint,
                    body,
                    connect_timeout_s=connect_timeout_s,
                    read_timeout_s=infer_timeout_s,
                )
                message: Any = to_inference_result(
                    response,
                    worker_id=worker_id,
                    device=device,
                    task_seq=task_seq,
                )
            except Exception as exc:  # noqa: BLE001 - timeout and disconnect are results
                message = NetworkFault(
                    worker_id=worker_id,
                    task_seq=task_seq,
                    error=f"{type(exc).__name__}: {exc}",
                )
            try:
                result_queue.put_nowait(message)
            except Exception:  # noqa: BLE001 - shutdown may have closed the queue
                pass

        threading.Thread(
            target=run,
            name=f"alpamayo-net-{worker_id}-task{task_seq}",
            daemon=True,
        ).start()
        self._remember_dispatch()

    def _mark_unreachable(self, handle: _WorkerHandle, error: str) -> None:
        """Drop the slot's current task and keep it out of the schedule.

        The state stays ``unreachable`` until a later health answer is
        explicitly ready.  This method must not set ``ready``.
        """
        task = handle.task
        handle.state = WORKER_UNREACHABLE
        handle.task = None
        handle.deadline = None
        handle.error = error
        handle.error_kind = ERROR_WORKER_EXCEPTION
        if task is not None:
            self.result_manager.discard_worker(handle.worker_id)
        self.metrics.faults.bump("network_worker_unreachable")
        logger.error(
            "network worker %d (%s) unreachable: %s",
            handle.worker_id,
            handle.endpoint,
            error,
        )

    # ------------------------------------------------------------------ #
    # Result side
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
        """Drain whatever the workers produced, then fill every idle worker.

        Called from the pipeline loop at ``scheduler_poll_interval_ms``.  The
        order matters: draining first means a worker that has just answered is
        already idle by the time the fill runs, so it takes the newest pending
        task in the same tick rather than one poll later.
        """
        if not self._started or self._stopping:
            return
        self._drain()
        self._check_workers()
        self._fill_idle()

    def _drain(self) -> None:
        result_queue = self._result_queue
        if result_queue is None:
            return
        while True:
            try:
                message = result_queue.get_nowait()
            except queue_module.Empty:
                return
            except (OSError, EOFError, ValueError):
                return
            try:
                self._handle_message(message)
            except Exception:  # noqa: BLE001 - one bad message must not stop the drain
                self.metrics.faults.bump("message_handling_error")
                logger.exception("failed to handle a pool message: %r", message)

    def _handle_message(self, message: Any) -> None:
        if isinstance(message, WorkerStatus):
            self._handle_status(message)
        elif isinstance(message, InferenceResult):
            self._handle_result(message)
        elif isinstance(message, NetworkFault):
            self._handle_network_fault(message)
        elif isinstance(message, NetworkHealth):
            self._handle_network_health(message)
        else:  # pragma: no cover - protocol guard
            self.metrics.faults.bump("unknown_message")
            logger.error("unknown pool message: %r", type(message))

    def _handle_status(self, status: WorkerStatus) -> None:
        handle = self._by_id.get(int(status.worker_id))
        if handle is None:  # pragma: no cover - a worker we did not spawn
            return
        if handle.kind == "network":
            # A network slot does not publish process status.  Ready comes
            # only from an explicit health answer.
            return
        handle.state = status.state
        if status.error is not None:
            handle.error = status.error
            handle.error_kind = status.error_kind
        if status.memory.allocated_bytes or status.memory.reserved_bytes:
            handle.memory = status.memory
        if status.infer_count:
            handle.infer_count = status.infer_count
        if status.last_infer_ms is not None:
            handle.last_infer_ms = status.last_infer_ms
        if status.state == WORKER_FAILED:
            self.metrics.faults.bump("worker_failed")
            logger.error(
                "gpu worker %d (device %d) failed: %s",
                handle.worker_id,
                handle.device,
                status.error,
            )
            if handle.task is not None:
                self.result_manager.discard_worker(handle.worker_id)
                handle.task = None
        elif status.state == WORKER_STOPPED:
            if handle.task is not None:
                self.result_manager.discard_worker(handle.worker_id)
                handle.task = None

    def _handle_result(self, result: InferenceResult) -> None:
        handle = self._by_id.get(int(result.worker_id))
        task = handle.task if handle is not None else None
        if handle is not None and handle.kind == "network":
            current = handle.task
            if current is None or int(current.task_seq) != int(result.task_seq):
                # A late answer for a task this slot has already dropped, or
                # for a task that was replaced after a timeout.  It must not
                # clear the task the slot is running now, and it must not
                # turn ``unreachable`` back into ``ready``.
                self.metrics.note_raw_inference(infer_ms=result.infer_ms)
                self.result_manager.validate(result)
                return
        if handle is not None:
            handle.task = None
            handle.deadline = None
            if handle.state != WORKER_FAILED:
                handle.state = WORKER_READY
            handle.infer_count += 1
            handle.last_infer_ms = result.infer_ms
            handle.infer_stats.add(result.infer_ms)
            handle.memory = result.memory
        # Counted before validation: this is the model's own rate, and a result
        # that is later dropped for being stale still cost a real inference.
        self.metrics.note_raw_inference(infer_ms=result.infer_ms)

        started = self._clock()
        verdict = self.result_manager.validate(result)
        self.metrics.validate_ms.add((self._clock() - started) * 1000.0)

        if not verdict.accepted:
            logger.warning(
                "result-dropped: reason=%s task_seq=%d worker=%d device=%d %s",
                verdict.reason,
                result.task_seq,
                result.worker_id,
                result.device,
                verdict.detail,
            )
            return
        end_to_end_ms = (
            None if task is None else task.age_ms(int(self._clock_ns()))
        )
        self.metrics.note_accepted(end_to_end_ms=end_to_end_ms)
        if task is not None:
            self.on_accepted(result, task)
        else:  # pragma: no cover - validate() rejects a result with no task
            self.metrics.faults.bump("accepted_without_task")

    def _check_workers(self) -> None:
        """Notice a worker that died and a worker that never finished loading."""
        now = self._clock()
        self._check_network(now)
        # Polling ``is_alive`` is cheap but not free; twice a second is enough
        # to notice a crash well inside the result-age budget.  Network slots
        # are not processes and are handled above, before this throttle.
        if now - self._last_exit_check < 0.5:
            return
        self._last_exit_check = now
        timeout_s = float(self.config.worker_start_timeout_s)
        for handle in self._handles:
            if handle.state in (WORKER_FAILED, WORKER_STOPPED):
                continue
            process = handle.process
            if handle.kind == "network" or process is None:
                continue
            if not process.is_alive():
                self.metrics.faults.bump("worker_exited")
                handle.state = WORKER_FAILED
                handle.error = (
                    f"worker process exited with code "
                    f"{process.exitcode}"
                )
                handle.error_kind = ERROR_WORKER_EXCEPTION
                if handle.task is not None:
                    self.result_manager.discard_worker(handle.worker_id)
                    handle.task = None
                logger.error(
                    "gpu worker %d (device %d) exited unexpectedly: %s",
                    handle.worker_id,
                    handle.device,
                    handle.error,
                )
            elif (
                handle.state == WORKER_LOADING
                and now - handle.spawned_at > timeout_s
            ):
                self.metrics.faults.bump("worker_load_timeout")
                handle.state = WORKER_FAILED
                handle.error = (
                    f"still loading after {timeout_s:.0f}s "
                    "(inference_pool.worker_start_timeout_s)"
                )
                handle.error_kind = ERROR_WORKER_EXCEPTION
                logger.error(
                    "gpu worker %d (device %d) did not finish loading within "
                    "%.0fs; marking it failed",
                    handle.worker_id,
                    handle.device,
                    timeout_s,
                )

    def _check_network(self, now: float) -> None:
        """Time out busy network slots, then ask the others if they are ready.

        A timed-out slot becomes ``unreachable`` in this call.  Health is only
        *started* here; the answer arrives later on the result queue, so this
        tick cannot both time a slot out and dispatch to it again.
        """
        start_timeout_s = float(self.config.worker_start_timeout_s)
        for handle in self._handles:
            if handle.kind != "network":
                continue
            if handle.state in (WORKER_FAILED, WORKER_STOPPED):
                continue
            if (
                handle.state == WORKER_BUSY
                and handle.deadline is not None
                and now >= handle.deadline
            ):
                self._mark_unreachable(handle, "inference timed out")
            elif (
                handle.state == WORKER_LOADING
                and now - handle.spawned_at > start_timeout_s
            ):
                self._mark_unreachable(
                    handle,
                    "health check did not return ready before "
                    "worker_start_timeout_s",
                )
        self._poll_network_health(now)

    def _poll_network_health(self, now: float) -> None:
        interval_s = float(self.config.network_health_interval_ms) / 1000.0
        for handle in self._handles:
            if handle.kind != "network":
                continue
            if handle.state not in (WORKER_LOADING, WORKER_UNREACHABLE):
                continue
            if handle.health_inflight:
                continue
            if now - handle.last_health_at < interval_s:
                continue
            handle.health_inflight = True
            handle.last_health_at = now
            self._start_health(handle)

    def _start_health(self, handle: _WorkerHandle) -> None:
        worker_id = int(handle.worker_id)
        endpoint = handle.endpoint
        connect_timeout_s = float(self.config.network_connect_timeout_s)
        transport = self._transport
        result_queue = self._result_queue

        def run() -> None:
            try:
                body = transport.health(
                    endpoint,
                    connect_timeout_s=connect_timeout_s,
                    read_timeout_s=connect_timeout_s,
                )
                notice = NetworkHealth(
                    worker_id=worker_id,
                    ready=health_is_ready(body),
                    state=str(body.get("state") or ""),
                    error=None if body.get("error") is None else str(body.get("error")),
                )
            except Exception as exc:  # noqa: BLE001 - a missed poll is not ready
                notice = NetworkHealth(
                    worker_id=worker_id,
                    ready=False,
                    state=WORKER_UNREACHABLE,
                    error=f"{type(exc).__name__}: {exc}",
                )
            try:
                result_queue.put_nowait(notice)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(
            target=run,
            name=f"alpamayo-health-{worker_id}",
            daemon=True,
        ).start()

    def _handle_network_fault(self, fault: NetworkFault) -> None:
        handle = self._by_id.get(int(fault.worker_id))
        if handle is None or handle.kind != "network":
            return
        task = handle.task
        if (
            handle.state != WORKER_BUSY
            or task is None
            or int(task.task_seq) != int(fault.task_seq)
        ):
            return
        self._mark_unreachable(handle, fault.error)

    def _handle_network_health(self, notice: NetworkHealth) -> None:
        handle = self._by_id.get(int(notice.worker_id))
        if handle is None or handle.kind != "network":
            return
        handle.health_inflight = False
        if (
            notice.ready
            and handle.task is None
            and handle.state in (WORKER_LOADING, WORKER_UNREACHABLE)
        ):
            handle.state = WORKER_READY
            handle.error = None
            handle.error_kind = ""
            logger.info(
                "network worker %d (%s) health returned ready",
                handle.worker_id,
                handle.endpoint,
            )

    def _fill_idle(self) -> None:
        pending = self._pending
        if pending is None or not pending:
            return
        for handle in self._handles:
            if not pending:
                break
            if not handle.idle:
                continue
            if not self._may_dispatch():
                return
            task = pending.popleft()
            self._dispatch(handle, task)

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #

    @property
    def ready_worker_count(self) -> int:
        return sum(
            1
            for h in self._handles
            if h.state in (WORKER_READY, WORKER_BUSY)
        )

    @property
    def failed_worker_count(self) -> int:
        return sum(1 for h in self._handles if h.state == WORKER_FAILED)

    @property
    def busy_worker_count(self) -> int:
        return sum(1 for h in self._handles if h.task is not None)

    @property
    def loading_worker_count(self) -> int:
        return sum(1 for h in self._handles if h.state == WORKER_LOADING)

    @property
    def unreachable_worker_count(self) -> int:
        return sum(1 for h in self._handles if h.state == WORKER_UNREACHABLE)

    @property
    def pending_task_seq(self) -> int | None:
        pending = self._pending
        if not pending:
            return None
        return pending[-1].task_seq if self.latest_only else pending[0].task_seq

    @property
    def pending_count(self) -> int:
        return 0 if self._pending is None else len(self._pending)

    @property
    def started(self) -> bool:
        return self._started

    def worker_views(self) -> tuple[WorkerView, ...]:
        return tuple(
            WorkerView(
                worker_id=h.worker_id,
                device=h.device,
                state=h.state,
                task_seq=None if h.task is None else h.task.task_seq,
                infer_count=h.infer_count,
                last_infer_ms=h.last_infer_ms,
                error=h.error,
                memory=h.memory,
                avg_infer_ms=h.infer_stats.average_ms,
                p50_infer_ms=h.infer_stats.percentile_ms(0.50),
                p95_infer_ms=h.infer_stats.percentile_ms(0.95),
                infer_samples=h.infer_stats.count,
                endpoint=h.endpoint,
            )
            for h in self._handles
        )

    def stop_report(self) -> dict[str, Any]:
        return dict(self._stop_report)


__all__ = [
    "DROP_PENDING_REPLACED",
    "GpuInfo",
    "GpuInventory",
    "GpuScheduler",
    "SUBMIT_DISPATCHED",
    "SUBMIT_DROPPED",
    "SUBMIT_QUEUED",
    "SUBMIT_REPLACED",
    "WORKER_DEVICE",
    "WorkerView",
    "probe_gpu_inventory",
    "validate_pool_startup",
    "worker_main",
]
