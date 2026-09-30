"""One coherent, immutable view of the running pipeline.

Why this module exists
----------------------
``/health`` and ``/status`` used to reach into the live ``PipelineRunner``
independently, one attribute at a time, while the inference loop ran on its own
thread in between those reads.  The two handlers could therefore disagree, and
neither could tell whether a value had been *measured* or merely *remembered*:
``inference_count`` was incremented without a lock and read from a threadpool
thread, and ``ok`` was hardcoded ``True`` so nothing distinguished "the loop is
serving" from "the loop is wedged inside ``model.infer()``" from "the model
never loaded".

This module is the fix.  The pipeline thread publishes a fresh **immutable**
:class:`RuntimeSnapshot` once per loop iteration; the HTTP handlers read the
last published object.  Nothing here imports the application - it is a leaf, so
``pipeline.py`` may depend on it and it can be tested with no runner, no camera
and no GPU.

What deliberately stays a live read
-----------------------------------
Rate-shaped fields are NOT in the snapshot, on purpose.  ``tools/check_chain.sh``
samples ``/health`` twice, two seconds apart, and derives each decoder's fps from
the delta of ``camera_stats.readers[].frames_total``; the same tool derives the
pose rate from the delta of ``/status``'s ``chassis_packets``.  Publishing those
once per inference cycle would freeze them whenever the pipeline is stuck in a
multi-second inference - reporting "0 fps on all four cameras" at exactly the
moment an operator is trying to tell a dead decoder from a dead pipeline.  So
the decoders' own diagnostics and the vehicle's rate counters stay live reads
off the objects that own them, and the snapshot carries the *pipeline's* state.

Concurrency
-----------
Single writer, no lock.  ``PipelineRunner`` publishes with
``self.snapshot = health.advance(self.snapshot, **updates)``; a reader takes one
reference into a local and uses that object.  A reference store is atomic in
CPython and every field here is a scalar, string, tuple or frozen dataclass, so
a published object is deeply immutable - even a torn read could not yield a
half-updated value.  A lock would not help: it only buys consistency if the
writer mutates one shared object under it, and holding it across a cycle would
block ``/health`` for a whole inference.  (A free-threaded build would need to
revisit the reference-store assumption; nothing here relies on more than that.)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from typing import Any

SNAPSHOT_VERSION = 1

# Phases of the process lifecycle, as the operator sees them.
PHASE_CREATED = "created"
PHASE_RUNNING = "running"
PHASE_STOPPING = "stopping"
PHASE_STOPPED = "stopped"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ``pipeline.py``'s "no camera window for N s" watchdog and readiness share this
# one number so they cannot drift: the loop's field-tuned threshold is the right
# readiness bound, because it is the point at which a healthy-but-slow pipeline
# was already considered stalled in the field.
#
# Do NOT use ``camera_stale_ms`` (1000 ms) here.  That is a per-frame gate
# budget, and ``_last_window_ok_at`` only advances once per loop iteration, so a
# healthy pipeline with a multi-second inference would flap to "not ready".
WINDOW_STALE_S = _env_float("ALPAMAYO_READY_WINDOW_STALE_S", 10.0)

# Matches ``start_realtime.ps1``'s own startup gate: 12 attempts x 5 s, each
# doing ``curl -fsS --max-time 3 /health``.  A process that is still loading the
# checkpoint inside that budget is "starting", not "broken".
STARTUP_GRACE_S = _env_float("ALPAMAYO_READY_STARTUP_GRACE_S", 60.0)

# Shutdown budget, split in two so the reap can never be starved by the drain.
# ``run_backend_real.sh`` signals uvicorn and starts the replacement one second
# later.  The reap joins the CIMG reader so the next start can bind its port.
DRAIN_S = _env_float("ALPAMAYO_SHUTDOWN_DRAIN_S", 6.0)
REAP_S = _env_float("ALPAMAYO_SHUTDOWN_REAP_S", 3.0)


@dataclass(frozen=True)
class TimeSyncView:
    """The timestamp clocks, frozen at publish time.

    A field of the snapshot rather than a live read because it is constant after
    construction, so freezing it cannot go stale - and because ``/health``
    should not have to reach into the runner for it.
    """

    camera_clock: str
    config_error: str | None = None


@dataclass(frozen=True)
class InferenceWorkerView:
    """One GPU worker, frozen at publish time.

    A plain copy rather than ``inference.gpu_pool.WorkerView``: ``health`` is
    imported by the HTTP handlers and by tests that never load torch, and
    reaching into the inference package for a type would drag the whole stack
    into every ``/health`` read.  The fields are the subset the endpoints
    report, which is also the subset that survives being written to JSON.
    """

    worker_id: int
    device: int
    state: str
    task_seq: int | None = None
    infer_count: int = 0
    last_infer_ms: float | None = None
    avg_infer_ms: float | None = None
    p50_infer_ms: float | None = None
    p95_infer_ms: float | None = None
    #: How many results the percentiles are computed over.  Reported next to
    #: them on purpose: a P95 over four samples and a P95 over four hundred are
    #: the same number in the same units and mean very different things.
    infer_samples: int = 0
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    error: str | None = None
    endpoint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "device": self.device,
            "state": self.state,
            "task_seq": self.task_seq,
            "infer_count": self.infer_count,
            "last_infer_ms": self.last_infer_ms,
            "avg_infer_ms": self.avg_infer_ms,
            "p50_infer_ms": self.p50_infer_ms,
            "p95_infer_ms": self.p95_infer_ms,
            "infer_samples": self.infer_samples,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "error": self.error,
            "endpoint": self.endpoint,
        }


@dataclass(frozen=True)
class InferencePoolView:
    """The inference pool as of one publish, or ``None`` when it is not in use.

    ``None`` is the inline mode, and it is not the same statement as a pool with
    zero ready workers: inline has no pool at all and every readiness check about
    it is vacuously satisfied, while a pool that exists and is still loading
    must fail ``/ready``.  The two are distinguished by this being ``None``
    rather than by a count.
    """

    mode: str
    state: str
    devices: tuple[int, ...]
    worker_count: int
    ready_workers: int
    busy_workers: int
    loading_workers: int
    failed_workers: int
    minimum_ready_workers: int
    latest_only: bool
    pending_capacity: int
    pending_task_seq: int | None
    pending_count: int
    unreachable_workers: int = 0
    workers: tuple[InferenceWorkerView, ...] = ()
    error: str | None = None
    #: The pool's last shutdown report, ``{}`` until it has stopped.
    #:
    #: A ``dict`` inside a frozen dataclass is the one mutable thing here, and
    #: it is safe for a reason worth stating: ``GpuScheduler.stop`` returns a
    #: fresh dict, the runner assigns it once and never touches it again, and
    #: nothing else holds a reference.  Turning it into a tuple of pairs would
    #: buy immutability and cost every reader a conversion to get JSON out.
    stop_report: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "state": self.state,
            "devices": list(self.devices),
            "worker_count": self.worker_count,
            "ready_workers": self.ready_workers,
            "busy_workers": self.busy_workers,
            "loading_workers": self.loading_workers,
            "failed_workers": self.failed_workers,
            "unreachable_workers": self.unreachable_workers,
            "minimum_ready_workers": self.minimum_ready_workers,
            "latest_only": self.latest_only,
            "pending_capacity": self.pending_capacity,
            "pending_task_seq": self.pending_task_seq,
            "pending_count": self.pending_count,
            "workers": [w.as_dict() for w in self.workers],
            "error": self.error,
            "stop_report": dict(self.stop_report),
        }


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Everything the pipeline thread knows about itself, as of one instant.

    Instants are stored, never ages: an age is derived at read time, so a
    snapshot that stops being republished visibly ages rather than silently
    reporting a stale "0 ms".
    """

    # -- bookkeeping, set by :func:`advance` --------------------------------
    version: int = SNAPSHOT_VERSION
    sequence: int = 0
    published_at_monotonic: float = 0.0
    phase: str = PHASE_CREATED
    started_at_monotonic: float | None = None

    # -- per-cycle progress -------------------------------------------------
    cycle_id: int = 0
    inference_count: int = 0
    model_loaded: bool = False
    last_error: str | None = None
    last_rejection_reason: str | None = None
    last_t0_us: int | None = None
    last_control_flag: int | None = None
    last_sent_control_flag: int | None = None
    last_num_waypoints: int | None = None
    last_cot: str | None = None
    # ``TakeoverDecision`` is already a frozen dataclass, so it is embedded by
    # reference: one object, no copy, and no key list to keep in sync.
    last_takeover_decision: Any | None = None

    # -- instants (monotonic seconds, except the pose which is microseconds) -
    last_window_ok_at: float | None = None
    last_inference_at: float | None = None
    # ``history.latest().arrival_us`` - the same monotonic-microsecond clock
    # ``InputBuilder`` already gates on, so no new time domain is introduced.
    last_pose_arrival_us: int | None = None
    #
    # There is deliberately no ``last_send_at``.  ``WaypointSender.last_send_at``
    # is a WALL clock (it feeds the package's ``send_timestamp``), and mixing it
    # with these monotonic instants would invite an age computation across two
    # time domains.  The send is already visible as ``last_sent_control_flag``
    # and in the package; nothing needs a monotonic send instant.

    # -- component state as of this publish ---------------------------------
    unhealthy_readers: tuple[str, ...] = ()
    route_receiver_listening: bool | None = None
    route_receiver_error: str | None = None

    # -- inference pool (multi-process mode only) ---------------------------
    # ``None`` in inline mode, and in a process with no runner at all.
    inference_pool: InferencePoolView | None = None
    #: The three frequencies, the drop and fault counters and the latency
    #: summaries, as ``InferenceMetrics.snapshot()`` produced them at publish
    #: time.  Published rather than read live because the pipeline thread is the
    #: only writer: a handler that called ``metrics.snapshot()`` itself would be
    #: reading counters as the loop bumps them, which is how a "queue wait" gets
    #: sampled from the middle of a task instead of at its end.  The dict is a
    #: fresh structure per publish and nothing mutates it afterwards.
    inference_metrics: dict[str, Any] | None = None

    # -- static values, repeated each publish so a handler reads one object --
    pose_mode: str = ""
    camera_stale_ms: float = 0.0
    pose_stale_ms: float = 0.0
    window_stale_s: float = WINDOW_STALE_S
    startup_grace_s: float = STARTUP_GRACE_S
    time_sync: TimeSyncView | None = None


# What a handler sees before any pipeline exists - and, deliberately, what it
# keeps seeing after one has been shut down, so the two are indistinguishable in
# the response shape.
EMPTY_SNAPSHOT = RuntimeSnapshot()


def advance(previous: RuntimeSnapshot, **updates: Any) -> RuntimeSnapshot:
    """Publish: a new snapshot with the sequence bumped and the clock stamped.

    An unknown keyword raises ``TypeError`` out of ``dataclasses.replace``,
    which is the point - a typo in a publisher becomes a loud failure instead of
    a field that silently never updates.
    """
    updates["version"] = SNAPSHOT_VERSION
    updates["sequence"] = previous.sequence + 1
    updates["published_at_monotonic"] = time.monotonic()
    return replace(previous, **updates)


def snapshot_age_ms(snapshot: RuntimeSnapshot, now_monotonic_ns: int) -> float | None:
    """How long ago this snapshot was published; ``None`` for the empty one.

    This is the field that makes staleness visible: a frozen ``sequence`` with a
    growing age is a pipeline that has stopped publishing.
    """
    if snapshot.sequence == 0 or snapshot.published_at_monotonic <= 0.0:
        return None
    return max(0.0, (now_monotonic_ns / 1e9 - snapshot.published_at_monotonic) * 1000.0)


def uptime_s(snapshot: RuntimeSnapshot, now_monotonic_ns: int) -> float | None:
    if snapshot.started_at_monotonic is None:
        return None
    return max(0.0, now_monotonic_ns / 1e9 - snapshot.started_at_monotonic)


def pose_age_ms(snapshot: RuntimeSnapshot, now_monotonic_ns: int) -> float | None:
    if snapshot.last_pose_arrival_us is None:
        return None
    return (now_monotonic_ns // 1000 - snapshot.last_pose_arrival_us) / 1000.0


def window_age_s(snapshot: RuntimeSnapshot, now_monotonic_ns: int) -> float | None:
    """Seconds since a camera window was last built; ``None`` if never.

    One definition, used by the readiness check and by ``/ready``'s diagnostics,
    so the number an operator reads cannot disagree with the verdict.
    """
    if snapshot.last_window_ok_at is None:
        return None
    return max(0.0, now_monotonic_ns / 1e9 - snapshot.last_window_ok_at)


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str
    failing: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: tuple[ReadinessCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "failing": list(self.failing),
            "warnings": list(self.warnings),
            "checks": [
                {"name": c.name, "ok": c.ok, "reason": c.reason, "detail": c.detail}
                for c in self.checks
            ],
        }


def evaluate_readiness(
    snapshot: RuntimeSnapshot,
    *,
    now_monotonic_ns: int,
    pipeline_alive: bool,
) -> Readiness:
    """Is this process healthy and receiving its inputs?

    Pure: the two live inputs (the clock, and an ``is_alive()`` call) are
    parameters, so this is testable with no runner and no threads.

    It answers "is this process working", NOT "is it taking over".  A missing
    planned route is a legitimate vehicle-side state that the decision layer
    already fails safe on, so it is a warning: turning it into a 503 would flap
    and would train operators to ignore the endpoint.  Likewise a persistent
    gate rejection flaps at the ``camera_pose_delta_ms`` boundary and belongs on
    a dashboard, not in a restart verdict.
    """
    checks: list[ReadinessCheck] = []

    def add(name: str, ok: bool, reason: str = "", detail: str = "") -> None:
        checks.append(
            ReadinessCheck(name=name, ok=bool(ok), reason="" if ok else reason, detail=detail)
        )

    up = uptime_s(snapshot, now_monotonic_ns)

    add(
        "not_stopping",
        snapshot.phase not in (PHASE_STOPPING, PHASE_STOPPED),
        "shutting_down",
        f"phase={snapshot.phase}",
    )
    add("pipeline_alive", pipeline_alive, "pipeline_not_running",
        f"is_alive={bool(pipeline_alive)}")
    add(
        "pipeline_started",
        snapshot.started_at_monotonic is not None,
        "starting",
        "never_started" if snapshot.started_at_monotonic is None else f"uptime_s={up:.1f}",
    )

    if snapshot.last_window_ok_at is not None:
        add("startup_complete", True, detail="window_seen")
    else:
        # This FAILS on both sides of the grace period, with two different
        # reasons.  "No camera window has ever been built" is not a ready state
        # at any age: a consumer of /ready would otherwise be told to proceed to
        # a pipeline that has never seen a frame.  What the 60 s budget buys is
        # not a pass but an honest explanation - inside it the expected cause is
        # the checkpoint load, past it the decoders are the suspect.  The budget
        # is the same one start_realtime.ps1 gates on, so a slow load cannot
        # outlive its own deploy script's patience.
        within_grace = up is not None and up <= snapshot.startup_grace_s
        add(
            "startup_complete",
            False,
            "starting" if within_grace else "no_camera_window",
            f"uptime_s={up:.1f} grace_s={snapshot.startup_grace_s:.0f}"
            if up is not None
            else "never_started",
        )

    window_age = window_age_s(snapshot, now_monotonic_ns)
    # "No window has EVER been built" is check 4's verdict, reported once, in
    # severity order.  Reporting it again here would make the first inference -
    # which runs for seconds before its iteration can publish - look like a
    # stale window rather than a starting pipeline.
    add(
        "camera_window_fresh",
        window_age is None or window_age <= snapshot.window_stale_s,
        "camera_window_stale",
        f"window_age_s={window_age:.1f} limit_s={snapshot.window_stale_s:.0f}"
        if window_age is not None
        else "no_window_yet",
    )

    pose_ms = pose_age_ms(snapshot, now_monotonic_ns)
    add(
        "pose_fresh",
        pose_ms is not None and pose_ms <= snapshot.pose_stale_ms,
        "pose_unavailable" if pose_ms is None else "pose_stale",
        f"pose_age_ms={pose_ms:.1f} limit_ms={snapshot.pose_stale_ms:.0f}"
        if pose_ms is not None
        else "no_pose_yet",
    )

    # The one thing ``/health`` fundamentally cannot express today: the loop is
    # alive and receiving, but the model never finished loading.
    #
    # Mode-aware, and it has to be: ``snapshot.model_loaded`` is the parent's own
    # ``AlpamayoRunner``, which in multi_process mode never loads the checkpoint
    # - the workers own the devices, and the parent has no model to load.  Left
    # as ``snapshot.model_loaded`` the check would fail forever on a perfectly
    # healthy pool.  In pool mode the same statement ("the model is loaded") is
    # made by the workers: enough of them are ready to answer.  The check keeps
    # its name, because ``/ready`` consumers and the deploy script key on it and
    # renaming it would break a currently-working gate; the detail string says
    # which of the two facts produced the verdict.
    pool = snapshot.inference_pool
    if pool is None:
        add("model_loaded", snapshot.model_loaded, "model_not_loaded",
            "runner=inline")
    else:
        # One check, not two.  A separate ``inference_workers_ready`` would carry
        # the same predicate as this one and therefore fail on the same instant
        # with the same cause - two lines in ``failing`` for one problem, in
        # exactly the verdict the module docstring says must not be noisy.  The
        # checks that DO earn a separate line are the ones that can disagree.
        #
        # ``minimum_ready_workers`` gates readiness through this check, which is
        # the whole point of the setting: with 4 devices and a minimum of 3, the
        # pool serves traffic while one worker is still loading, and with a
        # minimum of 4 it does not.
        add(
            "model_loaded",
            pool.ready_workers >= pool.minimum_ready_workers,
            "inference_workers_not_ready",
            f"mode={pool.mode} ready={pool.ready_workers}/{pool.worker_count} "
            f"minimum={pool.minimum_ready_workers} "
            f"loading={pool.loading_workers} failed={pool.failed_workers} "
            f"unreachable={pool.unreachable_workers}",
        )

    # Currently unreachable: ``TimestampManager.from_config()`` never populates
    # ``config_error``, so a broken config raises out of the runner's
    # constructor instead.  Kept because it costs nothing and fires the day
    # someone wires it up.
    ts = snapshot.time_sync
    add("time_sync_configured", ts is None or ts.config_error is None,
        "time_sync_config_error", "no_config_error" if ts is None else f"config_error={ts.config_error}")

    warnings: list[str] = []
    if pool is not None:
        # A warning, not a failure: a pool that has lost a worker but still meets
        # ``minimum_ready_workers`` is degraded and serving, and turning that into
        # a 503 would take a process that is driving the vehicle out of rotation
        # for a fault it was configured to survive.  When it stops being
        # survivable, ``model_loaded`` above fails and says so.
        if pool.failed_workers:
            warnings.append(f"inference_workers_failed:{pool.failed_workers}")
        if pool.unreachable_workers:
            warnings.append(
                f"inference_workers_unreachable:{pool.unreachable_workers}"
            )
        if pool.error:
            warnings.append(f"inference_pool_error:{pool.error}")
    if snapshot.route_receiver_listening is False:
        warnings.append("route_receiver_not_listening")
    if snapshot.route_receiver_error:
        warnings.append(f"route_receiver_error:{snapshot.route_receiver_error}")
    if snapshot.unhealthy_readers:
        warnings.append(f"readers_unhealthy:{len(snapshot.unhealthy_readers)}")
    if snapshot.last_error:
        warnings.append(f"pipeline_error:{snapshot.last_error}")
    if snapshot.last_rejection_reason:
        warnings.append(f"gate_rejected:{snapshot.last_rejection_reason}")

    failing = tuple(c.name for c in checks if not c.ok)
    first = next((c for c in checks if not c.ok), None)
    return Readiness(
        ready=not failing,
        reason="" if first is None else first.reason,
        failing=failing,
        warnings=tuple(warnings),
        checks=tuple(checks),
    )


# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ShutdownReport:
    """What shutdown actually stopped, and what it could not.

    A shutdown that silently fails to stop a decoder leaves the next instance
    colliding on UDP ports 31101-31104 ("bind failed: Address already in use",
    the 2026-09-10 outage).  ``run_backend_real.sh`` waits one second before
    starting the replacement, so a failure here is not recoverable by waiting -
    it has to be visible.  ``failed`` is the list an operator can act on, and
    ``app._shutdown`` logs it at ERROR when it is non-empty.

    ``components`` is every component the shutdown tried, so a component that
    was never reached is distinguishable from one that stopped cleanly: it
    appears in ``components`` and not in ``stopped``.
    """

    components: tuple[str, ...] = ()
    stopped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    drain_s: float = 0.0
    reap_s: float = 0.0
    budget_exceeded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "components": list(self.components),
            "stopped": list(self.stopped),
            "failed": list(self.failed),
            "elapsed_ms": self.elapsed_ms,
            "drain_s": self.drain_s,
            "reap_s": self.reap_s,
            "budget_exceeded": self.budget_exceeded,
        }


EMPTY_SHUTDOWN_REPORT = ShutdownReport()


# --------------------------------------------------------------------------- #
# Response builders
# --------------------------------------------------------------------------- #
#
# The JSON shape lives here, not in the handlers, so it can be tested without
# FastAPI, without cv2 and without a runner.  Key order is deliberate and
# preserved from the pre-Stage-3 handlers: ``tools/check_chain.sh`` reads these
# bodies by key, and an operator reading raw ``curl`` output should not see the
# fields move.

@dataclass(frozen=True)
class HealthSources:
    """The live half of ``/health`` - see the module docstring for why."""

    settings: Any
    camera_health: dict
    camera_stats: dict
    #: The direct-JPEG CIMG link's report.  Empty only when no runner exists.
    #: A live read for the same reason as the two fields above: connection
    #: state, peer, receive fps, frame age and queue depth are exactly the
    #: numbers an operator needs while the pipeline is stuck in a long
    #: inference, which is when a snapshot-time copy would be most wrong.
    camera_cimg: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StatusSources:
    """The live half of ``/status``.  Any of these may be ``None``."""

    settings: Any
    runner_present: bool
    timestamp_manager: Any | None = None
    pose_listener: Any | None = None
    # The newest pose comes from the shared history, NOT from the listener: the
    # listener is one of two writers (mock mode has no listener at all).
    history: Any | None = None
    planned_route_cache: Any | None = None
    route_tcp_receiver: Any | None = None
    #: See ``HealthSources.camera_cimg``.  Present on both bodies so an operator
    #: can reach the CIMG link's counters from whichever one they already poll.
    camera_cimg: dict = field(default_factory=dict)


def takeover_decision_view(decision: Any | None) -> dict[str, Any] | None:
    """The ``/status`` projection of a ``TakeoverDecision``.

    By reference rather than by copy, so this cannot drift from the dataclass.
    """
    if decision is None:
        return None
    return {
        "mean_lateral_error_m": decision.mean_lateral_error,
        "max_lateral_error_m": decision.max_lateral_error,
        "takeover_count": decision.takeover_count,
        "release_count": decision.release_count,
        "decision_reason": decision.decision_reason,
        "delay_s": decision.delay_s,
        "start_index": decision.start_index,
        "remaining_points": decision.remaining_points,
        "reference_source": decision.reference_source,
        "reference_fallback_reason": decision.reference_fallback_reason,
    }


def _time_sync_view(view: TimeSyncView | None) -> dict[str, Any] | None:
    if view is None:
        return None
    return {
        "camera_clock": view.camera_clock,
        "config_error": view.config_error,
    }


def build_health_body(
    snapshot: RuntimeSnapshot,
    sources: HealthSources,
    *,
    now_monotonic_ns: int,
    pipeline_alive: bool,
) -> dict[str, Any]:
    settings = sources.settings
    readiness = evaluate_readiness(
        snapshot, now_monotonic_ns=now_monotonic_ns, pipeline_alive=pipeline_alive
    )
    body = {
        # Unchanged on purpose: ``start_realtime.ps1`` and ``tools/check_chain.sh``
        # treat any HTTP 200 here as "uvicorn is up".  The verdict an operator
        # should act on is ``ready`` below (same predicate as ``GET /ready``).
        "ok": True,
        "pipeline_alive": bool(pipeline_alive),
        "pose_mode": settings.pose_mode,
        "inference_count": snapshot.inference_count,
        "last_error": snapshot.last_error,
        "camera_health": sources.camera_health,
        "camera_stats": sources.camera_stats,
        "pose_source": (
            "udp" if settings.pose_mode == "udp" else settings.pose_mode
        ),
        "camera_stale_ms": settings.camera_stale_ms,
        "time_sync": _time_sync_view(snapshot.time_sync),
        # Additive from Stage 3 onward.  Appended, never inserted, so the
        # historical field order above is byte-identical to what it was.
        "phase": snapshot.phase,
        "cycle_id": snapshot.cycle_id,
        "snapshot_sequence": snapshot.sequence,
        "snapshot_age_ms": snapshot_age_ms(snapshot, now_monotonic_ns),
        "uptime_s": uptime_s(snapshot, now_monotonic_ns),
        "model_loaded": snapshot.model_loaded,
        "ready": readiness.ready,
        # Appended, not inserted, for the same reason as the tail above: the
        # field order is asserted by tests/test_health_snapshot.py and read
        # positionally by nothing, but a mid-body insertion still churns every
        # stored body diff an operator compares against.  The CIMG link's
        # counters, including the vehicle peer.
        "camera_cimg": sources.camera_cimg,
        # Appended, never inserted, for the reason above.  ``None`` in inline
        # mode, so the body of a single-GPU run differs from what it was by two
        # nulls and nothing else.  Both are on /health AND /status for the same
        # reason ``camera_cimg`` is: an operator should be able to reach them
        # from whichever endpoint they already poll.
        "inference_pool": (
            None if snapshot.inference_pool is None else snapshot.inference_pool.as_dict()
        ),
        "inference_metrics": snapshot.inference_metrics,
    }
    return body


def build_ready_body(
    snapshot: RuntimeSnapshot,
    *,
    now_monotonic_ns: int,
    pipeline_alive: bool,
) -> dict[str, Any]:
    """The ``/ready`` body.  The verdict comes first; the evidence follows.

    ``GET /ready`` returns 200 or 503, always with this body - so a caller that
    only reads the status code and one that only reads the body agree.
    """
    readiness = evaluate_readiness(
        snapshot, now_monotonic_ns=now_monotonic_ns, pipeline_alive=pipeline_alive
    )
    body: dict[str, Any] = readiness.as_dict()
    body.update(
        {
            "phase": snapshot.phase,
            "cycle_id": snapshot.cycle_id,
            "snapshot_sequence": snapshot.sequence,
            "snapshot_age_ms": snapshot_age_ms(snapshot, now_monotonic_ns),
            "uptime_s": uptime_s(snapshot, now_monotonic_ns),
            "model_loaded": snapshot.model_loaded,
            "pipeline_alive": bool(pipeline_alive),
            # Both ages as numbers, next to the prose in ``checks``: a dashboard
            # should not have to parse a detail string.
            "window_age_s": window_age_s(snapshot, now_monotonic_ns),
            "pose_age_ms": pose_age_ms(snapshot, now_monotonic_ns),
            "unhealthy_readers": list(snapshot.unhealthy_readers),
            "route_receiver_listening": snapshot.route_receiver_listening,
            "route_receiver_error": snapshot.route_receiver_error,
        }
    )
    return body


def build_status_body(snapshot: RuntimeSnapshot, sources: StatusSources) -> dict[str, Any]:
    latest = (
        sources.history.latest() if sources.history is not None else None
    )
    chassis = (
        sources.pose_listener.latest_chassis() if sources.pose_listener else None
    )
    planned_route, planned_route_version = (
        sources.planned_route_cache.snapshot()
        if sources.planned_route_cache is not None
        else (None, 0)
    )
    receiver = sources.route_tcp_receiver
    return {
        "pose_timestamp_mode": (
            "unix" if sources.timestamp_manager is not None else None
        ),
        "pose_last_ts_us": latest.timestamp_us if latest else None,
        "pose_payload_ts_us": (
            latest.payload_timestamp_us if latest is not None else None
        ),
        "chassis_last_ts_us": chassis.timestamp_us if chassis else None,
        "chassis_speed_mps": chassis.speed_mps if chassis else None,
        "chassis_yaw_rate_rad_s": chassis.yaw_rate_rad_s if chassis else None,
        "chassis_packets": (
            sources.pose_listener._chassis_total if sources.pose_listener else 0
        ),
        "pose_timestamp": (
            {"timestamp_us": sources.pose_listener.last_timestamp_us}
            if sources.pose_listener
            else None
        ),
        "last_t0_us": snapshot.last_t0_us,
        "reference_source_mode": (
            sources.settings.reference_source if sources.runner_present else None
        ),
        "planned_route_cache": {
            "receiver_listening": bool(receiver is not None and receiver.listening),
            "receiver_error": receiver.last_error if receiver is not None else None,
            "available": planned_route is not None,
            "version": planned_route_version,
            "route_id": planned_route.route_id if planned_route else None,
            "coordinate_frame": (
                planned_route.coordinate_frame if planned_route else None
            ),
            "num_points": len(planned_route.points) if planned_route else 0,
            "source_timestamp_us": (
                planned_route.source_timestamp_us if planned_route else None
            ),
        },
        "last_control_flag": snapshot.last_control_flag,
        # The flag byte actually sent.  Differs from last_control_flag only for
        # a release frame emitted because the trajectory trim expired.
        "last_sent_control_flag": snapshot.last_sent_control_flag,
        "last_takeover_decision": takeover_decision_view(snapshot.last_takeover_decision),
        "last_num_waypoints": snapshot.last_num_waypoints or 0,
        "last_cot": (snapshot.last_cot[:400] + "...") if snapshot.last_cot else None,
        # Appended, for the same reason as on /health: this body's key order is
        # frozen by tests/test_health_snapshot.py, and an operator polling
        # /status for the pose rate should not have their diff churned.
        "camera_cimg": sources.camera_cimg,
        # Appended: same two objects as on /health, from the same snapshot.
        "inference_pool": (
            None if snapshot.inference_pool is None else snapshot.inference_pool.as_dict()
        ),
        "inference_metrics": snapshot.inference_metrics,
    }
