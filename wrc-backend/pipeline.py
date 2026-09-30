"""The real-time inference loop, and nothing else.

One cycle, in order:

    CameraWindowProvider.latest_window()          camera window acquisition
    -> InputBuilder.build()                       gates + ego tensors + ref pose
    -> AlpamayoRunner.infer()                     the model call
    -> TakeoverDecisionLayer.decide()             the control flag
    -> trim_waypoints_to_latest_pose()            the crop that is sent
    -> hold_waypoint_at_latest_pose()             a model stop, sent instead
    -> local_xy_to_latlon()                       model frame -> WGS84
    -> WaypointSender.send()                      the datagram
    -> _save_infer_package()                      the record

No formula, no model parameter and no wire byte is re-implemented here: each
line above is a call into the module that owns it.  This file reads as the
sequence it is, which is the point of the split - the order of these calls is
the safety property, and that order is now visible on one screen.

The rollout rules visible in the code below must stay:

* the crop that is *sent* is ``trim_waypoints_to_latest_pose``, except a
  ``model_stop_takeover``.  That frame keeps ``control=1`` and sends one
  waypoint at the latest pose.  The stop cluster is already behind the
  vehicle, and sending it would command a reverse; an empty crop would
  release the takeover;
* a crop that removed every waypoint, and is not a model stop, still sends
  a packet, with ``control=0`` and ``count=0``.  "No packet" and "not taking
  over" are indistinguishable on the wire, so staying silent would leave the
  vehicle holding its last ``control=1``.  That release frame exists only
  for this case: a usable result with nothing left to execute;
* a rejected input, an inference exception, or an unusable trajectory sends
  nothing, including no ``count=0 control=0`` release frame.
  ``TakeoverDecisionEngine.invalid`` is fail-hold: it keeps the last
  ``control_flag``.  A release frame would put ``control=0`` on the wire and
  drop an active takeover, which is the opposite of that policy.  Silence
  leaves the vehicle on the last datagram it accepted.

Logger name
    ``logging.getLogger("app")``, not ``__name__``.  Every line below appears in
    the operator's log with the ``[app]`` prefix it has always had, and the
    field greps that stream by eye; moving the code must not change what the log
    looks like.  Renaming this changes an operational interface - do it
    deliberately.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

import health
from camera.cimg_source import CimgConfig
from camera.models import CameraWindow
from camera.sync import CameraWindowProvider
from inference.alpamayo import AlpamayoRunner, ModelResult
from inference.camera_labels import _CAMERA_DISPLAY_NAMES, _camera_package_label
from inference.gpu_pool import GpuInventory, GpuScheduler
from inference.input_builder import (
    FRESHNESS_REASONS,
    REASON_CAMERA_WINDOW_STALE,
    REASON_POSE_HISTORY_UNAVAILABLE,
    InputBuilder,
    InputBuildResult,
)
from inference.metrics import InferenceMetrics
from inference.result_manager import ResultManager
from inference.tasks import InferenceResult, InferenceTask
from vehicle.planned_route import PlannedRouteTcpReceiver
from time_sync import TimestampManager
from trajectory.coordinates import local_xy_to_latlon
from trajectory.postprocess import (
    TrajectoryPostprocessResult,
    hold_waypoint_at_latest_pose,
    trim_waypoints_to_latest_pose,
)
from trajectory.takeover import (
    PlannedRoute,
    PlannedRouteCache,
    PlannedRouteReferenceProvider,
    ReferenceTrajectorySelector,
    TakeoverDecision,
    TakeoverDecisionEngine,
    TakeoverDecisionLayer,
    TrajectoryCompensator,
    TrajectoryDeviationCalculator,
)
from vehicle.history import PoseHistory
from vehicle.models import GlobalPose
from vehicle.receiver import MockPoseProvider, UdpPoseListener
from vehicle.trace import VEHICLE_TRACE_NAME, open_jsonl_trace, waypoint_record
from waypoint.sender import WaypointSender

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from config.loader import AppConfig

logger = logging.getLogger("app")

#: The inference pool's own share of the shutdown budget, used as an upper bound
#: only: ``PipelineRunner.stop`` carves the pool's slice out of whatever is left
#: of ``drain_s + reap_s``, so these can never extend the shutdown, just cap it.
#:
#: ``POOL_DRAIN_S`` has to cover one in-flight inference - a worker only reads
#: the sentinel once the model call it is in returns, so a shorter drain does not
#: make shutdown faster, it makes it a ``terminate`` on a worker holding VRAM.
POOL_DRAIN_S = 5.0
POOL_TERMINATE_S = 3.0

# Log field synonyms - two values are logged under two names each.  This is a
# deliberate record, NOT a to-do: every one of these names is grepped in the
# field (and by tools/check_chain.sh), so renaming either spelling would break
# a search that currently works.  Know the mapping before reading a log:
#
#     camera_ts  ==  trajectory_timestamp  ==  result.t0_us
#     pose_ts    ==  latest_pose_timestamp ==  the pose the crop consumed
#                                            (``ref.timestamp_us`` in the send
#                                            line, ``latest_for_trim`` in the
#                                            trim lines - the same pose, read
#                                            at two different moments)
#
# Do not add a third spelling and do not "unify" by deleting one.



def _cimg_config(settings: "AppConfig") -> CimgConfig:
    """The CIMG receiver settings taken from the effective configuration."""
    return CimgConfig(
        host=settings.cimg_host,
        port=settings.cimg_port,
        camera_indices=tuple(int(i) for i in settings.cimg_camera_indices),
        read_timeout_s=settings.cimg_read_timeout_s,
        max_payload_bytes=settings.cimg_max_payload_bytes,
        max_jpeg_bytes=settings.cimg_max_jpeg_bytes,
        queue_frames=settings.cimg_queue_frames,
        max_backlog_groups=settings.cimg_max_backlog_groups,
        group_fps=settings.cimg_group_fps,
        frame_width=settings.frame_width,
        frame_height=settings.frame_height,
    )


def _pose_payload(pose: GlobalPose | None) -> dict[str, Any] | None:
    """The package's full pose record, or None.

    Used for both ``pose_latest`` (the newest sample when the record is written)
    and ``trajectory_postprocess_input.latest_pose`` (the sample the crop
    actually saw).  Those two are usually the same object and are allowed to
    differ: the crop runs before the send, the record is written after it.
    """
    if pose is None:
        return None
    return {
        "timestamp_us": int(pose.timestamp_us),
        "payload_timestamp_us": (
            int(pose.payload_timestamp_us)
            if pose.payload_timestamp_us is not None
            else None
        ),
        "arrival_us": (
            int(pose.arrival_us) if pose.arrival_us is not None else None
        ),
        "latitude": float(pose.latitude),
        "longitude": float(pose.longitude),
        "altitude": float(pose.altitude),
        "heading_deg": float(pose.heading_deg()),
    }


class PipelineRunner(threading.Thread):
    def __init__(self, settings: AppConfig) -> None:
        super().__init__(daemon=True, name="alpamayo-pipeline")
        self.settings = settings
        self.history = PoseHistory()
        self.timestamp_manager = TimestampManager.from_config()
        self.cameras = CameraWindowProvider(
            _cimg_config(settings),
            frames_per_camera=settings.frames_per_camera,
            snapshot_keep=settings.snapshot_keep,
            target_fps=settings.target_fps,
            max_camera_skew_us=settings.max_camera_skew_us,
            window_max_age_ms=settings.window_max_age_ms,
            timestamp_manager=self.timestamp_manager,
        )
        self.input_builder = InputBuilder(
            history_steps=settings.history_steps,
            history_dt_s=settings.history_dt_s,
            camera_stale_ms=settings.camera_stale_ms,
            pose_stale_ms=settings.pose_stale_ms,
            camera_pose_delta_ms=settings.camera_pose_delta_ms,
        )
        self.model = AlpamayoRunner(settings)
        self.sender = WaypointSender(settings)
        # Pose lines and waypoint lines share this file, in arrival order.
        # Empty when ALPAMAYO_VEHICLE_TRACE_DIR is unset. The pose listener
        # and the send path are different threads; JsonlTrace locks each append.
        self._vehicle_trace = open_jsonl_trace(
            settings.vehicle_trace_dir, VEHICLE_TRACE_NAME
        )
        # -- inference pool -------------------------------------------------- #
        # Both modes construct the model runner, so every ``self.X`` published by
        # ``_publish_snapshot`` has a real writer in both modes and the inline
        # path stays rollback-able by flipping one config value.  In
        # multi_process the parent never loads the checkpoint: ``ensure_loaded``
        # is only called on the worker side, so ``self.model.loaded`` stays False
        # here and the pool's own view is what reports readiness.
        self.inference_metrics = InferenceMetrics(
            long_s=float(settings.inference_pool.metrics_window_s)
        )
        self.result_manager = ResultManager(
            result_max_age_ms=float(settings.inference_pool.result_max_age_ms),
            max_capture_lag_ms=float(settings.inference_pool.max_capture_lag_ms),
        )
        self.scheduler: GpuScheduler | None = None
        if settings.inference_pool.multi_process:
            self.scheduler = GpuScheduler(
                settings=settings,
                pool_config=settings.inference_pool,
                metrics=self.inference_metrics,
                result_manager=self.result_manager,
                on_accepted=self._on_worker_result,
                worker_options={
                    "warmup": bool(settings.inference_pool.worker_warmup),
                    # A spawned worker cannot see this process's settings, so the
                    # one endpoint it needs travels with it.
                    "hf_endpoint": settings.network.hf_endpoint,
                },
            )
        # Inline has no scheduler to allocate identities, and one counter that
        # never runs in multi_process is cheaper than a second code path.  Both
        # modes therefore number every accepted input the same way, and a
        # rejected cycle consumes nothing.
        self._inline_task_seq = 0
        # The pool's published view: state transitions and the last shutdown
        # report, written only by this thread.
        self.pool_state = "disabled" if self.scheduler is None else "created"
        self.pool_error: str | None = None
        self.pool_devices: tuple[int, ...] = (
            () if self.scheduler is None else self.scheduler.gpu_devices
        )
        self.pool_stop_report: dict[str, Any] = {}
        self.pool_inventory: GpuInventory | None = None
        self.planned_route_cache = PlannedRouteCache()
        self.route_tcp_receiver = (
            PlannedRouteTcpReceiver(self.planned_route_cache,
                                    settings.route_tcp_host, settings.route_tcp_port)
            if settings.route_tcp_enabled else None
        )
        self.planned_route_provider = PlannedRouteReferenceProvider(
            self.planned_route_cache, settings.route_max_age_s,
            settings.route_max_match_distance_m,
        )
        self.reference_provider = ReferenceTrajectorySelector(
            self.planned_route_provider,
        )
        self.trajectory_compensator = TrajectoryCompensator(
            settings.trajectory_dt_s,
            settings.takeover_min_remaining_points,
        )
        self.deviation_calculator = TrajectoryDeviationCalculator()
        self.takeover_engine = TakeoverDecisionEngine(
            settings.takeover_threshold_m,
            settings.release_threshold_m,
            settings.takeover_confirmations,
            settings.release_confirmations,
        )
        self.decision_layer = TakeoverDecisionLayer(
            takeover_engine=self.takeover_engine,
            trajectory_compensator=self.trajectory_compensator,
            deviation_calculator=self.deviation_calculator,
            reference_provider=self.reference_provider,
            planned_route_cache=self.planned_route_cache,
            # CIMG publishes one stats row per camera.  The count has to be
            # the row count ``stats()`` will actually return, or every cycle
            # is ``camera_status_unavailable`` and takeover can never start.
            expected_camera_count=self.cameras.reported_camera_count(),
            camera_stale_ms=settings.camera_stale_ms,
            pose_stale_ms=settings.pose_stale_ms,
            stop_progress_m=settings.stop_progress_m,
        )
        self.pose_listener: UdpPoseListener | None = None
        self.mock_pose: MockPoseProvider | None = None
        # NOT ``self._stop``: ``threading.Thread`` has a private ``_stop()``
        # method of its own, used by ``_wait_for_tstate_lock``.  An instance
        # attribute of that name shadows it, and the first ``join()`` on this
        # runner then dies with "'Event' object is not callable" - which is
        # exactly what adding a bounded drain in Stage 3 ran into.  The name is
        # load-bearing; do not shorten it.
        self._stop_event = threading.Event()
        self.last_result: ModelResult | None = None
        self.last_control_flag: int | None = None
        # The flag byte actually put on the wire.  It matches last_control_flag
        # except when the trajectory trim removed every waypoint, in which case
        # the release frame is sent with control=0.
        self.last_sent_control_flag: int | None = None
        self.last_takeover_decision: TakeoverDecision | None = None
        self.error: str | None = None
        self.inference_count = 0
        self._last_pose_warn_at = 0.0
        self._last_camera_stale_warn_at = 0.0
        self._last_freshness_warn_at = 0.0
        # Watchdog for "no camera window at all" (e.g. decoder stalled).
        self._last_window_ok_at = time.monotonic()
        self._last_no_window_warn_at = 0.0
        # Distinct from the watchdog baseline above, which is seeded at
        # construction so the first seconds do not log a stall.  This one is
        # None until a window has really been built, which is the only thing
        # that can answer "has this pipeline ever produced a window".
        self._last_window_seen_at: float | None = None
        self._last_rejection_reason: str | None = None
        self.last_inference_at: float | None = None

        # Stage 3: the immutable view published for /health, /status and /ready.
        # Only this thread writes it, so no lock is needed - see health.py.
        self.cycle_id = 0
        self.started_at_monotonic: float | None = None
        self.snapshot: health.RuntimeSnapshot = health.EMPTY_SNAPSHOT
        # Publish the static half now so a handler that somehow runs before
        # start() still sees the real pose mode and thresholds.
        self._publish_snapshot(phase=health.PHASE_CREATED)

    def update_planned_route(self, route: PlannedRoute) -> int:
        """Protocol adapters call this; no network format is assumed here."""
        version = self.planned_route_cache.update(route)
        logger.info(
            "planned route cache updated: version=%d route_id=%s points=%d frame=%s",
            version,
            route.route_id,
            len(route.points),
            route.coordinate_frame,
        )
        return version

    def _trajectory_control_decision(
        self, result: ModelResult
    ) -> TakeoverDecision:
        """Delegate to the decision layer, supplying its inputs in order.

        The camera stats and the latest pose are passed as callables, not
        values: the layer reads the camera stats, then the pose, then the
        clock, and fetching either one here would reorder those reads.
        """
        return self.decision_layer.decide(
            positions=result.positions,
            t0_us=result.t0_us,
            ref_pose=result.ref_pose,
            camera_stats_fn=lambda: self.cameras.stats().get("readers", []),
            latest_pose_fn=self.history.latest,
        )

    def _log_rejected_input(
        self, built: InputBuildResult, window: CameraWindow
    ) -> None:
        """Log a rejected cycle the way the inline loop always did.

        Each reason has its own throttled line and its own throttle window
        (1 s for freshness, 2 s for the other two); the formats are unchanged
        because they are read in the field.  This method only logs - the caller
        has already told the takeover engine, and the caller still owns the
        sleep and the retry.
        """
        now = time.monotonic()
        latest = built.latest_pose
        reason = built.rejection_reason
        if reason in FRESHNESS_REASONS:
            if now - self._last_freshness_warn_at >= 1.0:
                logger.warning(
                    "freshness gate blocked inference: reason=%s "
                    "frame_id=%s camera_ts=%s pose_ts=%s "
                    "frame_age_ms=%.1f pose_age_ms=%.1f delta_ms=%.1f "
                    "cycle_id=%d",
                    reason,
                    built.latest_frame_ids,
                    window.t0_us,
                    latest.timestamp_us if latest is not None else None,
                    built.camera_age_ms,
                    built.pose_age_ms,
                    built.camera_pose_delta_ms,
                    self.cycle_id,
                )
                self._last_freshness_warn_at = now
            return
        if reason == REASON_CAMERA_WINDOW_STALE:
            if now - self._last_camera_stale_warn_at >= 2.0:
                logger.warning(
                    "camera window is stale: t0=%s is %.0f ms "
                    "behind latest pose ts=%s; skipping inference "
                    "(if the lag is implausibly large, camera and "
                    "pose timestamps may use different time domains) "
                    "cycle_id=%d",
                    window.t0_us,
                    built.camera_lag_ms,
                    latest.timestamp_us,
                    self.cycle_id,
                )
                self._last_camera_stale_warn_at = now
            return
        if reason == REASON_POSE_HISTORY_UNAVAILABLE:
            # Only worth reporting once the pose is actually missing or older
            # than the pose budget; otherwise this is ordinary startup warm-up.
            if latest is None or (
                window.t0_us - latest.timestamp_us
            ) / 1000.0 > self.settings.pose_stale_ms:
                if now - self._last_pose_warn_at >= 2.0:
                    logger.warning(
                        "waiting for vehicle pose history at t0=%s",
                        window.t0_us,
                    )
                    self._last_pose_warn_at = now

    # ------------------------------------------------------------------ #
    # Stage 3: the published view
    # ------------------------------------------------------------------ #
    #
    # Log field synonyms.  Two values below are logged under two names each, and
    # both spellings are load-bearing:
    #
    #   camera_ts      == trajectory_timestamp  == result.t0_us
    #   pose_ts        == latest_pose_timestamp == the pose's timestamp_us
    #
    # The runbook greps one spelling on one line and the other spelling on
    # another, so unifying them would break a field grep.  Nothing new is added
    # under either name; this comment is the mapping.

    def _time_sync_view(self) -> health.TimeSyncView:
        """The timestamp authority's config, frozen for the snapshot.

        Constant after construction, so freezing it cannot go stale - the
        snapshot carries it so a handler reads one object instead of reaching
        into the runner for a static value.
        """
        return health.TimeSyncView(
            camera_clock="capture_time_ns",
            config_error=self.timestamp_manager.config_error,
        )

    def _inference_pool_view(self) -> health.InferencePoolView | None:
        """The pool as of now, or None when this process has no pool.

        Built from the scheduler's own views rather than from counters kept
        here: the scheduler is the single writer of worker state, and a second
        copy in the runner is a copy that can disagree with it.

        Only ever called from ``_publish_snapshot``, i.e. only from the loop
        thread, which is what makes reading ``worker_views()`` without a lock
        correct - ``GpuScheduler``'s own docstring makes the same promise.

        Costs about 50 us with four workers and full latency windows, measured;
        the loop publishes this at its poll rate, so it is well under a percent
        of a core.
        """
        scheduler = self.scheduler
        if scheduler is None:
            return None
        workers = tuple(
            health.InferenceWorkerView(
                worker_id=int(view.worker_id),
                device=int(view.device),
                state=str(view.state),
                task_seq=None if view.task_seq is None else int(view.task_seq),
                infer_count=int(view.infer_count),
                last_infer_ms=view.last_infer_ms,
                avg_infer_ms=view.avg_infer_ms,
                p50_infer_ms=view.p50_infer_ms,
                p95_infer_ms=view.p95_infer_ms,
                infer_samples=int(view.infer_samples),
                allocated_bytes=int(view.memory.allocated_bytes),
                reserved_bytes=int(view.memory.reserved_bytes),
                peak_allocated_bytes=int(view.peak_allocated_bytes),
                error=view.error,
                endpoint=str(view.endpoint),
            )
            for view in scheduler.worker_views()
        )
        return health.InferencePoolView(
            mode=str(self.settings.inference_pool.mode),
            state=str(self.pool_state),
            devices=tuple(int(d) for d in scheduler.gpu_devices),
            worker_count=len(workers),
            ready_workers=int(scheduler.ready_worker_count),
            busy_workers=int(scheduler.busy_worker_count),
            loading_workers=int(scheduler.loading_worker_count),
            failed_workers=int(scheduler.failed_worker_count),
            unreachable_workers=int(scheduler.unreachable_worker_count),
            minimum_ready_workers=int(
                self.settings.inference_pool.minimum_ready_workers
            ),
            latest_only=bool(scheduler.latest_only),
            pending_capacity=int(scheduler.pending_capacity),
            pending_task_seq=scheduler.pending_task_seq,
            pending_count=int(scheduler.pending_count),
            workers=workers,
            error=self.pool_error,
            stop_report=dict(self.pool_stop_report),
        )

    def _publish_snapshot(self, **updates: Any) -> None:
        """Publish this thread's own state as one immutable object.

        Called from the top of every loop iteration, so every exit path
        (``continue``, an exception, or running off the end) has already
        published the iteration it was in - which is why the loop body keeps its
        shape and there is no ``try``/``finally`` around it.

        Reads only attributes this thread writes.  Everything owned by another
        thread - decoder frame counters, the vehicle's packet counters, the
        planned-route cache - stays a live read in the handlers, for the reason
        in health.py: publishing a rate field once per cycle would freeze it
        exactly when the pipeline is slow enough to need it.
        """
        latest_pose = self.history.latest()
        last = self.last_result
        receiver = self.route_tcp_receiver
        fields: dict[str, Any] = {
            # Lifecycle.  Overridden by the caller's ``updates`` when start() or
            # stop() publishes a transition.
            "phase": self.snapshot.phase,
            "started_at_monotonic": self.started_at_monotonic,
            "cycle_id": self.cycle_id,
            # Cycle progress.
            "inference_count": self.inference_count,
            "model_loaded": self.model.loaded,
            "last_error": self.error,
            "last_rejection_reason": self._last_rejection_reason,
            "last_t0_us": last.t0_us if last is not None else None,
            "last_control_flag": self.last_control_flag,
            "last_sent_control_flag": self.last_sent_control_flag,
            "last_num_waypoints": len(last.positions) if last is not None else None,
            "last_cot": (last.cot or None) if last is not None else None,
            "last_takeover_decision": self.last_takeover_decision,
            # Instants.  ``last_window_ok_at`` is the "a window was really built"
            # stamp, NOT ``_last_window_ok_at``: the latter is a stall-watchdog
            # baseline seeded at construction, so publishing it would claim a
            # window had been seen before one ever was.
            "last_window_ok_at": self._last_window_seen_at,
            "last_inference_at": self.last_inference_at,
            "last_pose_arrival_us": (
                latest_pose.arrival_us if latest_pose is not None else None
            ),
            # Component state as of this publish.
            "unhealthy_readers": self.cameras.unhealthy_readers(),
            "route_receiver_listening": (
                None if receiver is None else bool(receiver.listening)
            ),
            "route_receiver_error": (
                None if receiver is None else receiver.last_error
            ),
            # Static values, repeated so a handler reads one object.
            "pose_mode": self.settings.pose_mode,
            "camera_stale_ms": self.settings.camera_stale_ms,
            "pose_stale_ms": self.settings.pose_stale_ms,
            "time_sync": self._time_sync_view(),
            # None in inline mode.  The scheduler is the single writer of
            # everything in here, so this is a projection, not a second record.
            "inference_pool": self._inference_pool_view(),
            # The three frequencies and the counters behind them, frozen at this
            # instant.  Handlers read this instead of calling the metrics object,
            # so a rate can never be sampled from the middle of a task the loop
            # is in the middle of recording.  About 130 us; see the note above.
            "inference_metrics": self.inference_metrics.snapshot(),
        }
        fields.update(updates)
        self.snapshot = health.advance(self.snapshot, **fields)

    def start(self) -> None:
        self.started_at_monotonic = time.monotonic()
        # The pool first, and deliberately.  Spawning the workers is the slowest
        # step here and the only one that can fail on its own terms - a device
        # that does not exist, a device listed twice, too little VRAM - and a
        # failure must land before the cameras and the pose listener are up, so
        # an aborted startup has not already taken the UDP ports the next attempt
        # needs.  It is left to raise: a pool that cannot start is a process that
        # cannot drive, and /ready would never pass - better a startup that fails
        # with the reason than one that runs blind.
        if self.scheduler is not None:
            try:
                self.pool_inventory = self.scheduler.start()
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                self.pool_state = "failed"
                self.pool_error = str(exc)
                raise
            self.pool_state = "loading"
        # Published before the thread exists, not after: "running" is a
        # lifecycle statement, and the loop's first act is to publish, so
        # publishing it afterwards could be undone by a "created" carried
        # forward from the initial snapshot.
        self._publish_snapshot(phase=health.PHASE_RUNNING)
        if self.settings.pose_mode == "mock":
            self.mock_pose = MockPoseProvider(self.history, self.timestamp_manager)
            self.mock_pose.start()
        else:
            self.pose_listener = UdpPoseListener(
                self.history,
                # Passed rather than left to the listener's own defaults: the
                # endpoint is configuration, and a setting that is read into
                # ``settings`` and then not forwarded is a knob that silently
                # does nothing - which is what ``pose_udp_host``/``port`` were
                # until now, while ``/health`` and the startup log reported the
                # port as if it were live.
                pose_host=self.settings.pose_udp_host,
                pose_port=self.settings.pose_udp_port,
                azimuth_offset_deg=self.settings.azimuth_offset_deg,
                timestamp_manager=self.timestamp_manager,
                trace=self._vehicle_trace,
            )
            self.pose_listener.start()
        self.cameras.start()
        if self.route_tcp_receiver is not None:
            self.route_tcp_receiver.start()
        super().start()

    def stop(self, *, drain_s: float | None = None, reap_s: float | None = None) -> health.ShutdownReport:
        """Stop everything, bounded, and report what could not be stopped.

        Four phases, in this order, because the order is the safety property:

        A  signal   flag only.  The CIMG reader sees the flag on its next poll
                    and closes the listening socket, so the next start can bind
                    ``network.cimg``.
        B  drain    wait for this thread.  An in-flight cycle is allowed to
                    finish and send: the vehicle holds its last ``control=1``
                    when no packet arrives, so aborting mid-cycle is worse than
                    arriving late.
        C  reap     the inference pool, then receivers and the CIMG reader.
                    Bounded on its own budget so it can never be starved by an
                    overrunning drain.  The pool goes first and only here,
                    because the loop ticks the scheduler every iteration.
        D  release  close the sender socket, but ONLY if B confirmed the loop
                    stopped: a late ``send()`` on a closed socket raises OSError,
                    which the loop would log as a ``pipeline cycle failed``.

        Returns a ``health.ShutdownReport``.  Callers that ignore it - including
        the pre-Stage-3 call sites and ``tests/test_waypoint_trimming.py`` - keep
        working unchanged.
        """
        drain_s = health.DRAIN_S if drain_s is None else drain_s
        reap_s = health.REAP_S if reap_s is None else reap_s
        started_at = time.monotonic()

        # -- phase A: signal, no waiting ---------------------------------- #
        self._stop_event.set()
        # Published after the flag, so the loop cannot publish again afterwards
        # (its ``while`` guard is the flag) and this is the last word on phase.
        self._publish_snapshot(phase=health.PHASE_STOPPING)
        self.cameras.stop()

        # -- phase B: drain the loop -------------------------------------- #
        components: list[str] = ["pipeline"]
        stopped: list[str] = []
        failed: list[str] = []
        if self.ident is None:
            # Never started: there is no thread to drain, and ``join()`` would
            # raise "cannot join thread before it is started".  Reachable in
            # production - ``_startup`` assigns ``_runner`` before it calls
            # ``start()``, so a failure in between (or an ASGI server that gives
            # up during startup) leaves an unstarted runner for ``_shutdown``.
            # Counted as stopped, not failed: it is stopped, and an ERROR line
            # about it would be wrong.
            stopped.append("pipeline")
        else:
            self.join(drain_s)
            if self.is_alive():
                failed.append("pipeline")
            else:
                stopped.append("pipeline")

        # -- phase C: reap everything else -------------------------------- #
        if self.scheduler is not None:
            # After the drain, never before: the loop calls ``scheduler.tick()``
            # on every iteration, and tick on a scheduler whose workers are gone
            # would read every in-flight result as a worker death.  Phase B has
            # confirmed the loop is no longer running, so nothing will tick.
            #
            # Its own budget, carved out of what is left of the shutdown's: a
            # worker part-way through a 10B inference does not see the sentinel
            # until it finishes, so the drain is a real deadline - and a shutdown
            # that overruns is a shutdown that leaves a GPU allocated for the
            # next process to trip over.  Under pressure the pool gets less drain
            # and then terminate, which is the right order to sacrifice in.
            components.append("gpu_pool")
            remaining_s = max(0.0, (drain_s + reap_s) - (time.monotonic() - started_at))
            pool_drain_s = min(POOL_DRAIN_S, remaining_s)
            pool_terminate_s = min(POOL_TERMINATE_S, max(0.0, remaining_s - pool_drain_s))
            try:
                self.pool_stop_report = self.scheduler.stop(
                    drain_s=pool_drain_s, terminate_s=pool_terminate_s
                )
                ungraceful = bool(
                    self.pool_stop_report.get("terminated")
                    or self.pool_stop_report.get("killed")
                )
                self.pool_state = "stopped"
                stopped.append("gpu_pool")
                if ungraceful:
                    # Reported as stopped AND logged as a warning: the workers are
                    # gone, which is what ``stopped`` claims, but they did not go
                    # quietly and that is worth an ERROR line on the way out.
                    logger.error(
                        "inference pool stopped ungracefully: terminated=%s "
                        "killed=%s elapsed_ms=%s",
                        self.pool_stop_report.get("terminated"),
                        self.pool_stop_report.get("killed"),
                        self.pool_stop_report.get("elapsed_ms"),
                    )
            except Exception:  # noqa: BLE001 - report, never raise
                logger.exception("inference pool did not stop")
                self.pool_state = "failed"
                failed.append("gpu_pool")
        if self.route_tcp_receiver is not None:
            components.append("route_tcp_receiver")
            try:
                self.route_tcp_receiver.stop()
                stopped.append("route_tcp_receiver")
            except Exception:  # noqa: BLE001 - report, never raise
                logger.exception("route receiver did not stop")
                failed.append("route_tcp_receiver")
        if self.pose_listener is not None:
            components.append("pose_listener")
            try:
                self.pose_listener.stop()
                # Only join a thread that actually started.  ``Thread.join()``
                # raises for one that did not, and a startup that failed before
                # ``pose_listener.start()`` leaves exactly that - which would be
                # reported as a component that failed to stop.
                if self.pose_listener.ident is not None:
                    self.pose_listener.join(reap_s)
                stopped.append("pose_listener")
            except Exception:  # noqa: BLE001
                logger.exception("pose listener did not stop")
                failed.append("pose_listener")
        if self.mock_pose is not None:
            components.append("mock_pose")
            try:
                self.mock_pose.stop()
                # ``stop()`` only sets the flag, and the flag is checked between
                # 0.1 s sleeps, so the thread is gone almost immediately - but
                # "almost" is not a bound.  Joined for the same reason the pose
                # listener is: a component that reports "stopped" while its
                # thread still runs is the failure this report exists to catch.
                # Guarded by ``ident`` because joining a thread that never
                # started raises, and ``pose_mode`` may have been "udp" this run.
                if self.mock_pose.ident is not None:
                    self.mock_pose.join(reap_s)
                stopped.append("mock_pose")
            except Exception:  # noqa: BLE001
                logger.exception("mock pose provider did not stop")
                failed.append("mock_pose")
        components.append("cameras")
        try:
            self.cameras.join(reap_s)
            still_alive = self.cameras.alive_readers()
            if still_alive:
                failed.append("cameras")
                logger.error(
                    "camera readers still running after %.1fs: %s",
                    reap_s,
                    ", ".join(still_alive),
                )
            else:
                stopped.append("cameras")
        except Exception:  # noqa: BLE001
            logger.exception("cameras did not stop")
            failed.append("cameras")

        # -- phase D: release --------------------------------------------- #
        if "pipeline" in stopped:
            try:
                self.sender.close()
            except Exception:  # noqa: BLE001
                logger.exception("waypoint sender did not close")
                failed.append("sender")
            # Published only here, where the drain has confirmed the loop is
            # gone.  The phase is not a statement about what this method did, it
            # is a statement about the process, and /ready reads it as one: check
            # 1 (``not_stopping``) passes for anything but "stopping"/"stopped".
            self._publish_snapshot(phase=health.PHASE_STOPPED)
        else:
            # The drain overran and the loop is still executing a cycle.  It may
            # yet stop, but it has not, so the phase stays where the truth is -
            # otherwise /ready would pass ``not_stopping`` and report readiness
            # for a process that is still running its pipeline.  Re-published
            # rather than left alone so the sequence advances and a reader can
            # see that shutdown concluded, and concluded unfinished.
            self._publish_snapshot(phase=health.PHASE_STOPPING)

        elapsed_ms = (time.monotonic() - started_at) * 1000.0
        report = health.ShutdownReport(
            components=tuple(components),
            stopped=tuple(stopped),
            failed=tuple(failed),
            elapsed_ms=elapsed_ms,
            drain_s=drain_s,
            reap_s=reap_s,
            budget_exceeded=elapsed_ms > (drain_s + reap_s) * 1000.0,
        )
        return report

    def _save_infer_package(
        self,
        window: CameraWindow,
        record: dict[str, Any],
    ) -> str | None:
        """Write one self-contained package per inference.

        Directory layout:
          <ALPAMAYO_INFER_PACKAGE_DIR>/<send_time>/infer.json
          <ALPAMAYO_INFER_PACKAGE_DIR>/<send_time>/images/<label>_t<N>.jpg

        The 16 JPEGs are exactly the model-visible window frames
        (4 cameras x 4 temporal frames), so the record can be replayed later.

        Best-effort by design: the vehicle's cycle must not fail because a disk
        filled up.  Every failure is logged and swallowed.
        """
        base_dir = self.settings.infer_package_dir
        if not base_dir:
            return None
        try:
            send_ts = float(record.get("send_timestamp", time.time()))
            sec_part = int(send_ts)
            frac_us = int(round((send_ts - sec_part) * 1_000_000))
            if frac_us >= 1_000_000:
                frac_us = 999_999
            base_name = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(sec_part)
            ) + f"_{frac_us:06d}"
            pkg_dir = os.path.join(base_dir, base_name)
            suffix = 1
            while os.path.exists(pkg_dir):
                pkg_dir = os.path.join(base_dir, f"{base_name}_{suffix}")
                suffix += 1
            img_dir = os.path.join(pkg_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            images: list[dict[str, Any]] = []
            frames = window.frames
            for cam_pos, cam_idx in enumerate(window.camera_indices):
                label = _camera_package_label(int(cam_idx))
                frame_ids = (
                    list(window.frame_ids[cam_pos])
                    if getattr(window, "frame_ids", None)
                    and cam_pos < len(window.frame_ids)
                    else []
                )
                frame_times = (
                    list(window.frame_timestamps_us[cam_pos])
                    if getattr(window, "frame_timestamps_us", None)
                    and cam_pos < len(window.frame_timestamps_us)
                    else []
                )
                frame_arrivals = (
                    list(window.frame_arrivals_us[cam_pos])
                    if getattr(window, "frame_arrivals_us", None)
                    and cam_pos < len(window.frame_arrivals_us)
                    else []
                )
                for t_idx in range(int(frames.shape[1])):
                    rgb_chw = frames[cam_pos, t_idx]
                    if hasattr(rgb_chw, "detach"):
                        arr = rgb_chw.detach().cpu().numpy()
                    else:
                        arr = np.asarray(rgb_chw)
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    hwc_bgr = cv2.cvtColor(
                        np.transpose(arr, (1, 2, 0)),
                        cv2.COLOR_RGB2BGR,
                    )
                    fname = f"{label}_t{t_idx}.jpg"
                    cv2.imwrite(
                        os.path.join(img_dir, fname),
                        hwc_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    images.append(
                        {
                            "file": fname,
                            "camera_index": int(cam_idx),
                            "camera_label": label,
                            "display": _CAMERA_DISPLAY_NAMES.get(
                                int(cam_idx), str(cam_idx)
                            ),
                            "temporal_index": int(t_idx),
                            "frame_id": (
                                frame_ids[t_idx]
                                if t_idx < len(frame_ids)
                                else None
                            ),
                            # Real decoded clock of this exact frame (what the
                            # buffer carries), as opposed to the target grid
                            # timestamp below.
                            "frame_timestamp_us": (
                                int(frame_times[t_idx])
                                if t_idx < len(frame_times)
                                else None
                            ),
                            # Server monotonic decode time: stays comparable to
                            # pose arrival even after the clock domain switch.
                            "frame_arrival_us": (
                                frame_arrivals[t_idx]
                                if t_idx < len(frame_arrivals)
                                else None
                            ),
                            "target_timestamp_us": (
                                int(window.timestamps_us[t_idx])
                                if t_idx < len(window.timestamps_us)
                                else None
                            ),
                        }
                    )

            record["package_dir"] = str(pkg_dir)
            record["images"] = images
            with open(
                os.path.join(pkg_dir, "infer.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2)
            return str(pkg_dir)
        except Exception as exc:  # noqa: BLE001 - package saving is best-effort
            logger.warning("failed to save inference package: %r", exc)
            return None

    def run(self) -> None:
        """The loop, in whichever of the two modes the config selected.

        A dispatcher rather than a branch inside the loop: the two modes differ
        in how often they look for work and in whether ``min_cycle_s`` gates
        them, and threading an ``if`` through the cycle body would make the
        inline path - the one that has flown and must stay rollback-able -
        depend on pool code.  Both call the same ``_handle_result``.
        """
        if self.scheduler is not None:
            self._run_multi_process()
        else:
            self._run_inline()

    def _acquire_window(self) -> CameraWindow | None:
        """The newest complete camera window, or None if there is not one yet.

        Shared by both loops so the stall bookkeeping has one implementation:
        the two "window is fresh" stamps below are what ``/ready``'s
        ``camera_window_fresh`` and ``/health``'s camera view read.

        The timeouts are the ones the inline path has always used, and they are
        deliberately NOT part of the pool config.  ``sync_timeout_s`` is how long
        the provider may spend accumulating the four frames a window needs, so
        it is part of the camera/Pose sync contract rather than a scheduling
        knob - shortening it to make the scheduler loop tighter would risk never
        filling a window at all.
        """
        window = self.cameras.latest_window(
            sync_timeout_s=2.0,
            catch_up_timeout_s=0.05,
        )
        if window is None:
            now = time.monotonic()
            stalled_s = now - self._last_window_ok_at
            if stalled_s * 1000 > self.settings.camera_stale_ms:
                self.last_takeover_decision = self.takeover_engine.invalid(
                    "camera_window_unavailable"
                )
            if stalled_s >= 10.0 and now - self._last_no_window_warn_at >= 10.0:
                logger.error(
                    "no camera window for %.0f s; camera stats: %s",
                    stalled_s,
                    json.dumps(self.cameras.stats(), ensure_ascii=False),
                )
                self._last_no_window_warn_at = now
            return None
        # Both stamps, from one clock read so they can never disagree.
        # The watchdog baseline above is seeded at construction; this one
        # is the record that the pipeline has really built a window,
        # which is what /ready's camera_window_fresh check needs - a
        # value that only moves on the *stall* path would leave a healthy
        # pipeline permanently reporting no_camera_window.
        now_window = time.monotonic()
        self._last_window_ok_at = now_window
        self._last_window_seen_at = now_window
        self.error = None
        return window

    def _run_inline(self) -> None:
        settings = self.settings
        while not self._stop_event.is_set():
            self.cycle_id += 1
            # Published at the top of the iteration, carrying the results of the
            # one before it.  Every exit path below - ``continue``, an
            # exception, or running off the end - has therefore already
            # published, which is why the body keeps its shape and there is no
            # try/finally re-indenting 380 lines for no behaviour change.
            self._publish_snapshot()
            try:
                window = self._acquire_window()
                if window is None:
                    time.sleep(0.05)
                    continue
                # Assemble the model input, or learn why this cycle cannot run.
                # A rejection is ordinary control flow: report it and retry.
                built = self.input_builder.build(window, self.history)
                if not built.accepted:
                    # Fail-hold: do not send a release frame.  control=0 on the
                    # wire would drop a takeover that invalid() is required to
                    # keep.  See the module docstring.
                    self.last_takeover_decision = self.takeover_engine.invalid(
                        built.rejection_reason
                    )
                    self._last_rejection_reason = built.rejection_reason
                    self._log_rejected_input(built, window)
                    time.sleep(0.05)
                    continue
                # Cleared here rather than never, so /ready's gate_rejected
                # warning means "the newest cycle was rejected" and cannot
                # become permanent wallpaper after one bad cycle hours ago.
                self._last_rejection_reason = None
                # The reference pose anchors the model frame (t0 = newest camera
                # frame, interpolated from the pose history to that instant).
                ref = built.reference_pose
                task = InferenceTask.from_build(
                    self._next_task_seq(), built, window, cycle_id=self.cycle_id
                )
                logger.info(
                    "inference started: camera_ts=%s pose_ts=%s frame_age_ms=%.1f "
                    "cycle_id=%d",
                    window.t0_us,
                    ref.timestamp_us,
                    built.camera_age_ms,
                    self.cycle_id,
                )
                inference_started = time.monotonic()
                result = self.model.infer(window, built.ego_xyz, built.ego_rot, ref)
                inference_latency_ms = (
                    time.monotonic() - inference_started
                ) * 1000.0
                # ``min_cycle_s`` is the inline mode's only rate limit and stays
                # exactly where it was: after the send, skipped by the
                # invalid-trajectory path, which is why the handler's verdict is
                # consulted here rather than unconditionally.
                if self._handle_result(
                    result, task, inference_latency_ms=inference_latency_ms
                ):
                    time.sleep(settings.min_cycle_s)
            except Exception as exc:  # noqa: BLE001
                # Same fail-hold as a rejected input: no release frame.
                self.last_takeover_decision = self.takeover_engine.invalid("pipeline_exception")
                self.error = str(exc)
                logger.exception("pipeline cycle failed cycle_id=%d", self.cycle_id)
                time.sleep(1.0)

    def _run_multi_process(self) -> None:
        """The pool mode: freeze input, hand it to a worker, publish what comes back.

        Three things make this loop different from ``_run_inline``, and each is
        deliberate:

        *no model call.*  This thread never touches a GPU.  It freezes input and
         drains results; the workers own the devices.  That is the whole reason
         the mode exists, and it is why nothing here can be blocked by a 1.5 s
         inference.

        *no* ``min_cycle_s``.  The knob exists to stop the inline loop from
         out-running its own model, which cannot happen when four workers are
         running concurrently - the pool is self-limiting through the pending
         slot.  Sleeping here would throw away exactly the throughput the extra
         GPUs were bought for, so the setting is inert and says so once, loudly,
         at startup.

        *``tick()`` twice per iteration*, immediately after acquiring a window
         and after submitting.  Tick is what moves results from the workers'
         queues through the Result Manager and out to UDP; the only place a
         finished inference can wait is a blocking ``latest_window()`` call, and
         the ticks on either side of it bound that wait.
        """
        scheduler = self.scheduler
        assert scheduler is not None  # run() only routes here when it exists
        pool = self.settings.inference_pool
        poll_s = pool.poll_interval_s
        if self.settings.min_cycle_s > 0:
            logger.warning(
                "mode=multi_process: min_cycle_s=%.3f is INERT - the pool is "
                "continuous and paced by scheduler_poll_interval_ms=%.1f "
                "(%.0f Hz), not by min_cycle_s. %d worker(s) on device(s) %s. "
                "Set min_cycle_s=0 to silence this.",
                self.settings.min_cycle_s,
                pool.scheduler_poll_interval_ms,
                1.0 / poll_s if poll_s > 0 else 0.0,
                len(scheduler.gpu_devices),
                ",".join(str(d) for d in scheduler.gpu_devices),
            )
        while not self._stop_event.is_set():
            # Published every iteration rather than once per inference: there is
            # no cycle to be in the middle of here, and /health is how an
            # operator watches a pool whose whole job is to be faster than the
            # one-inference-at-a-time view the snapshot used to describe.
            self._publish_snapshot()
            try:
                scheduler.tick()
                window = self._acquire_window()
                if window is None:
                    time.sleep(poll_s)
                    continue
                built = self.input_builder.build(window, self.history)
                if not built.accepted:
                    # Fail-hold, same as the inline loop: no release frame.
                    self.last_takeover_decision = self.takeover_engine.invalid(
                        built.rejection_reason
                    )
                    self._last_rejection_reason = built.rejection_reason
                    self._log_rejected_input(built, window)
                    time.sleep(poll_s)
                    continue
                self._last_rejection_reason = None
                # Only a complete, accepted input consumes a cycle_id - the same
                # meaning it has inline, where there is one cycle per inference.
                self.cycle_id += 1
                task = InferenceTask.from_build(
                    self._next_task_seq(), built, window, cycle_id=self.cycle_id
                )
                verdict = scheduler.submit(task)
                logger.info(
                    "task submitted: task_seq=%d capture_ts=%s cycle_id=%d "
                    "verdict=%s ready_workers=%d busy_workers=%d pending=%d",
                    task.task_seq,
                    task.capture_timestamp_us,
                    self.cycle_id,
                    verdict,
                    scheduler.ready_worker_count,
                    scheduler.busy_worker_count,
                    scheduler.pending_count,
                )
                # No sleep on this path.  The next iteration's tick() is what
                # turns a finished worker into a sent packet, and the window
                # fetch right after it returns None immediately for a key the
                # provider has already handed out - that, not a sleep, is what
                # paces this loop.
                scheduler.tick()
            except Exception as exc:  # noqa: BLE001
                # Fail-hold, same as the inline loop: no release frame.
                self.last_takeover_decision = self.takeover_engine.invalid("pipeline_exception")
                self.error = str(exc)
                logger.exception("pipeline cycle failed cycle_id=%d", self.cycle_id)
                time.sleep(poll_s)

    def _next_task_seq(self) -> int:
        """The next identity for a frozen input.

        ``task_seq`` is the only key anything downstream orders by, so it must
        be strictly increasing, unique, and handed out only to input that passed
        every gate - which is why it is taken here, after ``build()`` said yes,
        and not at the top of the cycle.

        Inline the counter is the pipeline's own.  Multi-process it belongs to
        the scheduler, which also keeps the pending slot and the
        dispatch record in the Result Manager keyed by it.  A capture timestamp
        is deliberately never consulted: the camera clock is remote and can
        step, so it cannot be allowed to decide which of two tasks is newer.

        Only ever called from the loop thread.
        """
        if self.scheduler is not None:
            return self.scheduler.next_task_seq()
        self._inline_task_seq += 1
        return self._inline_task_seq

    def _on_worker_result(self, result: InferenceResult, task: InferenceTask) -> None:
        """A worker's answer that the Result Manager has fully validated.

        Reached from ``GpuScheduler.tick()``, i.e. from the same thread that
        runs the loop - which is what makes the single-writer property hold
        without a lock and without a queue on this side.

        This is the only door into the takeover state machine, the latency crop,
        the coordinate transform, the UDP socket and the inference package when
        the pool is in use.  A worker cannot decide, cannot send and cannot
        write a record; everything it produces is a candidate until it arrives
        here.  Results may arrive in any order, and the Result Manager has
        already dropped the ones that lost - so nothing below needs to know
        which worker answered or when.
        """
        self._handle_result(
            result.model_result,
            task,
            inference_latency_ms=float(result.infer_ms),
            worker_id=int(result.worker_id),
            device=int(result.device),
            endpoint=self._worker_endpoint(int(result.worker_id)),
        )


    def _handle_result(
        self,
        result: ModelResult,
        task: InferenceTask,
        *,
        inference_latency_ms: float,
        worker_id: int | None = None,
        device: int | None = None,
        endpoint: str = "",
    ) -> bool:
        """Everything that happens after a model call, in exactly one place.

        This is the single writer.  The takeover state machine, the latency
        crop, the local-to-WGS84 transform, the UDP socket and the inference
        package are all touched here and nowhere else, which is what keeps
        ``WaypointSender.last_seq``, ``last_control_flag`` and the takeover
        engine's counters meaningful.

        Both modes call it, from this thread:

        *inline*          right after ``model.infer`` returns, with the window
                          still in hand;
        *multi_process*   from the result drain inside ``GpuScheduler.tick()``,
                          which the loop below also runs on this thread.

        The two paths therefore cannot drift apart - and because the argument
        that carries the input is a frozen :class:`InferenceTask` rather than
        the loop's own locals, the multi-process call does not need the window,
        the build result or the reference pose to still be in scope.

        ``worker_id``/``device`` are None inline, where there is no worker.

        Returns False when the trajectory was unusable and nothing was sent -
        the caller decides what that means for its own pacing.  Inline that
        skips ``min_cycle_s``, exactly as the ``continue`` it replaced did;
        multi-process has no cycle to cut short.
        """
        valid = bool(
            result.positions.ndim == 2
            and result.positions.shape[1] >= 2
            and np.isfinite(result.positions).all()
            and result.positions.shape[0] >= 1
        )
        if not valid:
            # Fail-hold.  An unusable trajectory is not "nothing left to
            # execute": there is no trajectory to release from, and sending
            # control=0 would drop a takeover the engine is holding.
            logger.error("invalid trajectory from model cycle_id=%d", task.cycle_id)
            self.last_takeover_decision = self.takeover_engine.invalid("invalid_ai_trajectory")
            return False

        # Neither model text nor manual flags bypass route safety gates.
        decision = self._trajectory_control_decision(result)
        control = decision.control_flag
        self.last_takeover_decision = decision
        self.last_control_flag = control
        logger.info(
            "takeover decision: control_flag=%d mean_lat_error=%s "
            "max_lat_error=%s takeover_count=%d release_count=%d "
            "reason=%s delay_s=%s start_index=%s remaining_points=%d "
            "reference_source=%s reference_fallback_reason=%s "
            "cycle_id=%d",
            control,
            (
                f"{decision.mean_lateral_error:.3f}m"
                if decision.mean_lateral_error is not None else "n/a"
            ),
            (
                f"{decision.max_lateral_error:.3f}m"
                if decision.max_lateral_error is not None else "n/a"
            ),
            decision.takeover_count,
            decision.release_count,
            decision.decision_reason,
            (
                f"{decision.delay_s:.3f}"
                if decision.delay_s is not None else "n/a"
            ),
            decision.start_index,
            decision.remaining_points,
            decision.reference_source,
            decision.reference_fallback_reason,
            task.cycle_id,
        )
        latest_for_trim = self.history.latest()
        trim = trim_waypoints_to_latest_pose(
            result.positions,
            latest_for_trim,
            result.ref_pose,
            time.monotonic_ns() // 1000,
        )
        if trim.fallback_reason is not None:
            logger.warning(
                "waypoint trim skipped: reason=%s original_waypoint_count=%d "
                "matched_segment_index=n/a projection_distance_m=n/a "
                "removed_waypoint_count=0 remaining_waypoint_count=%d "
                "latest_pose_timestamp=%s trajectory_timestamp=%s "
                "cycle_id=%d",
                trim.fallback_reason,
                len(result.positions),
                len(trim.positions),
                (
                    latest_for_trim.timestamp_us
                    if latest_for_trim is not None else None
                ),
                result.t0_us,
                task.cycle_id,
            )
        else:
            logger.info(
                "waypoint trim: original_waypoint_count=%d "
                "matched_segment_index=%d projection_distance_m=%.3f "
                "removed_waypoint_count=%d remaining_waypoint_count=%d "
                "latest_pose_timestamp=%s trajectory_timestamp=%s "
                "cycle_id=%d",
                len(result.positions),
                trim.matched_segment_index,
                trim.projection_distance_m,
                trim.removed_waypoint_count,
                len(trim.positions),
                latest_for_trim.timestamp_us,
                result.t0_us,
                task.cycle_id,
            )
        # ``crop`` is what the geometric trim did.  A model stop replaces
        # ``trim`` before the empty-release check, and the package keeps
        # ``crop`` so that replacement stays visible.
        crop = trim
        # A model stop's points are where the vehicle was at t0.  After
        # inference they are behind it, so the crop above is empty and the
        # release below would put control=0 on the wire.  Replace the sent
        # trajectory with the current position first; control stays 1.
        if decision.decision_reason == "model_stop_takeover":
            held = hold_waypoint_at_latest_pose(latest_for_trim, result.ref_pose)
            if held is None:
                logger.error(
                    "model stop has no current pose to hold; not sending "
                    "cycle_id=%d",
                    task.cycle_id,
                )
                return False
            trim = TrajectoryPostprocessResult(held, None, None, 0)
            logger.info(
                "model stop hold: control_flag=%d waypoints=%d "
                "latest_pose_timestamp=%s trajectory_timestamp=%s "
                "cycle_id=%d",
                control,
                len(trim.positions),
                (
                    latest_for_trim.timestamp_us
                    if latest_for_trim is not None else None
                ),
                result.t0_us,
                task.cycle_id,
            )
        if len(trim.positions) == 0:
            # Nothing is left to execute, so there is nothing to take
            # over *for*: with waypoint_count=0 the control flag must be
            # 0 as well (confirmed with the vehicle owner 2026-09-17).
            #
            # The packet is still sent.  Continuing without a send - the
            # previous behaviour - left the vehicle holding the last
            # control=1 it received, because "no update" and "not taking
            # over" are indistinguishable on the wire.  A count=0 frame
            # with control=0 is what actually releases it.
            control = 0
            logger.warning(
                "trajectory_expired / no_remaining_waypoints: "
                "original_waypoint_count=%d matched_segment_index=%s "
                "projection_distance_m=%s removed_waypoint_count=%d "
                "remaining_waypoint_count=0 latest_pose_timestamp=%s "
                "trajectory_timestamp=%s; sending count=0 control=0 "
                "cycle_id=%d",
                len(result.positions),
                trim.matched_segment_index,
                trim.projection_distance_m,
                trim.removed_waypoint_count,
                (
                    latest_for_trim.timestamp_us
                    if latest_for_trim is not None else None
                ),
                result.t0_us,
                task.cycle_id,
            )
        lat, lon = local_xy_to_latlon(
            trim.positions[:, 0],
            trim.positions[:, 1],
            task.reference_pose.latitude,
            task.reference_pose.longitude,
            task.reference_pose.heading_deg,
        )
        raw_lat, raw_lon = local_xy_to_latlon(
            result.positions[:, 0],
            result.positions[:, 1],
            task.reference_pose.latitude,
            task.reference_pose.longitude,
            task.reference_pose.heading_deg,
        )
        n = self.sender.send(control, lat, lon)
        self._record_sent_waypoints(
            lat,
            lon,
            n,
            latitude_before_trim=raw_lat,
            longitude_before_trim=raw_lon,
            control_flag=control,
            inference_latency_ms=inference_latency_ms,
            cycle_id=task.cycle_id,
            worker_id=worker_id,
            device=device,
            endpoint=endpoint,
        )
        self.last_result = result
        self.last_sent_control_flag = control
        self.inference_count += 1
        self.last_inference_at = time.monotonic()
        # A zero-waypoint send is a legitimate outcome (release frame),
        # so the endpoint columns are NaN rather than an IndexError.
        # This line is grepped in the field, so its shape must stay
        # stable: every pre-Stage-3 key keeps its name, its order and
        # its position in the string, and ``cycle_id=%d`` is appended
        # after the last of them.
        first_lat, first_lon = (lat[0], lon[0]) if n else (float("nan"),) * 2
        last_lat, last_lon = (lat[-1], lon[-1]) if n else (float("nan"),) * 2
        logger.info(
            "inference sent: control_flag=%d waypoints=%d "
            "waypoint_seq=%d frame_id=%s camera_ts=%s pose_ts=%s "
            "frame_age_ms=%.1f camera_pose_delta_ms=%.1f "
            "inference_latency_ms=%.1f first=(%.7f,%.7f) "
            "last=(%.7f,%.7f) cycle_id=%d",
            control,
            n,
            self.sender.last_seq,
            task.latest_frame_ids,
            result.t0_us,
            task.reference_pose.timestamp_us,
            task.camera_age_ms,
            task.camera_pose_delta_ms,
            inference_latency_ms,
            first_lat,
            first_lon,
            last_lat,
            last_lon,
            task.cycle_id,
        )
        # Structured per-inference record for offline replay/diagnosis.
        # It is saved in infer_packages rather than duplicated in the
        # operational backend log.
        pose_ref = {
            "timestamp_us": int(task.reference_pose.timestamp_us),
            "latitude": float(task.reference_pose.latitude),
            "longitude": float(task.reference_pose.longitude),
            "altitude": float(task.reference_pose.altitude),
            "heading_deg": float(task.reference_pose.heading_deg),
        }
        # Re-read rather than reuse latest_for_trim: this is a snapshot
        # of the vehicle as the record is written.  The pose the crop
        # actually consumed is recorded separately, under
        # trajectory_postprocess_input.
        latest_pose = self.history.latest()
        pose_latest = _pose_payload(latest_pose)
        waypoint_list = [
            [round(float(la), 8), round(float(lo), 8)]
            for la, lo in zip(lat[:n], lon[:n])
        ]
        waypoint_local = [
            [
                round(float(x), 4),
                round(float(y), 4),
                round(float(z), 4),
            ]
            for x, y, z in np.asarray(trim.positions)[:n]
        ]
        # What the model actually produced, BEFORE the crop.  Stored
        # unrounded and at full float64 precision on purpose: the crop
        # decides how much to remove from segment projections, so a
        # rounded copy could crop differently from the live run and the
        # replay would not be a replay.  Together with reference_pose and
        # pose_latest this makes the package sufficient to re-run the
        # crop offline (tools/replay_trajectory_postprocess.py).
        model_waypoint_local = [
            [float(x), float(y), float(z)]
            for x, y, z in np.asarray(result.positions)
        ]
        reference_pose = {
            "timestamp_us": int(task.reference_pose.timestamp_us),
            "latitude": float(task.reference_pose.latitude),
            "longitude": float(task.reference_pose.longitude),
            "altitude": float(task.reference_pose.altitude),
            "heading_deg": float(task.reference_pose.heading_deg),
            "payload_timestamp_us": task.reference_payload_timestamp_us,
        }
        cimg_view = self.cameras.cimg_health_view()
        cimg_peer = cimg_view.get("peer")
        infer_record = {
            "event": "infer_record",
            "timestamp": round(time.time(), 6),
            # Which model produced this trajectory.  Inline leaves these empty.
            # A network slot's endpoint is the PGX that returned the waypoints.
            "worker_id": None if worker_id is None else int(worker_id),
            "device": None if device is None else int(device),
            "endpoint": str(endpoint or ""),
            # Provenance: the CIMG peer is the vehicle's TCP address when the
            # push is live.  A package with pose over UDP and that peer present
            # came from the vehicle; mock pose or no peer is a local run.
            "data_source": {
                "pose_mode": str(self.settings.pose_mode),
                "pose_from_udp": self.settings.pose_mode == "udp",
                "camera_input": "cimg",
                "cimg_peer": cimg_peer,
                "cimg_connected": bool(cimg_view.get("connected")),
                "is_real_vehicle": bool(
                    self.settings.pose_mode == "udp" and cimg_peer
                ),
            },
            # Model + sampling configuration, so the run is reproducible.
            "model": {
                "checkpoint": str(self.settings.checkpoint),
                "temperature": float(self.settings.temperature),
                "top_p": float(self.settings.top_p),
                "num_traj_samples": int(self.settings.num_traj_samples),
                "max_generation_length": int(
                    self.settings.max_generation_length
                ),
                "frames_per_camera": int(self.settings.frames_per_camera),
                "history_steps": int(self.settings.history_steps),
                "history_dt_s": float(self.settings.history_dt_s),
            },
            # Hash of the exact tokenized model input (images included).
            "input_ids_sha256": str(result.input_ids_sha256),
            "input_ids_shape": [int(x) for x in result.input_ids_shape],
            "prompt_text": str(result.prompt_text),
            # Per-frame identity from the CIMG packet sequence.
            "camera_frame_id": task.camera_frame_id,
            "camera_frame_ids": task.camera_frame_ids,
            "camera_t0_us": int(task.window.t0_us),
            "camera_window_timestamps_us": [
                int(x) for x in task.window.timestamps_us
            ],
            # Both clocks are the vehicle's Unix time, in microseconds.
            "pose_timestamp_mode": "unix",
            "camera_clock": "capture_time_ns",
            "pose_payload_timestamp_us": (
                int(latest_pose.payload_timestamp_us)
                if latest_pose is not None
                and latest_pose.payload_timestamp_us is not None
                else None
            ),
            # Local-frame origin of the model output: the newest
            # history sample, interpolated to the camera t0.
            "reference_pose": reference_pose,
            # The 16 ego-history steps the model actually consumed,
            # in both WGS84 and vehicle-frame (+x forward, +y left).
            "ego_history": task.ego_history,
            "heading_source": "vehicle_azimuth",
            "azimuth_offset_deg": float(self.settings.azimuth_offset_deg),
            # Old key kept for existing tooling.
            "pose": pose_ref,
            "pose_latest": pose_latest,
            "heading_deg": float(task.reference_pose.heading_deg),
            # What the vehicle was actually told.  `control_flag` and
            # `takeover_decision_flag` differ in exactly one case: an
            # expired trajectory leaves nothing to execute, so the
            # release frame carries control=0 even though the decision
            # engine may still be holding a takeover.
            "control_flag": int(control),
            "takeover_decision_flag": int(decision.control_flag),
            "takeover_decision": {
                "mean_lateral_error_m": decision.mean_lateral_error,
                "max_lateral_error_m": decision.max_lateral_error,
                "takeover_count": decision.takeover_count,
                "release_count": decision.release_count,
                "decision_reason": decision.decision_reason,
                "delay_s": decision.delay_s,
                "start_index": decision.start_index,
                "remaining_points": decision.remaining_points,
                "reference_source": decision.reference_source,
                "reference_fallback_reason": (
                    decision.reference_fallback_reason
                ),
            },
            # Full model text (chain of thought). Kept verbatim: whether
            # the model saw the obstacle is the first thing to check
            # when reviewing an avoidance failure.
            "cot_text": str(result.cot),
            "waypoint_seq": int(self.sender.last_seq),
            "camera_frame_age_ms": round(task.camera_age_ms, 3),
            "camera_pose_delta_ms": round(task.camera_pose_delta_ms, 3),
            "inference_latency_ms": round(inference_latency_ms, 3),
            "num_waypoints": int(n),
            # SENT output in the vehicle frame, relative to
            # reference_pose.  After the latency crop, so it is shorter
            # than 64 whenever points were removed.  A model stop is the
            # exception: one point at the latest pose, not the crop.
            "output_waypoints_local_xyz": waypoint_local,
            "output_waypoints": waypoint_list,
            # Raw model output before the latency crop, relative to
            # reference_pose, 0.1 s apart.  Full precision, no rounding.
            "model_waypoints_local_xyz": model_waypoint_local,
            # The crop's exact input and output, for offline replay.
            #
            # `latest_pose` here is the pose the crop actually consumed,
            # which is NOT necessarily `pose_latest` above: the crop runs
            # before the send and the record is written after it, so a
            # fast vehicle can produce a newer `pose_latest`.  The
            # trajectory itself is not copied - it is
            # `model_waypoints_local_xyz`, at full precision - so the
            # package holds one copy of the model output, not two.
            "trajectory_postprocess_input": {
                "trajectory_key": "model_waypoints_local_xyz",
                "latest_pose": _pose_payload(latest_for_trim),
                "reference_pose": reference_pose,
                "inference_latency_ms": round(inference_latency_ms, 3),
                "original_waypoint_count": len(result.positions),
            },
            # What trim_waypoints_to_latest_pose did with that input.
            "trajectory_postprocess_result": {
                "output_key": "output_waypoints_local_xyz",
                "applied": bool(crop.applied),
                "expired": bool(crop.expired),
                "fallback_reason": crop.fallback_reason,
                "matched_segment_index": crop.matched_segment_index,
                "projection_distance_m": crop.projection_distance_m,
                "removed_waypoint_count": int(crop.removed_waypoint_count),
                "remaining_waypoint_count": int(
                    crop.remaining_waypoint_count
                ),
                "control_flag_sent": int(control),
            },
            "send_timestamp": round(self.sender.last_send_at, 6),
        }
        package_dir = self._save_infer_package(task.window, infer_record)
        if package_dir is not None:
            logger.info("infer-package-saved %s", package_dir)
        return True

    def _worker_endpoint(self, worker_id: int) -> str:
        """The URL of a network slot, or empty for a local GPU worker."""
        scheduler = self.scheduler
        if scheduler is None:
            return ""
        for view in scheduler.worker_views():
            if int(view.worker_id) == int(worker_id):
                return str(view.endpoint or "")
        return ""

    def _record_sent_waypoints(
        self,
        latitude: Any,
        longitude: Any,
        count: int,
        *,
        latitude_before_trim: Any,
        longitude_before_trim: Any,
        control_flag: int,
        inference_latency_ms: float,
        cycle_id: int,
        worker_id: int | None,
        device: int | None,
        endpoint: str,
    ) -> None:
        """Append the datagram that just went out.  Never affects the send.

        ``waypoints`` is the crop that was sent. ``waypoints_before_trim`` is
        the model trajectory in the same WGS84 frame, before that crop.
        ``host``, ``port`` and ``endpoint`` are the addresses this process was
        configured with.  They are not literals in this function.
        """
        trace = self._vehicle_trace
        if trace is None:
            return
        try:
            trace.append(
                waypoint_record(
                    send_timestamp=self.sender.last_send_at,
                    waypoint_seq=self.sender.last_seq,
                    control_flag=control_flag,
                    host=self.settings.waypoint_host,
                    port=self.settings.waypoint_port,
                    latitude=latitude,
                    longitude=longitude,
                    count=count,
                    latitude_before_trim=latitude_before_trim,
                    longitude_before_trim=longitude_before_trim,
                    inference_latency_ms=inference_latency_ms,
                    cycle_id=cycle_id,
                    worker_id=worker_id,
                    device=device,
                    endpoint=endpoint,
                )
            )
        except Exception as exc:  # noqa: BLE001 - the packet is already on the wire
            logger.warning("failed to record sent waypoints: %r", exc)


__all__ = ["PipelineRunner"]
