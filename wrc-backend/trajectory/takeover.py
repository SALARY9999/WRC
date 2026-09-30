"""Takeover decisions: the primitives, and the per-cycle layer that drives them.

Two layers live here now.  Before the directory closure they lived in two
separate modules whose names differed by a single character - the primitives in
a root ``trajectory_takeover.py``, the per-cycle layer in this file - and both
imported successfully, so a typo picked a module silently instead of failing.
That trap is gone along with the root module: there is one file, one import
path, and no second copy to drift.

The PRIMITIVES - ``ReferenceTrajectory``, ``ModelFrameOrigin``,
``ReferenceTrajectoryRequest``, the reference providers, ``PlannedRoute`` and
its cache, ``TrajectoryCompensator``, ``TrajectoryDeviationCalculator``,
``TakeoverDecision`` and ``TakeoverDecisionEngine`` - are pure: they take
arrays and dataclasses and return values, own no threads, and reach nothing
outside ``trajectory.coordinates`` and numpy.

What ``TakeoverDecisionLayer`` owns
    One inference cycle's control decision: align the pose clock, compensate
    inference latency, sample the planned route, measure mean lateral error,
    then the engine's hysteresis.  ``control_flag`` changes only inside
    ``TakeoverDecisionEngine.update`` and ``force_takeover``.  A model
    trajectory that barely moves while the planned route still runs ahead
    takes over on that cycle.  When the deviation cannot be measured
    the layer returns the current flag and leaves both confirmation counters
    alone.  It does not call ``invalid`` — that reset still belongs to the
    pipeline's own fail-hold paths.

What it does NOT own
    It does not crop the trajectory that gets sent.  That is
    ``trajectory.postprocess``, and the two deliberately consume different
    trajectory representations: the decision measures deviation on the
    time-compensated trajectory, the send path crops the raw model output.  The
    two can therefore disagree within one cycle, and that is intended.

Ordering is load-bearing
    The pose is read before the clock, in the order the original inline code
    used.  That is why the pose arrives as a callable rather than a value:
    fetching it in the caller would silently reorder the reads.  ``now_us`` is
    called only after a pose is in hand, because it advances the pose clock by
    its monotonic arrival age and therefore feeds straight into how much of the
    trajectory the compensator drops.  ``camera_stats_fn`` stays on the
    signature so existing callers do not change; the control decision does not
    read it.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np

from trajectory.coordinates import enu_from_ll, enu_to_ego

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from vehicle.models import GlobalPose, PoseSample

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reference trajectories
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReferenceTrajectory:
    xy: np.ndarray
    timestamps_us: np.ndarray
    source: str


@dataclass(frozen=True)
class ModelFrameOrigin:
    """Pose defining the local model frame; not a wire-protocol structure."""

    latitude: float
    longitude: float
    altitude: float
    heading_deg: float
    timestamp_us: int


@dataclass(frozen=True)
class ReferenceTrajectoryRequest:
    history_xy: np.ndarray
    history_timestamps_us: np.ndarray
    target_timestamps_us: np.ndarray
    model_frame_origin: ModelFrameOrigin | None = None
    current_xy: np.ndarray | None = None
    current_timestamp_us: int | None = None
    speed_mps: float | None = None


class ReferenceTrajectoryProvider(Protocol):
    def generate(self, request: ReferenceTrajectoryRequest) -> ReferenceTrajectory | None: ...


class HistoryPoseReferenceProvider:
    """Constant-trend extrapolation fitted from the latest pose positions."""

    def __init__(self, min_history_points: int = 3) -> None:
        self.min_history_points = max(2, int(min_history_points))

    def generate(self, request: ReferenceTrajectoryRequest) -> ReferenceTrajectory | None:
        xy = np.asarray(request.history_xy, dtype=np.float64)
        ts = np.asarray(request.history_timestamps_us, dtype=np.float64)
        targets = np.asarray(request.target_timestamps_us, dtype=np.int64)
        if (
            xy.ndim != 2
            or xy.shape[1] != 2
            or len(xy) < self.min_history_points
            or len(ts) != len(xy)
            or targets.ndim != 1
            or len(targets) == 0
            or not np.isfinite(xy).all()
            or not np.isfinite(ts).all()
            or np.any(np.diff(ts) <= 0)
        ):
            return None

        # Centre time near zero for a numerically stable least-squares fit.
        t0 = ts[-1]
        t_s = (ts - t0) / 1_000_000.0
        design = np.column_stack((t_s, np.ones_like(t_s)))
        try:
            coeff, _, _, _ = np.linalg.lstsq(design, xy, rcond=None)
        except np.linalg.LinAlgError:
            return None
        target_s = (targets.astype(np.float64) - t0) / 1_000_000.0
        reference_xy = target_s[:, None] * coeff[0] + coeff[1]
        if not np.isfinite(reference_xy).all():
            return None
        return ReferenceTrajectory(reference_xy, targets, "history_pose")


# --------------------------------------------------------------------------- #
# Planned route: the protocol-neutral data model and its cache
# --------------------------------------------------------------------------- #
#
# These three are defined HERE rather than in ``vehicle.planned_route``, and
# that direction is deliberate: ``vehicle.planned_route`` decodes the wire into
# them and imports them from this module, so the dependency runs
# ``vehicle.planned_route -> trajectory.takeover`` and never back.  Defining
# them the other way round would put the wire decoder underneath the decision
# engine, and the decision engine is what everything else is allowed to depend
# on - not the socket.

@dataclass(frozen=True)
class PlannedRoutePoint:
    """Protocol-neutral point produced by a future route adapter."""

    coordinates: tuple[float, ...]
    timestamp_us: int | None = None
    speed_mps: float | None = None
    heading_deg: float | None = None


@dataclass(frozen=True)
class PlannedRoute:
    """Normalized adapter output; coordinate_frame semantics come from protocol."""

    points: tuple[PlannedRoutePoint, ...]
    coordinate_frame: str
    route_id: str | None = None
    source_timestamp_us: int | None = None


class PlannedRouteAdapter(Protocol):
    """Future network-specific decoders implement this boundary."""

    def decode(self, payload: bytes) -> PlannedRoute: ...


class PlannedRouteCache:
    """Atomic latest-route storage shared by receiver and inference threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._route: PlannedRoute | None = None
        self._version = 0
        self.updated_monotonic = 0.0

    def update(self, route: PlannedRoute) -> int:
        with self._lock:
            self._route = route
            self._version += 1
            self.updated_monotonic = time.monotonic()
            return self._version

    def snapshot(self) -> tuple[PlannedRoute | None, int]:
        with self._lock:
            return self._route, self._version

    def snapshot_with_age(self) -> tuple[PlannedRoute | None, int, float]:
        with self._lock:
            return self._route, self._version, time.monotonic() - self.updated_monotonic

    def clear(self) -> None:
        with self._lock:
            self._route = None
            self._version += 1


class PlannedRouteReferenceProvider:
    """Match current position and sample the ordered route at current speed.

    A route stays in force after it is received. Age is not a reason to drop
    it; a later update replaces it, and an explicit clear removes it.
    """

    def __init__(self, cache: PlannedRouteCache, max_age_s: float = 30.0,
                 max_match_distance_m: float = 10.0) -> None:
        self.cache = cache
        if not np.isfinite([max_age_s, max_match_distance_m]).all() or min(max_age_s, max_match_distance_m) <= 0:
            raise ValueError("route age and matching distance must be finite and positive")
        self.max_age_s = max_age_s
        self.max_match_distance_m = max_match_distance_m
        self.last_unavailable_reason = "planned_route_not_received"

    def update_route(self, route: PlannedRoute) -> int:
        return self.cache.update(route)

    def generate(self, request: ReferenceTrajectoryRequest) -> ReferenceTrajectory | None:
        route, _version = self.cache.snapshot()
        if route is None:
            self.last_unavailable_reason = "planned_route_not_received"
            return None
        try:
            if route.coordinate_frame != "geographic_lon_lat_degrees":
                raise ValueError("planned_route_unsupported_coordinates")
            origin = request.model_frame_origin
            if origin is None or request.current_xy is None or request.speed_mps is None:
                raise ValueError("planned_route_pose_unavailable")
            ll = np.asarray([p.coordinates for p in route.points], dtype=float)
            current = np.asarray(request.current_xy, dtype=float)
            targets = np.asarray(request.target_timestamps_us, dtype=np.int64)
            if (ll.ndim != 2 or ll.shape[1] != 2 or len(ll) < 2
                    or not np.isfinite(ll).all() or current.shape != (2,)
                    or not np.isfinite(current).all()
                    or not np.isfinite([origin.latitude, origin.longitude, origin.heading_deg,
                                           request.speed_mps]).all()
                    or np.any(np.abs(ll[:, 0]) > 180) or np.any(np.abs(ll[:, 1]) > 90)
                    or targets.ndim != 1 or len(targets) < 2
                    or np.any(np.diff(targets) <= 0) or request.current_timestamp_us is None
                    or request.speed_mps < 0):
                raise ValueError("planned_route_invalid_data")
            # Identical short-range ENU constants and compass rotation to app.py.
            east = (ll[:, 0] - origin.longitude) * 111320.0 * math.cos(math.radians(origin.latitude))
            north = (ll[:, 1] - origin.latitude) * 110540.0
            h = math.radians(origin.heading_deg)
            xy = np.column_stack((east * math.sin(h) + north * math.cos(h),
                                  -east * math.cos(h) + north * math.sin(h)))
            keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
            xy = xy[keep]
            if len(xy) < 2:
                raise ValueError("planned_route_degenerate")
            vec = np.diff(xy, axis=0)
            lengths = np.linalg.norm(vec, axis=1)
            arc = np.r_[0.0, np.cumsum(lengths)]
            fractions = np.clip(np.sum((current - xy[:-1]) * vec, axis=1) / lengths**2, 0, 1)
            projections = xy[:-1] + fractions[:, None] * vec
            distances = np.linalg.norm(projections - current, axis=1)
            match = int(np.argmin(distances))
            if distances[match] > self.max_match_distance_m:
                raise ValueError("planned_route_vehicle_too_far")
            start = arc[match] + fractions[match] * lengths[match]
            offsets = (targets - request.current_timestamp_us) / 1e6
            sample_arc = start + np.maximum(offsets, 0) * request.speed_mps
            if sample_arc[-1] > arc[-1] + 1e-6 or start >= arc[-1] - 1e-6:
                raise ValueError("planned_route_insufficient_future_length")
            if request.speed_mps < 1e-3:
                raise ValueError("planned_route_stationary_reference")
            reference = np.column_stack([np.interp(sample_arc, arc, xy[:, i]) for i in range(2)])
            if not np.isfinite(reference).all():
                raise ValueError("planned_route_coordinate_conversion_failed")
            self.last_unavailable_reason = ""
            return ReferenceTrajectory(reference, targets, "planned_route")
        except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
            self.last_unavailable_reason = str(exc)
            return None


@dataclass(frozen=True)
class ReferenceTrajectorySelection:
    trajectory: ReferenceTrajectory | None
    fallback_reason: str | None = None


class ReferenceTrajectorySelector:
    """Planned route only; no history fallback or configurable source."""

    def __init__(
        self,
        planned_provider: PlannedRouteReferenceProvider,
    ) -> None:
        self.planned_provider = planned_provider
        self.source_mode = "planned_route"

    def generate(self, request: ReferenceTrajectoryRequest) -> ReferenceTrajectorySelection:
        planned = self.planned_provider.generate(request)
        return ReferenceTrajectorySelection(planned, None if planned is not None else
                                            self.planned_provider.last_unavailable_reason)


# --------------------------------------------------------------------------- #
# Latency compensation, deviation, hysteresis
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CompensatedTrajectory:
    xy: np.ndarray
    original_indices: np.ndarray
    timestamps_us: np.ndarray
    delay_s: float
    start_index: int


class TrajectoryCompensator:
    def __init__(self, trajectory_dt_s: float, min_remaining_points: int) -> None:
        if trajectory_dt_s <= 0:
            raise ValueError("trajectory_dt_s must be positive")
        self.trajectory_dt_s = float(trajectory_dt_s)
        if int(round(self.trajectory_dt_s * 1_000_000.0)) <= 0:
            raise ValueError("trajectory_dt_s is below timestamp resolution")
        self.min_remaining_points = max(1, int(min_remaining_points))

    def compensate(
        self,
        ai_trajectory: np.ndarray,
        model_input_timestamp_us: int,
        current_timestamp_us: int,
    ) -> tuple[CompensatedTrajectory | None, str]:
        trajectory = np.asarray(ai_trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2 or len(trajectory) == 0:
            return None, "invalid_ai_trajectory_shape"
        if not np.isfinite(trajectory[:, :2]).all():
            return None, "non_finite_ai_trajectory"
        delay_us = int(current_timestamp_us) - int(model_input_timestamp_us)
        delay_s = delay_us / 1_000_000.0
        if not np.isfinite(delay_s) or delay_s < 0:
            return None, "invalid_model_input_timestamp"
        # Use integer microseconds to avoid 2.3 / 0.1 becoming 22.999999...
        # and retaining an already-expired point.
        dt_us = int(round(self.trajectory_dt_s * 1_000_000.0))
        start_index = delay_us // dt_us
        if start_index >= len(trajectory):
            return None, "trajectory_fully_expired"
        indices = np.arange(start_index, len(trajectory), dtype=np.int64)
        if len(indices) < self.min_remaining_points:
            return None, "insufficient_remaining_trajectory"
        timestamps = int(model_input_timestamp_us) + np.rint(
            indices * self.trajectory_dt_s * 1_000_000.0
        ).astype(np.int64)
        return (
            CompensatedTrajectory(
                xy=trajectory[indices, :2],
                original_indices=indices,
                timestamps_us=timestamps,
                delay_s=delay_s,
                start_index=start_index,
            ),
            "ok",
        )


@dataclass(frozen=True)
class DeviationMetrics:
    mean_lateral_error: float
    max_lateral_error: float


class TrajectoryDeviationCalculator:
    def calculate(
        self, ai_xy: np.ndarray, reference_xy: np.ndarray
    ) -> DeviationMetrics | None:
        ai = np.asarray(ai_xy, dtype=np.float64)
        ref = np.asarray(reference_xy, dtype=np.float64)
        if ai.shape != ref.shape or ai.ndim != 2 or ai.shape[1] != 2 or len(ai) < 2:
            return None
        if not np.isfinite(ai).all() or not np.isfinite(ref).all():
            return None
        tangent = np.gradient(ref, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        valid = norm > 1e-6
        if not np.any(valid):
            # The common frame is vehicle-aligned, so +y remains lateral when
            # a stationary reference has no tangent direction of its own.
            lateral = np.abs((ai - ref)[:, 1])
        else:
            tangent[~valid] = tangent[np.flatnonzero(valid)[0]]
            norm[~valid] = np.linalg.norm(tangent[~valid], axis=1)
            delta = ai - ref
            lateral = np.abs(tangent[:, 0] * delta[:, 1] - tangent[:, 1] * delta[:, 0]) / norm
        return DeviationMetrics(float(np.mean(lateral)), float(np.max(lateral)))


@dataclass(frozen=True)
class TakeoverDecision:
    control_flag: int
    mean_lateral_error: float | None
    max_lateral_error: float | None
    takeover_count: int
    release_count: int
    decision_reason: str
    delay_s: float | None = None
    start_index: int | None = None
    remaining_points: int = 0
    reference_source: str | None = None
    reference_fallback_reason: str | None = None


class TakeoverDecisionEngine:
    """Two-counter hysteresis; invalid input can never start a takeover."""

    def __init__(
        self,
        takeover_threshold_m: float,
        release_threshold_m: float,
        takeover_confirmations: int = 2,
        release_confirmations: int = 2,
    ) -> None:
        if (not np.isfinite([release_threshold_m, takeover_threshold_m]).all()
                or release_threshold_m < 0 or release_threshold_m >= takeover_threshold_m):
            raise ValueError("release_threshold_m must be less than takeover_threshold_m")
        self.takeover_threshold_m = float(takeover_threshold_m)
        self.release_threshold_m = float(release_threshold_m)
        self.takeover_confirmations = max(1, int(takeover_confirmations))
        self.release_confirmations = max(1, int(release_confirmations))
        self.control_flag = 0
        self.takeover_count = 0
        self.release_count = 0

    def invalid(self, reason: str, **details: object) -> TakeoverDecision:
        self.takeover_count = 0
        self.release_count = 0
        return TakeoverDecision(
            self.control_flag, None, None, self.takeover_count, self.release_count,
            reason, **details,
        )

    def update(
        self,
        metrics: DeviationMetrics,
        **details: object,
    ) -> TakeoverDecision:
        mean = metrics.mean_lateral_error
        reason = "within_hysteresis_band"
        if self.control_flag == 0:
            self.release_count = 0
            if mean > self.takeover_threshold_m:
                self.takeover_count += 1
                reason = "mean_lateral_error_above_takeover_threshold"
                if self.takeover_count >= self.takeover_confirmations:
                    self.control_flag = 1
                    self.takeover_count = 0
                    reason = "takeover_confirmed"
            else:
                self.takeover_count = 0
                reason = "mean_lateral_error_below_takeover_threshold"
        else:
            self.takeover_count = 0
            if mean < self.release_threshold_m:
                self.release_count += 1
                reason = "mean_lateral_error_below_release_threshold"
                if self.release_count >= self.release_confirmations:
                    self.control_flag = 0
                    self.release_count = 0
                    reason = "release_confirmed"
            else:
                self.release_count = 0
                reason = "takeover_held"
        return TakeoverDecision(
            self.control_flag,
            metrics.mean_lateral_error,
            metrics.max_lateral_error,
            self.takeover_count,
            self.release_count,
            reason,
            **details,
        )

    def force_takeover(
        self,
        reason: str,
        metrics: DeviationMetrics | None = None,
        **details: object,
    ) -> TakeoverDecision:
        """Take over on this cycle. Confirmation counters do not apply."""
        self.control_flag = 1
        self.takeover_count = 0
        self.release_count = 0
        return TakeoverDecision(
            self.control_flag,
            None if metrics is None else metrics.mean_lateral_error,
            None if metrics is None else metrics.max_lateral_error,
            self.takeover_count,
            self.release_count,
            reason,
            **details,
        )


# --------------------------------------------------------------------------- #
# The per-cycle layer
# --------------------------------------------------------------------------- #

class TakeoverDecisionLayer:
    """Produces one cycle's control decision from route deviation."""

    def __init__(
        self,
        *,
        takeover_engine: "TakeoverDecisionEngine",
        trajectory_compensator: "TrajectoryCompensator",
        deviation_calculator: "TrajectoryDeviationCalculator",
        reference_provider: "ReferenceTrajectorySelector",
        planned_route_cache: "PlannedRouteCache",
        expected_camera_count: int,
        camera_stale_ms: float,
        pose_stale_ms: float,
        stop_progress_m: float = 1.0,
        now_us_fn: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.takeover_engine = takeover_engine
        self.trajectory_compensator = trajectory_compensator
        self.deviation_calculator = deviation_calculator
        self.reference_provider = reference_provider
        self.planned_route_cache = planned_route_cache
        # These three come from settings and are fixed for the process
        # lifetime, the same lifetime the camera provider is built for.
        self.expected_camera_count = int(expected_camera_count)
        self.camera_stale_ms = camera_stale_ms
        self.pose_stale_ms = pose_stale_ms
        if not np.isfinite(stop_progress_m) or stop_progress_m <= 0:
            raise ValueError("stop_progress_m must be finite and positive")
        self.stop_progress_m = float(stop_progress_m)
        # Injectable so a test can pin the clock; monotonic_ns is in ns.
        self._now_us_fn = now_us_fn

    @staticmethod
    def _span_m(xy: np.ndarray) -> float:
        """Largest distance from the first point, in metres."""
        points = np.asarray(xy, dtype=np.float64)
        if len(points) == 0:
            return 0.0
        return float(np.max(np.linalg.norm(points - points[0], axis=1)))

    def _unchanged(self, reason: str, **details: object) -> TakeoverDecision:
        """Current flag and confirmation counters, with no engine transition."""
        engine = self.takeover_engine
        return TakeoverDecision(
            engine.control_flag,
            None,
            None,
            engine.takeover_count,
            engine.release_count,
            reason,
            **details,
        )

    def decide(
        self,
        *,
        positions: np.ndarray,
        t0_us: int,
        ref_pose: "PoseSample",
        camera_stats_fn: Callable[[], list[dict]],
        latest_pose_fn: Callable[[], "GlobalPose | None"],
    ) -> TakeoverDecision:
        engine = self.takeover_engine
        # Kept on the signature so callers stay stable. Camera health is not
        # an input to the control flag.
        del camera_stats_fn

        latest = latest_pose_fn()
        if latest is None:
            return self._unchanged("pose_unavailable")

        now_us = self._now_us_fn() // 1000
        pose_arrival_us = latest.arrival_us
        if pose_arrival_us is None:
            return self._unchanged("pose_arrival_timestamp_unavailable")
        if int(pose_arrival_us) > now_us:
            return self._unchanged("pose_arrival_timestamp_in_future")
        if (now_us - int(pose_arrival_us)) / 1000.0 > self.pose_stale_ms:
            return self._unchanged("pose_stale")

        # Advance the pose clock by its monotonic arrival age: inference latency
        # must not be underestimated just because the latest pose preceded completion.
        current_timestamp_us = latest.timestamp_us + max(0, now_us - int(pose_arrival_us))
        compensated, reason = self.trajectory_compensator.compensate(
            positions,
            t0_us,
            current_timestamp_us,
        )
        if compensated is None:
            return self._unchanged(reason)

        try:
            east, north, up = enu_from_ll(latest.latitude, latest.longitude, latest.altitude,
                (ref_pose.latitude, ref_pose.longitude, ref_pose.altitude))
            current_xy = np.asarray(enu_to_ego(east, north, up,
                math.radians(ref_pose.heading_deg))[:2])
        except (ValueError, OverflowError, FloatingPointError) as exc:
            logger.warning("takeover reference coordinate conversion failed: %s", exc)
            return self._unchanged(
                "reference_coordinate_conversion_failed",
                delay_s=compensated.delay_s,
                start_index=compensated.start_index,
                remaining_points=len(compensated.xy),
            )
        selection = self.reference_provider.generate(
            ReferenceTrajectoryRequest(
                history_xy=np.empty((0, 2)),
                history_timestamps_us=np.empty(0, dtype=np.int64),
                current_xy=current_xy,
                current_timestamp_us=current_timestamp_us,
                speed_mps=latest.speed_mps,
                target_timestamps_us=compensated.timestamps_us,
                model_frame_origin=ModelFrameOrigin(
                    latitude=ref_pose.latitude,
                    longitude=ref_pose.longitude,
                    altitude=ref_pose.altitude,
                    heading_deg=ref_pose.heading_deg,
                    timestamp_us=ref_pose.timestamp_us,
                ),
            )
        )
        reference = selection.trajectory
        if reference is None:
            return self._unchanged(
                "reference_trajectory_unavailable",
                delay_s=compensated.delay_s,
                start_index=compensated.start_index,
                remaining_points=len(compensated.xy),
                reference_fallback_reason=selection.fallback_reason,
            )
        metrics = self.deviation_calculator.calculate(compensated.xy, reference.xy)
        if metrics is None:
            return self._unchanged(
                "trajectory_deviation_calculation_failed",
                delay_s=compensated.delay_s,
                start_index=compensated.start_index,
                remaining_points=len(compensated.xy),
                reference_source=reference.source,
                reference_fallback_reason=selection.fallback_reason,
            )
        details = dict(
            delay_s=compensated.delay_s,
            start_index=compensated.start_index,
            remaining_points=len(compensated.xy),
            reference_source=reference.source,
            reference_fallback_reason=selection.fallback_reason,
        )
        # The model has stopped while the route still runs ahead. One cycle
        # takes over; a later moving trajectory returns to lateral hysteresis.
        if (
            self._span_m(compensated.xy) < self.stop_progress_m
            and self._span_m(reference.xy) >= self.stop_progress_m
        ):
            return engine.force_takeover("model_stop_takeover", metrics, **details)
        return engine.update(metrics, **details)


__all__ = [
    "CompensatedTrajectory",
    "DeviationMetrics",
    "HistoryPoseReferenceProvider",
    "ModelFrameOrigin",
    "PlannedRoute",
    "PlannedRouteAdapter",
    "PlannedRouteCache",
    "PlannedRoutePoint",
    "PlannedRouteReferenceProvider",
    "ReferenceTrajectory",
    "ReferenceTrajectoryProvider",
    "ReferenceTrajectoryRequest",
    "ReferenceTrajectorySelection",
    "ReferenceTrajectorySelector",
    "TakeoverDecision",
    "TakeoverDecisionEngine",
    "TakeoverDecisionLayer",
    "TrajectoryCompensator",
    "TrajectoryDeviationCalculator",
]
