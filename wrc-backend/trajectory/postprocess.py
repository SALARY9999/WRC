"""Latency-driven crop of a model trajectory onto the vehicle's position.

This is the *sendable waypoint* layer, and it answers exactly one question:
"does the trajectory we hand the vehicle start where the vehicle now is?".

Inference takes time.  By the time the model's 64 points come back, the
vehicle has moved, so the first few points are already behind it and sending
them unchanged would make the controller drive backwards along them:

    model output   P0 P1 P2 P3 P4 P5
    vehicle is at  P2
    send           P3 P4 P5

The pose used for that projection is the latest packet, moved forward by its
``v_east`` / ``v_north`` over the time since the packet arrived, and only
that long when the wait is under about 0.2 s.  After the polyline crop, a
leading point that is still behind the vehicle's current heading is dropped
too.  That heading is the latest pose's body heading, not the t0 reference
heading.  A point a few centimetres behind the nose stays, and a point that
only falls behind because the path folds back farther along is left alone.

This is deliberately NOT the same thing as ``trajectory.takeover``, which
answers "is the model's prediction wrong?" and only produces a control flag.
The two layers consume different trajectory representations on purpose (the
decision uses the time-compensated trajectory, this uses the raw model output)
and are not required to agree; see the module docstring of ``trajectory.takeover``
and ``tests/test_waypoint_trimming.py``.  ``TrajectoryCompensator.compensate``
is the 0.1 s time crop on that decision path.  It is not applied here.

``trim_waypoints_to_latest_pose`` is the crop that gets sent for a moving
trajectory.

An empty ``positions`` is a normal outcome, not an error: it means the model
output was entirely behind the vehicle.  This module makes no decision about
what to send in that case - the pipeline owns the release-frame policy.

A model stop is the exception.  Its points sit where the vehicle was when
inference started, so the crop above deletes them once the vehicle has moved.
``hold_waypoint_at_latest_pose`` is the point the pipeline sends instead: the
vehicle's position now, so takeover means stop here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from trajectory.coordinates import enu_to_ego
from vehicle.history import poses_to_model_frame_xy

# A pose older than this is not pushed with its stale velocity.  0.2 s is
# the longest the packet's east/north speed is still treated as current.
_MAX_POSE_EXTRAPOLATION_US = 200_000
# A leading point this close to the nose plane is on the vehicle.  Farther
# behind, the controller would turn around to chase it.
_NOSE_BEHIND_TOLERANCE_M = 0.05

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from vehicle.models import GlobalPose, PoseSample


@dataclass(frozen=True)
class TrajectoryPostprocessResult:
    """The raw model trajectory cropped to what the vehicle has not passed.

    ``positions`` is the array to execute and may be empty.  The diagnostic
    fields are what the operator needs to tell "nothing left because the
    trajectory genuinely expired" apart from "the crop silently fell back":

    * ``fallback_reason`` is set ONLY when the crop could not be applied at all
      and ``positions`` is the untouched model output - that is a degraded
      path, not an expired one;
    * ``expired`` is the legitimate empty case, where the crop applied and
      removed everything.
    """

    positions: np.ndarray
    matched_segment_index: int | None
    projection_distance_m: float | None
    removed_waypoint_count: int
    fallback_reason: str | None = None

    @property
    def remaining_waypoint_count(self) -> int:
        return int(len(self.positions))

    @property
    def applied(self) -> bool:
        """True when the crop was computed rather than fallen back to."""
        return self.fallback_reason is None

    @property
    def expired(self) -> bool:
        """True when the crop applied and nothing is left to send."""
        return self.applied and len(self.positions) == 0


# Historical name.  Kept as an alias rather than a subclass so that
# ``isinstance`` and identity checks cannot disagree between the two names.
WaypointTrimResult = TrajectoryPostprocessResult


def _pose_wait_s(latest_pose: "GlobalPose", now_us: int | None) -> float:
    """Seconds of velocity to apply, or 0 when the wait cannot be trusted.

    ``now_us`` and ``arrival_us`` are the same monotonic microsecond clock.
    A missing stamp, a clock that stepped backwards, or a wait past
    ``_MAX_POSE_EXTRAPOLATION_US`` all mean "do not push the old speed".
    """
    if now_us is None or latest_pose.arrival_us is None:
        return 0.0
    wait_us = int(now_us) - int(latest_pose.arrival_us)
    if wait_us <= 0 or wait_us > _MAX_POSE_EXTRAPOLATION_US:
        return 0.0
    return wait_us / 1_000_000.0


def _advance_pose_xy(
    current: np.ndarray,
    latest_pose: "GlobalPose",
    ref_pose: "PoseSample",
    now_us: int | None,
) -> np.ndarray:
    """Move ``current`` by the packet velocity over the receive wait.

    The offset is ``v_east`` / ``v_north`` in the model frame.  It is not
    the body heading, and it is not the 0.1 s trajectory time crop.
    """
    wait_s = _pose_wait_s(latest_pose, now_us)
    if wait_s <= 0.0:
        return current
    v_east = float(latest_pose.v_east)
    v_north = float(latest_pose.v_north)
    if not np.isfinite((v_east, v_north)).all():
        return current
    dx, dy, _ = enu_to_ego(
        v_east * wait_s,
        v_north * wait_s,
        0.0,
        math.radians(ref_pose.heading_deg),
    )
    if not np.isfinite((dx, dy)).all():
        return current
    return current + np.array([dx, dy], dtype=np.float64)


def _leading_points_behind_nose(
    positions: np.ndarray,
    vehicle_xy: np.ndarray,
    latest_pose: "GlobalPose",
    ref_pose: "PoseSample",
) -> int:
    """How many leading points still sit behind the vehicle's nose.

    Only a prefix is removed.  The first point on or in front of the nose
    stops the scan, so a later point where a curve folds back is kept.
    ``latest_pose.heading_deg()`` is the body heading; ``ref_pose`` only
    defines the model frame the waypoints are already in.
    """
    if len(positions) == 0:
        return 0
    heading_deg = float(latest_pose.heading_deg())
    if not np.isfinite(heading_deg):
        return 0
    heading = math.radians(heading_deg)
    forward_x, forward_y, _ = enu_to_ego(
        math.sin(heading),
        math.cos(heading),
        0.0,
        math.radians(ref_pose.heading_deg),
    )
    forward = np.array([forward_x, forward_y], dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if not np.isfinite(norm) or norm < 1e-9:
        return 0
    forward /= norm
    along = (positions[:, :2] - vehicle_xy) @ forward
    behind = along < -_NOSE_BEHIND_TOLERANCE_M
    if not bool(behind[0]):
        return 0
    ahead = np.flatnonzero(~behind)
    if len(ahead) == 0:
        return int(len(behind))
    return int(ahead[0])


def trim_waypoints_to_latest_pose(
    positions: np.ndarray,
    latest_pose: "GlobalPose | None",
    ref_pose: "PoseSample",
    now_us: int | None = None,
) -> TrajectoryPostprocessResult:
    """Crop ``positions`` so the sent trajectory starts ahead of the vehicle.

    ``now_us`` is the current monotonic clock in the same microseconds as
    ``latest_pose.arrival_us``.  The pose is advanced by its east/north
    velocity over that wait before it is projected, and a wait longer than
    about 0.2 s does not use the velocity.  The polyline crop is unchanged.
    Leading points that remain behind the latest pose's heading are then
    dropped.  No pose, or a failed coordinate conversion, returns
    ``positions`` unchanged.
    """
    trajectory = np.asarray(positions, dtype=np.float64)
    if latest_pose is None:
        return TrajectoryPostprocessResult(trajectory, None, None, 0, "latest_pose_unavailable")
    try:
        pose_values = (
            latest_pose.latitude,
            latest_pose.longitude,
            latest_pose.altitude,
            ref_pose.latitude,
            ref_pose.longitude,
            ref_pose.altitude,
            ref_pose.heading_deg,
        )
        if not np.isfinite(pose_values).all():
            raise ValueError("non_finite_pose_coordinates")
        if trajectory.ndim != 2 or trajectory.shape[1] < 2 or len(trajectory) < 2:
            raise ValueError("trajectory_has_no_segments")
        xy = trajectory[:, :2]
        if not np.isfinite(xy).all():
            raise ValueError("non_finite_trajectory_coordinates")
        current_xy, _ = poses_to_model_frame_xy([latest_pose], ref_pose)
        current = np.asarray(current_xy[0], dtype=np.float64)
        if current.shape != (2,) or not np.isfinite(current).all():
            raise ValueError("latest_pose_conversion_non_finite")
        # The packet is already behind the vehicle by the time it is used.
        # Push the projection point by the packet's own velocity, and use
        # that same point as the nose for the leading-point check below.
        current = _advance_pose_xy(current, latest_pose, ref_pose, now_us)

        vectors = np.diff(xy, axis=0)
        length_sq = np.sum(vectors * vectors, axis=1)
        valid_indices = np.flatnonzero(length_sq > 1e-12)
        if not len(valid_indices):
            raise ValueError("trajectory_has_no_nonzero_segments")
        starts = xy[valid_indices]
        valid_vectors = vectors[valid_indices]
        raw_fractions = (
            np.sum((current - starts) * valid_vectors, axis=1)
            / length_sq[valid_indices]
        )
        fractions = np.clip(raw_fractions, 0.0, 1.0)
        projections = starts + fractions[:, None] * valid_vectors
        distances = np.linalg.norm(projections - current, axis=1)
        best = int(np.argmin(distances))
        segment_index = int(valid_indices[best])
        projection_distance = float(distances[best])

        # A projection clamped before the trajectory start means P0 is still
        # ahead; do not discard it. Otherwise P0..Pi are already behind.
        if segment_index == 0 and raw_fractions[best] <= 0.0:
            remove_count = 0
        elif segment_index == len(trajectory) - 2 and raw_fractions[best] >= 1.0:
            # At or beyond the polyline end, even its final endpoint is no
            # longer ahead of the vehicle.
            remove_count = len(trajectory)
        else:
            remove_count = segment_index + 1
        remove_count += _leading_points_behind_nose(
            trajectory[remove_count:], current, latest_pose, ref_pose
        )
        return TrajectoryPostprocessResult(
            trajectory[remove_count:],
            segment_index,
            projection_distance,
            remove_count,
        )
    except (ValueError, TypeError, IndexError, OverflowError, FloatingPointError) as exc:
        return TrajectoryPostprocessResult(trajectory, None, None, 0, str(exc))


def hold_waypoint_at_latest_pose(
    latest_pose: "GlobalPose | None",
    ref_pose: "PoseSample",
) -> np.ndarray | None:
    """One model-frame point at the vehicle's current position.

    Returned as shape ``(1, 3)`` so it can replace a cropped trajectory on the
    send path.  ``None`` means there is no pose to hold: the caller must not
    turn that into an empty release frame.
    """
    if latest_pose is None:
        return None
    try:
        pose_values = (
            latest_pose.latitude,
            latest_pose.longitude,
            latest_pose.altitude,
            ref_pose.latitude,
            ref_pose.longitude,
            ref_pose.altitude,
            ref_pose.heading_deg,
        )
        if not np.isfinite(pose_values).all():
            return None
        current_xy, _ = poses_to_model_frame_xy([latest_pose], ref_pose)
        current = np.asarray(current_xy[0], dtype=np.float64)
        if current.shape != (2,) or not np.isfinite(current).all():
            return None
        point = np.zeros((1, 3), dtype=np.float64)
        point[0, :2] = current
        return point
    except (ValueError, TypeError, IndexError, OverflowError, FloatingPointError):
        return None


__all__ = [
    "TrajectoryPostprocessResult",
    "WaypointTrimResult",
    "hold_waypoint_at_latest_pose",
    "trim_waypoints_to_latest_pose",
]
