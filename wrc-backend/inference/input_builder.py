"""Assemble one model input from a camera window and the pose history.

What this module owns:

    CameraWindow + PoseHistory
        -> freshness gates (camera age, pose age, camera-pose delta, camera lag)
        -> 16 interpolated ego samples (PoseHistory.aligned_samples)
        -> ego_history_xyz / ego_history_rot tensors + the reference pose
        -> diagnostics: the 16 ego rows and the per-camera frame ids

What this module deliberately does not do:

* it does not load or call the model - that is ``inference.alpamayo``;
* it does not decide a takeover - that is ``trajectory.takeover``;
* it does not read the planned route, crop a trajectory or touch the waypoint
  sender.  It is importable without torch's CUDA stack and without the vehicle
  link, which is what makes the gates below testable.

The gate order is load-bearing
    Metrics are computed first, in one pass, because the rejection *logs* report
    them: ``camera_age``/``pose_age``/``delta`` are needed even when the cycle
    is rejected at the first check.  The gates themselves then run in the
    historical order - freshness, then camera lag, then pose history - because
    the rejection reason reaches ``TakeoverDecisionEngine.invalid()`` and a
    reordering would change which reason an operator sees in the field.

Rejection reasons are a compatibility surface
    ``camera stale``, ``pose stale`` and ``camera-pose delta`` are spelled with
    SPACES, and they are NOT the decision layer's ``camera_stale`` /
    ``pose_stale`` / ``camera_window_stale``.  Both spellings are already in
    the logs and in recorded inference packages; do not unify them.

A rejection is not an exception
    The builder never raises for a bad input; it returns a result whose
    ``rejection_reason`` is set and whose diagnostics are still populated, so
    the caller can log the same line it always did and then decide to sleep and
    retry.  That keeps the loop's control flow (and its ``takeover_engine``
    call) where it was.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from inference.camera_labels import _CAMERA_DISPLAY_NAMES, _camera_package_label
from trajectory.coordinates import enu_from_ll, enu_to_ego
from vehicle.history import build_ego_arrays

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import torch

    from camera.models import CameraWindow
    from vehicle.history import PoseHistory
    from vehicle.models import GlobalPose, PoseSample

# The first three are the historical space-separated spellings: they are handed
# straight to ``TakeoverDecisionEngine.invalid()`` and appear in the operator
# log, so they are part of the observable surface.  See the module docstring.
REASON_CAMERA_STALE = "camera stale"
REASON_POSE_STALE = "pose stale"
REASON_CAMERA_POSE_DELTA = "camera-pose delta"
REASON_CAMERA_WINDOW_STALE = "camera_window_stale"
REASON_POSE_HISTORY_UNAVAILABLE = "model_pose_history_unavailable"

#: The freshness spellings, as one set: the caller logs them with a single
#: shared throttle, exactly as the inline loop did.
FRESHNESS_REASONS = frozenset(
    {REASON_CAMERA_STALE, REASON_POSE_STALE, REASON_CAMERA_POSE_DELTA}
)


@dataclass(frozen=True)
class InputBuildResult:
    """One attempt at building a model input.

    On rejection only the diagnostic fields are meaningful; on success every
    field is populated.  ``rejection_reason`` is a plain string rather than an
    exception so that a rejected cycle stays ordinary control flow.
    """

    rejection_reason: str | None = None

    # --- always populated ------------------------------------------------- #
    latest_pose: "GlobalPose | None" = None
    now_us: int = 0
    camera_arrival_us: int | None = None
    pose_arrival_us: int | None = None
    camera_age_ms: float = float("inf")
    pose_age_ms: float = float("inf")
    camera_pose_delta_ms: float = float("inf")
    camera_lag_ms: float | None = None
    #: Newest frame id per camera, in window order - for the logs.
    latest_frame_ids: list[str | None] = field(default_factory=list)

    # --- populated only when accepted ------------------------------------- #
    ego_xyz: "torch.Tensor | None" = None
    ego_rot: "torch.Tensor | None" = None
    reference_pose: "PoseSample | None" = None
    reference_payload_timestamp_us: int | None = None
    #: The 16 history steps the model consumed, in WGS84 and vehicle frame.
    ego_history: list[dict[str, Any]] = field(default_factory=list)
    #: Per-camera frame identity rows, and the same data keyed by label.
    camera_frame_ids: list[dict[str, Any]] = field(default_factory=list)
    camera_frame_id: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None


class InputBuilder:
    """Turns (camera window, pose history) into the model call's arguments."""

    def __init__(
        self,
        *,
        history_steps: int,
        history_dt_s: float,
        camera_stale_ms: float,
        pose_stale_ms: float,
        camera_pose_delta_ms: float,
        now_us_fn: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.history_steps = int(history_steps)
        self.history_dt_s = float(history_dt_s)
        self.camera_stale_ms = camera_stale_ms
        self.pose_stale_ms = pose_stale_ms
        self.camera_pose_delta_ms = camera_pose_delta_ms
        # Nanoseconds, matching TakeoverDecisionLayer: the microsecond
        # conversion happens in build() so a test can pin the clock.
        self._now_us_fn = now_us_fn

    def build(
        self, window: "CameraWindow", history: "PoseHistory"
    ) -> InputBuildResult:
        """Build the model input, or explain why this cycle was rejected."""
        # Read order matches the inline loop: the pose, then the clock.
        latest_pose = history.latest()
        now_us = self._now_us_fn() // 1000

        # Every camera's newest frame arrival, on the server monotonic clock.
        # Partial coverage is treated as no coverage: a window missing one
        # camera's arrival cannot be aged by its stalest leg.
        camera_arrivals = [
            int(row[-1])
            for row in (window.frame_arrivals_us or [])
            if row and row[-1] is not None
        ]
        camera_arrival_us = (
            min(camera_arrivals)
            if len(camera_arrivals) == len(window.camera_indices)
            else None
        )
        camera_age_ms = (
            (now_us - camera_arrival_us) / 1000.0
            if camera_arrival_us is not None
            else float("inf")
        )
        pose_arrival_us = (
            int(latest_pose.arrival_us)
            if latest_pose is not None and latest_pose.arrival_us is not None
            else None
        )
        pose_age_ms = (
            (now_us - pose_arrival_us) / 1000.0
            if pose_arrival_us is not None
            else float("inf")
        )
        camera_pose_delta_ms = (
            abs(camera_arrival_us - pose_arrival_us) / 1000.0
            if camera_arrival_us is not None and pose_arrival_us is not None
            else float("inf")
        )
        latest_frame_ids = [
            row[-1] if row else None for row in (window.frame_ids or [])
        ]

        def rejected(reason: str, **extra: Any) -> InputBuildResult:
            return InputBuildResult(
                rejection_reason=reason,
                latest_pose=latest_pose,
                now_us=now_us,
                camera_arrival_us=camera_arrival_us,
                pose_arrival_us=pose_arrival_us,
                camera_age_ms=camera_age_ms,
                pose_age_ms=pose_age_ms,
                camera_pose_delta_ms=camera_pose_delta_ms,
                latest_frame_ids=latest_frame_ids,
                **extra,
            )

        if camera_age_ms > self.camera_stale_ms:
            return rejected(REASON_CAMERA_STALE)
        if pose_age_ms > self.pose_stale_ms:
            return rejected(REASON_POSE_STALE)
        if camera_pose_delta_ms > self.camera_pose_delta_ms:
            return rejected(REASON_CAMERA_POSE_DELTA)

        camera_lag_ms: float | None = None
        if latest_pose is not None:
            # Never run the model on an old camera window when the vehicle pose
            # has already moved on. Compared against the *camera* staleness
            # budget, as it always has been - not the pose budget.
            camera_lag_ms = (latest_pose.timestamp_us - window.t0_us) / 1000.0
            if camera_lag_ms > self.camera_stale_ms:
                return rejected(
                    REASON_CAMERA_WINDOW_STALE, camera_lag_ms=camera_lag_ms
                )

        poses = history.aligned_samples(
            window.t0_us, self.history_steps, self.history_dt_s
        )
        if poses is None:
            return rejected(
                REASON_POSE_HISTORY_UNAVAILABLE, camera_lag_ms=camera_lag_ms
            )
        ego_xyz, ego_rot, reference_pose = build_ego_arrays(poses)
        # Derived once, then summarised: the label-keyed dict must describe
        # exactly the rows recorded beside it.
        frame_rows = self._camera_frame_rows(window)

        return InputBuildResult(
            latest_pose=latest_pose,
            now_us=now_us,
            camera_arrival_us=camera_arrival_us,
            pose_arrival_us=pose_arrival_us,
            camera_age_ms=camera_age_ms,
            pose_age_ms=pose_age_ms,
            camera_pose_delta_ms=camera_pose_delta_ms,
            camera_lag_ms=camera_lag_ms,
            latest_frame_ids=latest_frame_ids,
            ego_xyz=ego_xyz,
            ego_rot=ego_rot,
            reference_pose=reference_pose,
            reference_payload_timestamp_us=history.nearest_payload_timestamp_us(
                int(reference_pose.timestamp_us)
            ),
            ego_history=self._ego_history_rows(poses, reference_pose),
            camera_frame_ids=frame_rows,
            camera_frame_id={
                row["camera_label"]: row["latest_frame_id"]
                for row in frame_rows
            },
        )

    # --------------------------------------------------------------------- #
    # Diagnostics.  These are what a later reviewer reads out of the package
    # to answer "what did the model actually see this cycle?".
    # --------------------------------------------------------------------- #

    @staticmethod
    def _ego_history_rows(poses, reference_pose) -> list[dict[str, Any]]:
        """The 16 consumed steps, in both WGS84 and the vehicle frame."""
        ref_ll = (
            reference_pose.latitude,
            reference_pose.longitude,
            reference_pose.altitude,
        )
        yaw = math.radians(reference_pose.heading_deg)
        rows: list[dict[str, Any]] = []
        for sample in poses:
            east_m, north_m, up_m = enu_from_ll(
                sample.latitude,
                sample.longitude,
                sample.altitude,
                ref_ll,
            )
            x_m, y_m, _z_m = enu_to_ego(east_m, north_m, up_m, yaw)
            rows.append(
                {
                    "timestamp_us": int(sample.timestamp_us),
                    "latitude": round(float(sample.latitude), 8),
                    "longitude": round(float(sample.longitude), 8),
                    "altitude": round(float(sample.altitude), 3),
                    "heading_deg": round(float(sample.heading_deg), 3),
                    "x_m": round(float(x_m), 4),
                    "y_m": round(float(y_m), 4),
                }
            )
        return rows

    @staticmethod
    def _camera_frame_rows(window: "CameraWindow") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cam_pos, cam_idx in enumerate(window.camera_indices):
            ids = (
                list(window.frame_ids[cam_pos])
                if getattr(window, "frame_ids", None)
                and cam_pos < len(window.frame_ids)
                else []
            )
            rows.append(
                {
                    "camera_index": int(cam_idx),
                    "camera_label": _camera_package_label(int(cam_idx)),
                    "display": _CAMERA_DISPLAY_NAMES.get(
                        int(cam_idx), str(cam_idx)
                    ),
                    "frame_ids": ids,
                    "latest_frame_id": ids[-1] if ids else None,
                }
            )
        return rows

__all__ = [
    "FRESHNESS_REASONS",
    "InputBuildResult",
    "InputBuilder",
    "REASON_CAMERA_POSE_DELTA",
    "REASON_CAMERA_STALE",
    "REASON_CAMERA_WINDOW_STALE",
    "REASON_POSE_HISTORY_UNAVAILABLE",
    "REASON_POSE_STALE",
]
