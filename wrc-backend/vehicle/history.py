"""Ego pose history and the buffers the model consumes.

``PoseHistory`` is the only store of raw vehicle poses; ``build_ego_arrays``
and ``poses_to_model_frame_xy`` convert slices of it into the frames Alpamayo
expects.  The resampling, the heading unwrap and the local ENU anchoring are
unchanged from ``app.py`` - the model was tuned against exactly this
behaviour.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections import deque

import numpy as np
import torch

from trajectory.coordinates import enu_from_ll, enu_to_ego, yaw_to_matrix
from vehicle.models import GlobalPose, PoseSample


class PoseHistory:
    """Thread-safe buffer of decoded GlobalPose samples."""

    def __init__(self, maxlen: int = 5000) -> None:
        self._lock = threading.Lock()
        self._poses: deque[GlobalPose] = deque(maxlen=maxlen)

    def add(self, pose: GlobalPose) -> None:
        with self._lock:
            if self._poses and pose.timestamp_us <= self._poses[-1].timestamp_us:
                return
            self._poses.append(pose)

    def latest(self) -> GlobalPose | None:
        with self._lock:
            return self._poses[-1] if self._poses else None

    def snapshot(self, limit: int | None = None) -> list[GlobalPose]:
        """Return a stable recent snapshot without exposing the live deque."""
        with self._lock:
            poses = list(self._poses)
        return poses[-limit:] if limit is not None and limit > 0 else poses

    def nearest_payload_timestamp_us(self, t_us: int) -> int | None:
        """Vehicle-side timestamp of the raw sample closest to ``t_us``.

        Measured (not extrapolated) so that the offset between the vehicle
        clock and the clock the pipeline aligns on can be audited per
        inference.
        """
        with self._lock:
            poses = list(self._poses)
        if not poses:
            return None
        times = [p.timestamp_us for p in poses]
        idx = bisect.bisect_left(times, t_us)
        if idx <= 0:
            best = poses[0]
        elif idx >= len(poses):
            best = poses[-1]
        else:
            before, after = poses[idx - 1], poses[idx]
            best = (
                before
                if (t_us - before.timestamp_us) <= (after.timestamp_us - t_us)
                else after
            )
        return best.payload_timestamp_us

    def aligned_samples(
        self, t0_us: int, num_steps: int, dt_s: float
    ) -> list[PoseSample] | None:
        """Resample pose history at [t0-(N-1)*dt ... t0] by linear interpolation."""
        step_us = int(round(dt_s * 1_000_000))
        times = [t0_us - (num_steps - 1 - i) * step_us for i in range(num_steps)]
        with self._lock:
            poses = list(self._poses)
        if len(poses) < 2 or poses[0].timestamp_us > times[0] or poses[-1].timestamp_us < times[-1]:
            return None

        ts = np.array([p.timestamp_us for p in poses], dtype=np.float64)
        lat = np.array([p.latitude for p in poses], dtype=np.float64)
        lon = np.array([p.longitude for p in poses], dtype=np.float64)
        alt = np.array([p.altitude for p in poses], dtype=np.float64)
        # Heading interpolation across 0/360 wraps is intentionally linear here;
        # replace with quaternion interpolation once full attitude is available.
        heading = np.array([p.heading_deg() for p in poses], dtype=np.float64)
        heading = np.unwrap(np.radians(heading))

        out: list[PoseSample] = []
        for t in times:
            lat_i = float(np.interp(t, ts, lat))
            lon_i = float(np.interp(t, ts, lon))
            alt_i = float(np.interp(t, ts, alt))
            h_i = math.degrees(float(np.interp(t, ts, heading))) % 360.0
            out.append(
                PoseSample(
                    timestamp_us=int(t),
                    latitude=lat_i,
                    longitude=lon_i,
                    altitude=alt_i,
                    heading_deg=h_i,
                )
            )
        return out


def build_ego_arrays(
    samples: list[PoseSample],
) -> tuple[torch.Tensor, torch.Tensor, PoseSample]:
    """Convert aligned pose samples into Alpamayo ego history tensors.

    Uses a local ENU tangent plane anchored at the first sample.  Only 2D
    heading is used; roll/pitch are ignored until formal attitude semantics are
    supplied.
    """
    ref = (
        samples[0].latitude,
        samples[0].longitude,
        samples[0].altitude,
    )
    enu = np.array(
        [
            enu_from_ll(s.latitude, s.longitude, s.altitude, ref)
            for s in samples
        ],
        dtype=np.float64,
    )
    t0 = samples[-1]
    e0, n0, u0 = enu_from_ll(
        t0.latitude, t0.longitude, t0.altitude, ref
    )
    yaw0 = math.radians(t0.heading_deg)

    xyz = np.zeros((len(samples), 3), dtype=np.float32)
    rot = np.zeros((len(samples), 3, 3), dtype=np.float32)
    for i, (e, n, u) in enumerate(enu):
        xyz[i] = enu_to_ego(e - e0, n - n0, u - u0, yaw0)
        yaw_i = math.radians(samples[i].heading_deg)
        # Compass heading increases clockwise; mathematical yaw increases
        # counterclockwise. B(h0).T @ B(hi) = Rz(h0 - hi).
        rot[i] = yaw_to_matrix(yaw0 - yaw_i)
    xyz_t = torch.from_numpy(xyz).unsqueeze(0).unsqueeze(0)
    rot_t = torch.from_numpy(rot).unsqueeze(0).unsqueeze(0)
    return xyz_t, rot_t, t0


def poses_to_model_frame_xy(
    poses: list[GlobalPose], ref_pose: PoseSample
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pose positions to the same local frame as model output."""
    yaw = math.radians(ref_pose.heading_deg)
    xy = []
    for pose in poses:
        east, north, up = enu_from_ll(
            pose.latitude,
            pose.longitude,
            pose.altitude,
            (ref_pose.latitude, ref_pose.longitude, ref_pose.altitude),
        )
        x, y, _ = enu_to_ego(east, north, up, yaw)
        xy.append((x, y))
    return (
        np.asarray(xy, dtype=np.float64),
        np.asarray([pose.timestamp_us for pose in poses], dtype=np.int64),
    )


__all__ = ["PoseHistory", "build_ego_arrays", "poses_to_model_frame_xy"]
