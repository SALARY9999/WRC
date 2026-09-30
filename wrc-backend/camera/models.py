"""Camera data models shared by the CIMG receiver and the synchronization layer.

The two classes below are pure data: they hold no behaviour and, deliberately,
this module imports nothing at runtime (``numpy``/``torch`` appear only inside
string annotations).  That keeps ``camera.cimg`` and ``camera.cimg_source``
free of the heavy imports.

Field names, units and semantics are a compatibility surface: ``FrameEntry``
and ``CameraWindow`` are constructed by offline tools (``tools/*_harness.py``,
``test_backend.py``, ``eval_bag_waypoints.py``) as well as by the live
pipeline, so they must not be renamed or reordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations are never evaluated
    import numpy as np
    import torch


@dataclass
class FrameEntry:
    """One decoded frame.

    ``timestamp_us`` is the clock the pipeline aligns on: the CIMG packet's
    ``capture_time_ns``, converted from nanoseconds to microseconds.

    ``pts`` and ``arrival_us`` are kept unconditionally so that
    - a frame can always be identified on the vehicle side ("cam1:217471778"),
    - the video pipeline latency / frame loss can still be measured after the
      clock domain switches (arrival_us is monotonic by definition).
    """

    timestamp_us: int
    bgr: np.ndarray
    pts: int | None = None
    arrival_us: int | None = None
    # --- CIMG (direct JPEG push) provenance ------------------------------- #
    #
    # Appended rather than inserted so offline tools that build this class
    # positionally keep the original field order.
    #
    #: Pre-formatted frame identity, e.g. ``cam1:s217`` for a frame whose
    #: vehicle sequence is 217.  ``camera.sync`` prefers it over ``pts`` so the
    #: inference package records the vehicle's own frame number.
    frame_id: str | None = None
    #: The vehicle's per-camera frame number, verbatim (CIMG packets only).
    sequence: int | None = None
    #: The ``camera_id`` the vehicle sent (1..4), before the model-index map.
    camera_id: int | None = None
    #: The vehicle's unified group timestamp for the packet this frame arrived
    #: in.  All four frames of one packet share it; ``timestamp_us`` above is
    #: this individual frame's own capture time, so the difference between the
    #: two is the per-camera offset inside the group.
    group_time_us: int | None = None
    #: The JPEG the vehicle sent, before decode.  Appended so positional
    #: constructors keep the original field order.  Network workers receive
    #: these bytes; the decoded tensor stays on PGX-1 for the local worker.
    jpeg_bytes: bytes | None = None


@dataclass
class CameraWindow:
    t0_us: int
    # (N_camera, T, 3, H, W) uint8 RGB, sorted by model camera index
    frames: torch.Tensor
    camera_indices: list[int]
    timestamps_us: list[int]
    # Per-frame identity from the CIMG packet sequence:
    # (N_camera, T) entries like "cam1:s217" (None when unavailable).
    frame_ids: list[list[str | None]] | None = None
    # Per-frame clock from capture_time_ns, in microseconds. Kept so
    # inference records can be audited frame by frame.
    frame_timestamps_us: list[list[int]] | None = None
    # Decode/arrival time on the server monotonic clock, independent of the
    # alignment clock above; survives the switch to the vehicle time domain.
    frame_arrivals_us: list[list[int | None]] | None = None
    # Age of the stalest camera's newest frame when this window was built.
    # Always uses server monotonic arrival time, regardless of alignment mode.
    frame_age_us: int | None = None
    # Original JPEGs in the same (camera, time) order as ``frames``,
    # ``frame_ids`` and ``frame_timestamps_us``.  ``None`` for a frame that
    # did not arrive as JPEG.  The network path sends this, not ``frames``.
    jpeg_bytes: list[list[bytes | None]] | None = None


__all__ = ["CameraWindow", "FrameEntry"]
