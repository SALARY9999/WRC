"""Vehicle-side data models.

Field names, units and defaults are a compatibility surface: these classes
are constructed by the UDP receiver, by the offline tools and by
``test_backend.py``, and their values are written into the per-inference
package, so they must not be renamed or reordered.

Units follow the vehicle payload exactly: degrees, metres, metres/second,
Unix microseconds for ``payload_timestamp_us``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Vehicle ``azimuth`` is clockwise from west: 0 west, 90 north, 180 east,
# 270 south.  The rest of this stack uses a compass heading, clockwise from
# north: 0 north, 90 east, 180 south, 270 west.  The two zeros differ by 90°.
VEHICLE_AZIMUTH_TO_COMPASS_DEG = -90.0


def vehicle_azimuth_for_compass(heading_deg: float) -> float:
    """Vehicle azimuth that ``heading_deg()`` maps back to this compass heading.

    Installation offset is not included; callers that set
    ``azimuth_offset_deg`` must account for it themselves.
    """
    return (float(heading_deg) - VEHICLE_AZIMUTH_TO_COMPASS_DEG) % 360.0


@dataclass
class GlobalPose:
    timestamp_us: int
    latitude: float
    longitude: float
    altitude: float
    v_north: float
    v_east: float
    v_up: float
    roll_deg: float
    pitch_deg: float
    azimuth_deg: float
    # Original vehicle payload timestamp (Unix epoch, us) kept for diagnostics.
    # ``timestamp_us`` is the same Unix axis the camera uses, in microseconds.
    payload_timestamp_us: int | None = None
    # Server monotonic receive time. Used to measure how long the packet has
    # been here; it is not the clock camera and pose are aligned on.
    arrival_us: int | None = None
    # Fixed vehicle/sensor installation correction loaded from config.json.
    azimuth_offset_deg: float = 0.0

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.v_east, self.v_north)

    def heading_deg(self) -> float:
        """Compass heading of the vehicle body, clockwise from north.

        ``azimuth_deg`` is the vehicle payload: clockwise from west.  It is
        converted with ``VEHICLE_AZIMUTH_TO_COMPASS_DEG`` before the optional
        installation correction from ``vehicle_heading`` in config.json.
        Course-over-ground is not a heading source: it can differ from the
        body direction while reversing, turning or sliding.
        """
        return (
            self.azimuth_deg
            + VEHICLE_AZIMUTH_TO_COMPASS_DEG
            + self.azimuth_offset_deg
        ) % 360.0


@dataclass
class ChassisInfo:
    timestamp_us: int
    speed_mps: float
    yaw_rate_rad_s: float
    lon_acc_mps2: float
    lat_acc_mps2: float
    gear: int


@dataclass
class PoseSample:
    timestamp_us: int
    latitude: float
    longitude: float
    altitude: float
    heading_deg: float


__all__ = [
    "ChassisInfo",
    "GlobalPose",
    "PoseSample",
    "VEHICLE_AZIMUTH_TO_COMPASS_DEG",
    "vehicle_azimuth_for_compass",
]
