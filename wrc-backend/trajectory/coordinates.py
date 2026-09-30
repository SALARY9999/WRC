"""Coordinate transforms between WGS84, local ENU and the vehicle frame.

These four functions were previously private to ``app.py``.  They are pulled
out here because they are shared by the pose history builder (ego tensors),
the planned-route takeover decision and the waypoint sender - three modules
that should not have to import the FastAPI application to convert a
coordinate.

Sign conventions are load-bearing and unchanged:

* ``enu_from_ll`` - east/north/up metres relative to ``ref`` (lat, lon, alt).
* ``enu_to_ego`` - vehicle frame ``+x`` forward, ``+y`` left, ``+z`` up.
* ``local_xy_to_latlon`` - the inverse of the two above, used on the outgoing
  waypoint path.

The metre-per-degree constants are the same local-tangent-plane approximation
the vehicle has been validated against; do not "improve" them without a
field comparison.
"""

from __future__ import annotations

import math

import numpy as np

_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQ = 111_320.0


def enu_from_ll(lat: float, lon: float, alt: float, ref: tuple[float, float, float]):
    lat_rad = math.radians(ref[0])
    east = (lon - ref[1]) * _M_PER_DEG_LON_EQ * math.cos(lat_rad)
    north = (lat - ref[0]) * _M_PER_DEG_LAT
    up = alt - ref[2]
    return east, north, up


def yaw_to_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def enu_to_ego(
    east: float, north: float, up: float, heading_rad: float
) -> tuple[float, float, float]:
    """ENU delta to vehicle frame (+x forward, +y left, +z up)."""
    x = east * math.sin(heading_rad) + north * math.cos(heading_rad)
    y = -east * math.cos(heading_rad) + north * math.sin(heading_rad)
    return x, y, up


def local_xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    heading_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vehicle-frame x/y (forward/left) to WGS84 degrees using a local tangent plane."""
    h = math.radians(heading_deg)
    sin_h = math.sin(h)
    cos_h = math.cos(h)
    east = x * sin_h - y * cos_h
    north = x * cos_h + y * sin_h
    m_per_deg_lon = _M_PER_DEG_LON_EQ * math.cos(math.radians(ref_lat))
    lon = ref_lon + east / m_per_deg_lon
    lat = ref_lat + north / _M_PER_DEG_LAT
    return lat, lon


__all__ = [
    "enu_from_ll",
    "enu_to_ego",
    "local_xy_to_latlon",
    "yaw_to_matrix",
]
