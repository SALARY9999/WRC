#!/usr/bin/env python3
"""Verify the azimuth-to-heading mapping and lat/lon direction."""

import os
import sys

import numpy as np

BACKEND = "/data/users_data/wanghaohao/701/alpamayo_infer_backend"
sys.path.insert(0, BACKEND)
import app  # noqa: E402


def make_pose(azimuth: float, speed: float = 0.0) -> "app.GlobalPose":
    pose = app.GlobalPose(
        timestamp_us=0,
        latitude=28.0,
        longitude=113.0,
        altitude=0.0,
        v_north=0.0,
        v_east=0.0,
        v_up=0.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        azimuth_deg=azimuth,
    )
    heading = pose.heading_deg()
    pose.v_north = speed * np.cos(np.radians(heading))
    pose.v_east = speed * np.sin(np.radians(heading))
    return pose


def main() -> int:
    print("heading_source=vehicle_azimuth")
    for azimuth in (0.0, 90.0, 178.912, 271.967):
        pose = make_pose(azimuth)
        print(f"azimuth={azimuth:>8.3f} -> heading={pose.heading_deg():>8.3f}")

    x = np.array([10.0])
    y = np.array([0.0])
    lat_n, lon_n = app.local_xy_to_latlon(x, y, 28.0, 113.0, 0.0)
    lat_e, lon_e = app.local_xy_to_latlon(x, y, 28.0, 113.0, 90.0)
    print(
        f"x=10m heading=0 -> dlat={lat_n[0] - 28.0:+.7f} "
        f"dlon={lon_n[0] - 113.0:+.7f}"
    )
    print(
        f"x=10m heading=90 -> dlat={lat_e[0] - 28.0:+.7f} "
        f"dlon={lon_e[0] - 113.0:+.7f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
