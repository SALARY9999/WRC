#!/usr/bin/env python3
"""Verify chassis decoding for top-level and pose-nested JSON layouts."""

import json
import sys

BACKEND = "/data/users_data/wanghaohao/701/alpamayo_infer_backend"
sys.path.insert(0, BACKEND)

import app  # noqa: E402


def main() -> int:
    nested = {
        "type": "vehicle_state",
        "timestamp": 1788426435.404,
        "pose": {
            "latitude": 28.0,
            "longitude": 113.0,
            "height": 0.0,
            "azimuth": 89.956,
            "vNorth": 0.0,
            "vEast": 0.0,
            "local_time": 17,
            "chassis": {"cur_spd": 1.23, "yaw_rate": 4.5, "local_time": 0.0},
        },
    }
    top = {
        "type": "vehicle_state",
        "timestamp": 1788426435.404,
        "pose": {"latitude": 28.0, "longitude": 113.0, "azimuth": 89.956},
        "chassis": {"cur_spd": 2.34, "yaw_rate": -6.0},
    }
    for name, payload in (("pose.chassis", nested), ("top-level", top)):
        chassis = app.decode_chassis_json(
            json.dumps(payload).encode("utf-8")
        )
        print(
            f"{name}: speed={chassis.speed_mps} "
            f"yaw_rate_rad_s={chassis.yaw_rate_rad_s:.6f} "
            f"ts_us={chassis.timestamp_us}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
