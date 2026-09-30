"""Append-only record of the vehicle streams that pass through PGX-1.

One file, ``vehicle.jsonl``, under ``ALPAMAYO_VEHICLE_TRACE_DIR``. Pose
datagrams and outbound waypoint packets are appended in the order they
happen, so one ``tail -f`` shows both. Each line's ``event`` field is
``pose`` or ``waypoints``. A waypoint line keeps the points that were sent
in ``waypoints`` and, when the caller supplies them, the model points from
before the latency crop in ``waypoints_before_trim``. Both lists are
``[latitude, longitude]``.

Lines are only appended. Nothing in this module rotates or deletes a file.
A write that fails is logged and swallowed: losing a line is preferable to
dropping a pose or skipping a waypoint send.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from vehicle.models import ChassisInfo, GlobalPose

logger = logging.getLogger(__name__)

#: The single trace file. Pose and waypoint lines share it.
VEHICLE_TRACE_NAME = "vehicle.jsonl"


class JsonlTrace:
    """One append-only JSONL file.

    ``append`` never raises.  The caller is on the pose thread or the pipeline
    thread, and either one must keep running when the disk does not.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            directory = os.path.dirname(self.path)
            with self._lock:
                if directory:
                    os.makedirs(directory, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
        except Exception as exc:  # noqa: BLE001 - a trace must not stop the vehicle
            logger.warning("failed to append vehicle trace %s: %r", self.path, exc)


def open_jsonl_trace(directory: str, filename: str) -> JsonlTrace | None:
    """A writer for ``directory/filename``, or None when recording is off.

    An empty directory is the unset state.  Tests and mock runs leave the
    variable unset and must not create files.
    """
    root = (directory or "").strip()
    if not root:
        return None
    return JsonlTrace(os.path.join(root, filename))


def pose_record(
    pose: GlobalPose,
    chassis: ChassisInfo | None,
    *,
    recorded_at: float | None = None,
) -> dict[str, Any]:
    """One inbound vehicle pose, including the chassis that arrived with it."""
    return {
        "event": "pose",
        "timestamp": round(
            time.time() if recorded_at is None else float(recorded_at), 6
        ),
        "latitude": float(pose.latitude),
        "longitude": float(pose.longitude),
        "altitude": float(pose.altitude),
        "azimuth_deg": float(pose.azimuth_deg),
        "heading_deg": float(pose.heading_deg()),
        "v_north": float(pose.v_north),
        "v_east": float(pose.v_east),
        "chassis_speed_mps": (
            None if chassis is None else float(chassis.speed_mps)
        ),
        "chassis_yaw_rate_rad_s": (
            None if chassis is None else float(chassis.yaw_rate_rad_s)
        ),
        "payload_timestamp_us": (
            None
            if pose.payload_timestamp_us is None
            else int(pose.payload_timestamp_us)
        ),
        "timestamp_us": int(pose.timestamp_us),
        "arrival_us": (
            None if pose.arrival_us is None else int(pose.arrival_us)
        ),
    }


def _latlon_points(
    latitude: Any, longitude: Any, count: int | None = None
) -> list[list[float]]:
    points: list[list[float]] = []
    for index, (lat, lon) in enumerate(zip(latitude, longitude)):
        if count is not None and index >= count:
            break
        points.append([round(float(lat), 8), round(float(lon), 8)])
    return points


def waypoint_record(
    *,
    send_timestamp: float,
    waypoint_seq: int,
    control_flag: int,
    host: str,
    port: int,
    latitude: Any,
    longitude: Any,
    count: int,
    inference_latency_ms: float,
    cycle_id: int,
    worker_id: int | None,
    device: int | None,
    endpoint: str,
    latitude_before_trim: Any = None,
    longitude_before_trim: Any = None,
) -> dict[str, Any]:
    """One outbound waypoint datagram, with every point that was sent.

    ``latitude_before_trim`` / ``longitude_before_trim`` are the model
    trajectory in the same WGS84 frame, before the latency crop. They are
    recorded in full; ``count`` limits only the points that were sent.
    """
    record: dict[str, Any] = {
        "event": "waypoints",
        "timestamp": round(float(send_timestamp), 6),
        "waypoint_seq": int(waypoint_seq),
        "control_flag": int(control_flag),
        "host": str(host),
        "port": int(port),
        "inference_latency_ms": round(float(inference_latency_ms), 3),
        "cycle_id": int(cycle_id),
        "worker_id": None if worker_id is None else int(worker_id),
        "device": None if device is None else int(device),
        "endpoint": str(endpoint or ""),
        "waypoints": _latlon_points(latitude, longitude, int(count)),
    }
    if latitude_before_trim is not None and longitude_before_trim is not None:
        record["waypoints_before_trim"] = _latlon_points(
            latitude_before_trim, longitude_before_trim
        )
    return record


__all__ = [
    "VEHICLE_TRACE_NAME",
    "JsonlTrace",
    "open_jsonl_trace",
    "pose_record",
    "waypoint_record",
]
