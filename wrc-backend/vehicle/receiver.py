"""Vehicle payload decoders and the pose/chassis listeners.

Moved verbatim out of ``app.py``.  The wire formats, the accepted JSON
layouts and the timestamp-resolution order are a compatibility surface: the
real vehicle sends pose and chassis in a single ``vehicle_state`` datagram on
port 7001, with the chassis nested under ``"pose"``, and both the documented
and the actual layouts must keep decoding.

``MockPoseProvider`` remains the offline bring-up source; it is deterministic
on purpose so that a mock run is reproducible.
"""

from __future__ import annotations

import json
import logging
import math
import select
import socket
import threading
import time
from collections import deque

from time_sync import TimestampManager
from trajectory.coordinates import _M_PER_DEG_LAT, _M_PER_DEG_LON_EQ
from vehicle.history import PoseHistory
from vehicle.models import ChassisInfo, GlobalPose, vehicle_azimuth_for_compass
from vehicle.trace import JsonlTrace, pose_record

logger = logging.getLogger(__name__)


def decode_global_pose_json(payload: bytes) -> GlobalPose:
    data = json.loads(payload.decode("utf-8", "replace"))
    if not isinstance(data, dict) or not isinstance(data.get("pose"), dict):
        raise ValueError("unsupported GlobalPose JSON layout")

    # Actual vehicle packet:
    # {"type":"vehicle_state","timestamp":<sec>,
    #  "pose":{...},"chassis":{...}}
    body = data["pose"]
    if "latitude" not in body or "longitude" not in body:
        raise ValueError("unsupported GlobalPose JSON layout")
    azimuth = body.get("azimuth")
    if (
        isinstance(azimuth, bool)
        or not isinstance(azimuth, (int, float))
        or not math.isfinite(azimuth)
    ):
        raise ValueError("vehicle_state pose.azimuth must be a finite number in degrees")
    ts_raw = data.get("timestamp")
    if ts_raw is None:
        raise ValueError("vehicle_state JSON is missing top-level timestamp")
    ts_us = int(float(ts_raw) * 1_000_000)  # s -> us

    return GlobalPose(
        timestamp_us=ts_us,
        payload_timestamp_us=ts_us,
        latitude=float(body["latitude"]),
        longitude=float(body["longitude"]),
        altitude=float(body.get("height", body.get("altitude", 0.0))),
        v_north=float(body.get("vNorth", body.get("v_north", 0.0))),
        v_east=float(body.get("vEast", body.get("v_east", 0.0))),
        v_up=float(body.get("vUp", body.get("v_up", 0.0))),
        roll_deg=float(body.get("roll", 0.0)),
        pitch_deg=float(body.get("pitch", 0.0)),
        azimuth_deg=float(azimuth),
    )


def decode_vehicle_timestamp_json(payload: bytes) -> int | float:
    """Return the unscaled VehicleState timestamp for configured normalization."""
    data = json.loads(payload.decode("utf-8", "replace"))
    if not isinstance(data, dict) or data.get("timestamp") is None:
        raise ValueError("vehicle_state JSON is missing top-level timestamp")
    value = data["timestamp"]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("vehicle_state JSON timestamp must be numeric")
    return value


def decode_chassis_json(payload: bytes) -> ChassisInfo:
    data = json.loads(payload.decode("utf-8", "replace"))
    body = None
    if isinstance(data, dict):
        # Documented layout: top-level "chassis".
        if isinstance(data.get("chassis"), dict):
            body = data["chassis"]
        else:
            # Actual vehicle layout: chassis nested inside "pose".
            pose = data.get("pose")
            if isinstance(pose, dict) and isinstance(pose.get("chassis"), dict):
                body = pose["chassis"]
    if not isinstance(body, dict):
        raise ValueError("unsupported ChassisInfo JSON layout")
    if "cur_spd" not in body:
        raise ValueError("unsupported ChassisInfo JSON layout")

    ts_raw = data.get("timestamp")
    if ts_raw is None:
        raise ValueError("vehicle_state JSON is missing top-level timestamp")
    ts = int(float(ts_raw) * 1_000_000)

    return ChassisInfo(
        timestamp_us=ts,
        speed_mps=float(body["cur_spd"]),
        yaw_rate_rad_s=math.radians(float(body.get("yaw_rate", 0.0))),
        lon_acc_mps2=float(body.get("lon_acc", 0.0)),
        lat_acc_mps2=float(body.get("lat_acc", 0.0)),
        gear=int(body.get("cur_gear", 0)),
    )


class UdpPoseListener(threading.Thread):
    """Receives GlobalPose/ChassisInfo over UDP and decodes them.

    Default decoder is JSON because the real binary payload is not yet
    available.  Replace ``globalpose_decoder``/``chassis_decoder`` once a
    capture sample is obtained.
    """

    def __init__(
        self,
        history: PoseHistory,
        *,
        pose_host: str = "0.0.0.0",
        pose_port: int = 7001,
        azimuth_offset_deg: float = 0.0,
        timestamp_manager: TimestampManager | None = None,
        pose_decoder=decode_global_pose_json,
        chassis_decoder=decode_chassis_json,
        trace: JsonlTrace | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.history = history
        self._trace = trace
        self._pose_port = int(pose_port)
        self._azimuth_offset_deg = float(azimuth_offset_deg)
        self._hosts = [(pose_host, pose_port)]
        self._pose_decoder = pose_decoder
        self._chassis_decoder = chassis_decoder
        self._timestamp_manager = timestamp_manager or TimestampManager()
        # _stop_event, not _stop: threading.Thread defines an internal _stop()
        # method that join() calls, so an attribute named _stop shadows it and
        # makes join() raise "TypeError: 'Event' object is not callable".
        # PipelineRunner.stop() does not join today, but Stage 3 needs it to.
        self._stop_event = threading.Event()
        self._sockets: list[socket.socket] = []
        self.last_error: str | None = None
        self._pose_total = 0
        self._pose_window = 0
        self._pose_window_start = time.monotonic()
        self._last_pose: GlobalPose | None = None
        self._chassis_total = 0
        self._last_chassis: ChassisInfo | None = None
        self.chassis_history: deque[ChassisInfo] = deque(maxlen=2000)
        self.last_timestamp_us: int | None = None

    def latest_chassis(self) -> ChassisInfo | None:
        return self._last_chassis

    def _apply_timestamp(
        self,
        pose: GlobalPose,
        arrival_us: int,
        raw_unix_timestamp: int | float | None = None,
    ) -> None:
        """Apply the vehicle Unix timestamp, in microseconds."""
        pose.arrival_us = arrival_us
        if pose.payload_timestamp_us is None:
            pose.payload_timestamp_us = pose.timestamp_us
        if raw_unix_timestamp is None:
            timestamp_us = int(pose.timestamp_us)
        else:
            timestamp_us = self._timestamp_manager.resolve_pose(raw_unix_timestamp)
        self.last_timestamp_us = timestamp_us
        pose.timestamp_us = timestamp_us

    def run(self) -> None:
        for host, port in self._hosts:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                s.setblocking(False)
                self._sockets.append(s)
                logger.info("UDP listener bound %s:%s", host, port)
            except OSError as exc:
                logger.warning("UDP bind %s:%s failed: %s", host, port, exc)
                self.last_error = str(exc)
        while not self._stop_event.is_set():
            try:
                readable, _, _ = select.select(self._sockets, [], [], 1.0)
            except OSError:
                break
            for s in readable:
                while not self._stop_event.is_set():
                    try:
                        payload, _addr = s.recvfrom(65535)
                    except (BlockingIOError, socket.timeout):
                        break
                    except OSError:
                        break
                    try:
                        if s.getsockname()[1] == self._pose_port:
                            self.ingest(payload)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("UDP decode failed: %s", exc)

    def ingest(self, payload: bytes) -> None:
        """Decode one pose datagram and append it to the trace, if recording.

        The UDP loop and tests share this path.  A trace write that fails is
        logged inside the writer and does not discard the pose already stored
        in ``history``.
        """
        pose = self._pose_decoder(payload)
        pose.azimuth_offset_deg = self._azimuth_offset_deg
        raw_unix_timestamp = (
            decode_vehicle_timestamp_json(payload)
            if self._pose_decoder is decode_global_pose_json
            else None
        )
        arrival_us = time.monotonic_ns() // 1000
        self._apply_timestamp(pose, arrival_us, raw_unix_timestamp)
        self.history.add(pose)
        self._pose_total += 1
        self._pose_window += 1
        self._last_pose = pose
        # The vehicle sends pose and chassis in the same
        # 7001 JSON (chassis nested under pose).  A missing chassis is recorded
        # as empty for this packet; the summary below keeps the last good one.
        chassis: ChassisInfo | None = None
        try:
            chassis = self._chassis_decoder(payload)
            self._last_chassis = chassis
            self._chassis_total += 1
            self.chassis_history.append(chassis)
        except Exception:  # noqa: BLE001 - optional field
            pass
        self._append_pose_trace(pose, chassis)
        now = time.monotonic()
        if now - self._pose_window_start >= 5.0:
            p = self._last_pose
            c = self._last_chassis
            logger.info(
                "pose-in summary: packets_5s=%d total=%d "
                "lat=%.7f lon=%.7f heading=%.2f speed=%.2f "
                "ts_us=%d chassis_spd=%.2f chassis_yaw=%.3f "
                "chassis_n=%d",
                self._pose_window,
                self._pose_total,
                p.latitude,
                p.longitude,
                p.heading_deg(),
                p.speed_mps,
                p.timestamp_us,
                c.speed_mps if c is not None else float("nan"),
                c.yaw_rate_rad_s if c is not None else float("nan"),
                self._chassis_total,
            )
            self._pose_window = 0
            self._pose_window_start = now

    def _append_pose_trace(
        self, pose: GlobalPose, chassis: ChassisInfo | None
    ) -> None:
        trace = self._trace
        if trace is None:
            return
        try:
            trace.append(pose_record(pose, chassis))
        except Exception as exc:  # noqa: BLE001 - recording is not the pose path
            logger.warning("failed to record pose: %r", exc)

    def stop(self) -> None:
        self._stop_event.set()
        for s in self._sockets:
            try:
                s.close()
            except OSError:
                pass
        self._sockets.clear()


class MockPoseProvider(threading.Thread):
    """Simple deterministic pose generator for offline bring-up."""

    def __init__(
        self, history: PoseHistory, timestamp_manager: TimestampManager
    ) -> None:
        super().__init__(daemon=True)
        self.history = history
        self.timestamp_manager = timestamp_manager
        # See UdpPoseListener: do not rename this back to _stop.
        self._stop_event = threading.Event()

    def run(self) -> None:
        lat = 31.1190589
        lon = 114.4639576
        heading_deg = 90.0
        speed = 2.0
        dt_s = 0.1
        while not self._stop_event.is_set():
            arrival_us = time.monotonic_ns() // 1000
            timestamp_us = self.timestamp_manager.resolve_pose(time.time())
            # straight east at 2 m/s
            east_m = speed * dt_s * math.sin(math.radians(heading_deg))
            north_m = speed * dt_s * math.cos(math.radians(heading_deg))
            lon += east_m / (_M_PER_DEG_LON_EQ * math.cos(math.radians(lat)))
            lat += north_m / _M_PER_DEG_LAT
            self.history.add(
                GlobalPose(
                    timestamp_us=timestamp_us,
                    latitude=lat,
                    longitude=lon,
                    altitude=0.0,
                    v_north=speed * math.cos(math.radians(heading_deg)),
                    v_east=speed * math.sin(math.radians(heading_deg)),
                    v_up=0.0,
                    roll_deg=0.0,
                    pitch_deg=0.0,
                    azimuth_deg=vehicle_azimuth_for_compass(heading_deg),
                    arrival_us=arrival_us,
                )
            )
            self._stop_event.wait(dt_s)

    def stop(self) -> None:
        self._stop_event.set()


__all__ = [
    "MockPoseProvider",
    "UdpPoseListener",
    "decode_chassis_json",
    "decode_global_pose_json",
    "decode_vehicle_timestamp_json",
]
