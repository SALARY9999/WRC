"""Outgoing waypoint datagrams: the backend -> vehicle wire protocol.

Moved verbatim out of ``app.py``.  The byte layout is a compatibility surface
and is frozen by ``tests/test_protocol_golden.py``:

    <IBBIHH  magic(0x514D3031) version control seq vehicle_id count
    N x <dd  latitude, longitude

Little-endian, 14-byte header, 16 bytes per waypoint.  ``count`` is derived
from the arrays actually passed in - the caller must not pre-truncate - and a
zero count is a legal frame: it is how the pipeline releases a takeover whose
trajectory has expired (see ``trim_waypoints_to_latest_pose``).
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotation only
    import numpy as np

    from config.loader import AppConfig as Settings

logger = logging.getLogger(__name__)


class WaypointSender:
    """Serializes waypoints and sends them over UDP, one sequence per packet."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0
        self._lock = threading.Lock()
        self.last_seq = -1
        self.last_send_at = 0.0
        self._closed = False

    def send(
        self,
        control_flag: int,
        lat: "np.ndarray",
        lon: "np.ndarray",
    ) -> int:
        cfg = self.settings
        if not cfg.allow_unconfigured_udp and (cfg.magic == 0 or cfg.vehicle_id == 0):
            raise RuntimeError(
                "waypoint UDP magic/vehicle_id not configured; "
                "set ALPAMAYO_WAYPOINT_MAGIC / ALPAMAYO_VEHICLE_ID "
                "or ALPAMAYO_ALLOW_UNCONFIGURED_UDP=1"
            )
        n = int(min(len(lat), len(lon)))
        if n > 65535:
            raise ValueError("too many waypoints")
        with self._lock:
            seq = self._seq
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            header = struct.pack(
                cfg.byte_order + "IBBIHH",
                cfg.magic,
                cfg.version,
                int(control_flag) & 0xFF,
                seq,
                cfg.vehicle_id & 0xFFFF,
                n,
            )
            body = b"".join(
                struct.pack(cfg.byte_order + "dd", float(la), float(lo))
                for la, lo in zip(lat[:n], lon[:n])
            )
        packet = header + body
        self.sock.sendto(packet, (cfg.waypoint_host, cfg.waypoint_port))
        with self._lock:
            self.last_seq = seq
            self.last_send_at = time.time()
        first = (float(lat[0]), float(lon[0])) if n else (None, None)
        last = (float(lat[n - 1]), float(lon[n - 1])) if n else (None, None)
        logger.debug(
            "waypoint-out: -> %s:%d magic=0x%X ver=%d control=%d seq=%d "
            "vehicle_id=%d n=%d first=(%.7f,%.7f) last=(%.7f,%.7f)",
            cfg.waypoint_host,
            cfg.waypoint_port,
            cfg.magic,
            cfg.version,
            int(control_flag),
            seq,
            cfg.vehicle_id,
            n,
            first[0] or 0.0,
            first[1] or 0.0,
            last[0] or 0.0,
            last[1] or 0.0,
        )
        return n

    def close(self) -> None:
        """Release the UDP socket.  Idempotent, and never raises.

        Called by ``PipelineRunner.stop()`` only once the loop thread has
        confirmed it is no longer running.  A late ``send()`` on a closed socket
        raises ``OSError``, which the loop's own ``except Exception`` would turn
        into a ``pipeline cycle failed`` traceback during shutdown - so the order
        matters, and the caller enforces it.

        Before Stage 3 nothing closed this socket at all: it was released only
        when the process exited, which is why a test that built a runner used to
        leak one descriptor per construction.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.sock.close()
            except OSError:
                pass

    @property
    def closed(self) -> bool:
        return self._closed


__all__ = ["WaypointSender"]
