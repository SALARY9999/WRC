"""Vehicle route protocol: <BBIIB header followed by <ff lon/lat points."""
from __future__ import annotations

import logging
import math
import socket
import struct
import threading
import time

from trajectory.takeover import PlannedRoute, PlannedRouteCache, PlannedRoutePoint

logger = logging.getLogger(__name__)
HEADER = struct.Struct("<BBIIB")
POINT = struct.Struct("<ff")
MAGIC = b"\xf1\x10"


class VehiclePlannedRouteAdapter:
    def __init__(self, max_points: int = 100_000) -> None:
        self.max_points = max_points

    def decode(self, payload: bytes) -> PlannedRoute | None:
        if len(payload) < HEADER.size:
            raise ValueError("truncated route header")
        high, low, length, count, reached = HEADER.unpack_from(payload)
        if bytes((high, low)) != MAGIC:
            raise ValueError("invalid route magic")
        if count > self.max_points or length != HEADER.size + count * POINT.size:
            raise ValueError("invalid route length/count")
        if length != len(payload) or reached not in (0, 1):
            raise ValueError("invalid route size/reached")
        points = []
        for offset in range(HEADER.size, length, POINT.size):
            lon, lat = POINT.unpack_from(payload, offset)
            if not math.isfinite(lon) or not math.isfinite(lat):
                raise ValueError("non-finite route coordinate")
            if not -180 <= lon <= 180 or not -90 <= lat <= 90:
                raise ValueError("out-of-range route coordinate")
            points.append(PlannedRoutePoint((lon, lat)))
        # Completion and an empty replacement both invalidate the old route.
        if reached or not points:
            return None
        return PlannedRoute(tuple(points), "geographic_lon_lat_degrees")


class RouteStreamParser:
    """Per-connection bounded buffer for arbitrary TCP chunk boundaries."""
    def __init__(self, adapter: VehiclePlannedRouteAdapter | None = None) -> None:
        self.adapter = adapter or VehiclePlannedRouteAdapter()
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[PlannedRoute | None]:
        self.buffer.extend(data)
        updates = []
        while len(self.buffer) >= HEADER.size:
            high, low, length, count, reached = HEADER.unpack_from(self.buffer)
            if bytes((high, low)) != MAGIC:
                raise ValueError("invalid route frame header; closing connection")
            if count > self.adapter.max_points or length != HEADER.size + count * POINT.size:
                raise ValueError("invalid route frame length/count")
            if reached not in (0, 1):
                raise ValueError("invalid route reached flag")
            if len(self.buffer) < length:
                break
            payload = bytes(self.buffer[:length])
            del self.buffer[:length]
            updates.append(self.adapter.decode(payload))
        return updates


class PlannedRouteTcpReceiver(threading.Thread):
    """Single route writer; malformed/incomplete frames never overwrite cache."""
    def __init__(self, cache: PlannedRouteCache, host: str = "0.0.0.0", port: int = 30006,
                 client_timeout_s: float = 30.0) -> None:
        super().__init__(daemon=True, name="planned-route-tcp")
        self.cache = cache
        self.host, self.port = host, port
        self.client_timeout_s = client_timeout_s
        self._stop_event = threading.Event()
        self.listening = False
        self.last_error: str | None = None
        self.updates = 0

    def run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.host, self.port))
                self.port = listener.getsockname()[1]
                listener.listen(4)
                listener.settimeout(0.5)
                self.listening = True
                logger.info("planned route TCP listening: %s:%d", self.host, self.port)
                while not self._stop_event.is_set():
                    try:
                        client, peer = listener.accept()
                    except socket.timeout:
                        continue
                    with client:
                        client.settimeout(0.5)
                        parser = RouteStreamParser()
                        last_data_at = time.monotonic()
                        logger.info("planned route client connected: %s", peer)
                        try:
                            while not self._stop_event.is_set():
                                try:
                                    data = client.recv(65536)
                                except socket.timeout:
                                    if time.monotonic() - last_data_at > self.client_timeout_s:
                                        raise ValueError("route client idle timeout")
                                    continue
                                if not data:
                                    if parser.buffer:
                                        raise ValueError("incomplete route frame on disconnect")
                                    break
                                last_data_at = time.monotonic()
                                for route in parser.feed(data):
                                    if route is None:
                                        self.cache.clear()
                                        logger.info("planned route cleared: completed_or_empty")
                                    else:
                                        version = self.cache.update(route)
                                        logger.info("planned route replaced: version=%d points=%d first=%s last=%s",
                                                    version, len(route.points), route.points[0].coordinates,
                                                    route.points[-1].coordinates)
                                    self.updates += 1
                        except (ValueError, OSError) as exc:
                            self.last_error = str(exc)
                            logger.warning("planned route client rejected: peer=%s reason=%s", peer, exc)
        except OSError as exc:
            self.last_error = str(exc)
            logger.error("planned route TCP unavailable: %s:%d reason=%s", self.host, self.port, exc)
        finally:
            self.listening = False

    def stop(self) -> None:
        """Signal, then wait up to 2 s for the listener thread.

        ``timeout=2.0`` is unchanged and load-bearing: ``tools/receive_route_test.py``
        and ``tests/test_planned_route_tcp.py`` both depend on this method
        returning within that bound.

        The ``ident is None`` guard is new.  ``Thread.join()`` raises
        ``RuntimeError("cannot join thread before it is started")`` on a thread
        that was never started, and shutdown reaches exactly that state: the
        runner is constructed and only then started, so a startup that failed in
        between (or an ASGI server that gave up) calls ``stop()`` on a receiver
        whose thread never ran.  Without the guard, shutdown reports the receiver
        as *failed* and logs a traceback at the one moment an operator is reading
        the log - for a component that is simply not running.
        """
        self._stop_event.set()
        if self.ident is None:
            self.listening = False
            return
        self.join(timeout=2.0)
