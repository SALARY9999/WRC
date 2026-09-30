import socket
import struct
import time
import unittest

from vehicle.planned_route import PlannedRouteTcpReceiver, RouteStreamParser, VehiclePlannedRouteAdapter
from trajectory.takeover import PlannedRouteCache


def frame(points=((113.0, 28.0), (113.01, 28.01)), reached=0):
    return struct.pack("<BBIIB", 0xF1, 0x10, 11 + 8 * len(points), len(points), reached) + b"".join(
        struct.pack("<ff", *p) for p in points)


class RouteTcpTests(unittest.TestCase):
    def test_little_endian_longitude_first(self):
        route = VehiclePlannedRouteAdapter().decode(frame())
        self.assertEqual(route.points[0].coordinates, (113.0, 28.0))

    def test_fragmented_and_coalesced_frames(self):
        parser = RouteStreamParser()
        payload = frame() + frame(reached=1)
        updates = []
        for byte in payload:
            updates.extend(parser.feed(bytes((byte,))))
        self.assertEqual(len(updates), 2)
        self.assertIsNone(updates[1])
        self.assertEqual(len(RouteStreamParser().feed(payload)), 2)

    def test_invalid_frames_rejected(self):
        for payload in (b"bad header!", frame()[:-1], frame(((float('nan'), 28),)),
                        frame(((190, 28),)), frame(reached=2)):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                VehiclePlannedRouteAdapter().decode(payload)
        with self.assertRaises(ValueError):
            RouteStreamParser().feed(struct.pack("<BBIIB", 0xF1, 0x10, 0xffffffff, 0xffffffff, 0))

    def test_empty_route_clears(self):
        self.assertIsNone(VehiclePlannedRouteAdapter().decode(frame(())))

    def test_real_socket_replace_clear_and_reconnect(self):
        cache = PlannedRouteCache()
        receiver = PlannedRouteTcpReceiver(cache, "127.0.0.1", 0)
        # Use a bound listener port published by the receiver.
        receiver.start()
        deadline = time.monotonic() + 3
        while not receiver.listening and time.monotonic() < deadline:
            time.sleep(.01)
        try:
            self.assertTrue(receiver.listening)
            with socket.create_connection(("127.0.0.1", receiver.port)) as client:
                client.sendall(frame())
                deadline = time.monotonic() + 2
                while cache.snapshot()[1] < 1 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(len(cache.snapshot()[0].points), 2)
                client.sendall(frame(((114, 30),)) + frame(reached=1))
                while cache.snapshot()[1] < 3 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIsNone(cache.snapshot()[0])
            with socket.create_connection(("127.0.0.1", receiver.port)) as client:
                client.sendall(frame())
                deadline = time.monotonic() + 2
                while cache.snapshot()[1] < 4 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIsNotNone(cache.snapshot()[0])
        finally:
            receiver.stop()


if __name__ == "__main__":
    unittest.main()
