"""Frozen wire-protocol baselines for the vehicle-facing packets.

These tests exist to make the progressive refactor provably byte-compatible.
Every expected value below is written out independently of the production
encoder (literal struct format strings, literal hex), so a change to the
encoder that alters a single byte fails here instead of on the vehicle.

Two protocols are frozen:

  waypoint      <IBBIHH header + N x <dd (latitude, longitude)
                backend -> vehicle, UDP (address comes from settings)
  planned route <BBIIB header + M x <ff (longitude, latitude)
                vehicle -> backend, TCP :30006

Only pure encoding/decoding is exercised; no socket is opened for listening.
"""

from __future__ import annotations

import struct
import sys
import unittest

# The waypoint encoder needs neither a camera nor a model, so this file reaches
# it through the module that owns it rather than through ``app``, which imports
# cv2 for unrelated camera code.  See tests/_cv2_isolation.py for why that
# distinction is worth keeping.
from config.loader import AppConfig as Settings
from vehicle.planned_route import MAGIC as ROUTE_MAGIC
from vehicle.planned_route import RouteStreamParser, VehiclePlannedRouteAdapter
from waypoint.sender import WaypointSender

# --------------------------------------------------------------------------- #
# Frozen fixtures
# --------------------------------------------------------------------------- #

# <IBBIHH: magic, version, control, seq, vehicle_id, waypoint_count
WAYPOINT_HEADER = "31304d5101010000000001000200"
# 2 x <dd: (28.0, 113.0), (28.0001, 113.0002)
WAYPOINT_BODY = (
    "0000000000003c400000000000405c40"
    "c7bab88d06003c40645ddc4603405c40"
)
WAYPOINT_PACKET = WAYPOINT_HEADER + WAYPOINT_BODY
WAYPOINT_HEADER_SIZE = 14

# <BBIIB: magic_hi, magic_lo, length, count, reached
# + 2 x <ff (113.0, 28.0), (113.5, 28.25)  -- both exactly representable in
# float32, so the frame round-trips without rounding ambiguity.
ROUTE_FRAME = "f1101b00000002000000000000e2420000e0410000e3420000e241"
ROUTE_HEADER_SIZE = 11

# Header field offsets, kept explicit so a layout change cannot pass silently.
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_CONTROL = 5
_OFF_SEQ = 6
_OFF_VEHICLE_ID = 10
_OFF_COUNT = 12


class _RecordingSocket:
    """Captures datagrams instead of sending them."""

    def __init__(self) -> None:
        self.packets: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, packet: bytes, address: tuple[str, int]) -> int:
        self.packets.append((bytes(packet), address))
        return len(packet)


def _sender(*, magic: int = 0x514D3031, vehicle_id: int = 1) -> tuple:
    settings = Settings(
        magic=magic,
        vehicle_id=vehicle_id,
        allow_unconfigured_udp=False,
        waypoint_host="127.0.0.1",
        waypoint_port=7010,
    )
    sender = WaypointSender(settings)
    # Drop the real UDP socket the constructor opened: these tests never send.
    sender.sock.close()
    recorder = _RecordingSocket()
    sender.sock = recorder
    return sender, recorder


class WaypointGoldenPacketTests(unittest.TestCase):
    def test_packet_is_byte_identical_to_frozen_baseline(self):
        sender, recorder = _sender()
        n = sender.send(1, [28.0, 28.0001], [113.0, 113.0002])
        self.assertEqual(n, 2)
        self.assertEqual(len(recorder.packets), 1)
        packet, address = recorder.packets[0]
        self.assertEqual(packet.hex(), WAYPOINT_PACKET)
        self.assertEqual(address, ("127.0.0.1", 7010))

    def test_header_layout_is_little_endian_at_fixed_offsets(self):
        sender, recorder = _sender()
        sender.send(1, [28.0, 28.0001], [113.0, 113.0002])
        packet = recorder.packets[0][0]
        magic, version, control, seq, vehicle_id, count = struct.unpack_from(
            "<IBBIHH", packet
        )
        self.assertEqual(magic, 0x514D3031)
        self.assertEqual(version, 1)
        self.assertEqual(control, 1)
        self.assertEqual(seq, 0)
        self.assertEqual(vehicle_id, 1)
        self.assertEqual(count, 2)
        # Lat/lon pairs follow the header, latitude first.
        self.assertEqual(
            struct.unpack_from("<dd", packet, WAYPOINT_HEADER_SIZE), (28.0, 113.0)
        )
        self.assertEqual(
            struct.unpack_from("<dd", packet, WAYPOINT_HEADER_SIZE + 16),
            (28.0001, 113.0002),
        )

    def test_packet_length_is_header_plus_sixteen_bytes_per_waypoint(self):
        for count in (0, 1, 2, 64):
            with self.subTest(count=count):
                sender, recorder = _sender()
                lat = [28.0 + i for i in range(count)]
                lon = [113.0 + i for i in range(count)]
                n = sender.send(0, lat, lon)
                self.assertEqual(n, count)
                packet = recorder.packets[0][0]
                self.assertEqual(
                    len(packet), WAYPOINT_HEADER_SIZE + 16 * count
                )

    def test_empty_trajectory_encodes_zero_count_and_no_body(self):
        # This is the release frame: when the trajectory trim expires with
        # nothing left to execute, the pipeline sends control=0 with count=0
        # rather than staying silent, so the vehicle stops holding the last
        # control=1 it received.
        sender, recorder = _sender()
        n = sender.send(0, [], [])
        self.assertEqual(n, 0)
        packet = recorder.packets[0][0]
        self.assertEqual(len(packet), WAYPOINT_HEADER_SIZE)
        self.assertEqual(
            struct.unpack_from("<H", packet, _OFF_COUNT)[0], 0
        )

    def test_sequence_increments_per_packet(self):
        sender, recorder = _sender()
        for _ in range(3):
            sender.send(0, [28.0], [113.0])
        sequences = [
            struct.unpack_from("<I", packet, _OFF_SEQ)[0]
            for packet, _ in recorder.packets
        ]
        self.assertEqual(sequences, [0, 1, 2])
        self.assertEqual(sender.last_seq, 2)

    def test_control_flag_is_a_single_byte_and_masked(self):
        sender, recorder = _sender()
        sender.send(1, [28.0], [113.0])
        self.assertEqual(recorder.packets[0][0][_OFF_CONTROL], 1)
        sender.send(0, [28.0], [113.0])
        self.assertEqual(recorder.packets[1][0][_OFF_CONTROL], 0)
        # The control byte must not bleed into adjacent fields.
        self.assertEqual(recorder.packets[0][0][_OFF_VERSION], 1)
        self.assertEqual(
            struct.unpack_from("<H", recorder.packets[0][0], _OFF_COUNT)[0], 1
        )

    def test_unconfigured_identifiers_are_refused(self):
        sender, _ = _sender(magic=0, vehicle_id=0)
        with self.assertRaises(RuntimeError):
            sender.send(0, [28.0], [113.0])


class PlannedRouteGoldenFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = RouteStreamParser()

    def test_frame_is_byte_identical_to_frozen_baseline(self):
        frame = bytes.fromhex(ROUTE_FRAME)
        updates = self.parser.feed(frame)
        self.assertEqual(len(updates), 1)
        route = updates[0]
        self.assertIsNotNone(route)
        self.assertEqual(route.coordinate_frame, "geographic_lon_lat_degrees")
        # Point order on the wire is (longitude, latitude).
        self.assertEqual(
            [p.coordinates for p in route.points], [(113.0, 28.0), (113.5, 28.25)]
        )

    def test_header_layout_matches_frozen_offsets(self):
        frame = bytes.fromhex(ROUTE_FRAME)
        high, low, length, count, reached = struct.unpack_from("<BBIIB", frame)
        self.assertEqual(bytes((high, low)), ROUTE_MAGIC)
        self.assertEqual(length, len(frame))
        self.assertEqual(count, 2)
        self.assertEqual(reached, 0)
        self.assertEqual(length, ROUTE_HEADER_SIZE + 8 * count)

    def test_fragmented_frame_survives_arbitrary_chunk_boundaries(self):
        frame = bytes.fromhex(ROUTE_FRAME)
        for split in range(1, len(frame)):
            with self.subTest(split=split):
                parser = RouteStreamParser()
                self.assertEqual(parser.feed(frame[:split]), [])
                updates = parser.feed(frame[split:])
                self.assertEqual(len(updates), 1)
                self.assertIsNotNone(updates[0])

    def test_coalesced_frames_are_both_decoded(self):
        parser = RouteStreamParser()
        updates = parser.feed(bytes.fromhex(ROUTE_FRAME) * 2)
        self.assertEqual(len(updates), 2)
        self.assertTrue(all(u is not None for u in updates))

    def test_completed_flag_clears_and_empty_route_clears(self):
        frame = bytes.fromhex(ROUTE_FRAME)
        # reached=1 must yield None so the receiver clears the cached route.
        completed = bytearray(frame)
        struct.pack_into("<B", completed, 10, 1)
        self.assertEqual(RouteStreamParser().feed(bytes(completed)), [None])
        # A zero-point route likewise decodes to None.
        empty = struct.pack("<BBIIB", 0xF1, 0x10, ROUTE_HEADER_SIZE, 0, 0)
        self.assertEqual(RouteStreamParser().feed(empty), [None])

    def test_replacement_decodes_to_a_new_route_object(self):
        adapter = VehiclePlannedRouteAdapter()
        first = adapter.decode(bytes.fromhex(ROUTE_FRAME))
        longer = (
            struct.pack("<BBIIB", 0xF1, 0x10, ROUTE_HEADER_SIZE + 8 * 3, 3, 0)
            + struct.pack("<ff", 113.0, 28.0)
            + struct.pack("<ff", 113.5, 28.25)
            + struct.pack("<ff", 114.0, 28.5)
        )
        second = adapter.decode(longer)
        self.assertEqual(len(first.points), 2)
        self.assertEqual(len(second.points), 3)
        self.assertIsNot(first, second)

    def test_malformed_frames_are_rejected(self):
        frame = bytes.fromhex(ROUTE_FRAME)
        cases = {
            "bad_magic": b"\x00\x00" + frame[2:],
            "truncated": frame[:-1],
            "count_mismatch": struct.pack("<BBIIB", 0xF1, 0x10, len(frame), 99, 0)
            + frame[ROUTE_HEADER_SIZE:],
            "bad_reached": frame[:10] + b"\x02" + frame[11:],
            "out_of_range": struct.pack(
                "<BBIIB", 0xF1, 0x10, ROUTE_HEADER_SIZE + 8, 1, 0
            )
            + struct.pack("<ff", 200.0, 28.0),
            "non_finite": struct.pack(
                "<BBIIB", 0xF1, 0x10, ROUTE_HEADER_SIZE + 8, 1, 0
            )
            + struct.pack("<ff", float("nan"), 28.0),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    VehiclePlannedRouteAdapter().decode(payload)


if __name__ == "__main__":
    unittest.main()
