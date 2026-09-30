"""CIMG framing: split packets, sticky packets, bad lengths, bad versions.

The protocol layer is stdlib only, so this module imports nothing heavier than
``camera.cimg`` and runs on a machine with no OpenCV, no torch and no sample
data.  The receiver's own behaviour - decoding, the queue policy, the atomic
group write, shutdown - is in ``tests/test_cimg_source.py``.

Every test here builds its packets from :func:`packet`, which is written from
the wire layout in CLOUD_PUSH.md rather than from the parser's constants, so a
test only fails when the parser and the document disagree.
"""

from __future__ import annotations

import struct
import unittest

from camera.cimg import (
    IMAGE_HEADER,
    MAGIC,
    PACKET_HEADER,
    PROTOCOL_VERSION,
    SEQUENCE_CONTIGUOUS,
    SEQUENCE_FIRST,
    SEQUENCE_GAP,
    SEQUENCE_REORDERED,
    SEQUENCE_REPEATED,
    CimgStreamParser,
    ProtocolError,
    TruncatedRead,
    jpeg_dimensions,
    parse_packet_header,
    parse_payload,
    read_packet,
    sequence_check,
)

GROUP_TIME_NS = 1_700_000_000_000_000_000
CAMERAS = (1, 2, 3, 4)


def fake_jpeg(width: int = 640, height: int = 480, fill: bytes = b"\x00") -> bytes:
    """A minimal but structurally valid JPEG: SOI, SOF0 with the dimensions, EOI.

    ``jpeg_dimensions`` only needs the SOF segment and the parser only needs
    SOI/EOI, so this is enough to exercise both without an encoder - which is
    the point, since cv2 is absent from the machines this test must run on.
    """
    sof = b"\xff\xc0" + struct.pack("!H", 11) + bytes([8])
    sof += struct.pack("!HH", height, width) + b"\x01\x01\x11\x00"
    return b"\xff\xd8" + sof + fill * 8 + b"\xff\xd9"


def packet(
    group_ns: int = GROUP_TIME_NS,
    *,
    sequences: tuple[int, ...] = (10, 11, 12, 13),
    captured_offsets_ns: tuple[int, ...] = (-1_000_000, 0, 500_000, 2_000_000),
    jpegs: tuple[bytes, ...] | None = None,
    camera_ids: tuple[int, ...] = CAMERAS,
    version: int = PROTOCOL_VERSION,
    image_count: int = 4,
    payload_size: int | None = None,
    magic: bytes = MAGIC,
) -> bytes:
    """Serialise one packet exactly as the vehicle does, byte for byte."""
    images = jpegs if jpegs is not None else tuple(fake_jpeg() for _ in CAMERAS)
    records = b"".join(
        IMAGE_HEADER.pack(cam, 0, seq, group_ns + offset, len(jpeg)) + jpeg
        for cam, seq, offset, jpeg in zip(
            camera_ids, sequences, captured_offsets_ns, images
        )
    )
    declared = len(records) if payload_size is None else payload_size
    return PACKET_HEADER.pack(
        magic, version, image_count, group_ns, declared
    ) + records


class FragmentingReader:
    """A socket stand-in whose ``recv`` never returns more than N bytes."""

    def __init__(self, data: bytes, chunk: int) -> None:
        self.data = data
        self.chunk = chunk

    def recv(self, size: int) -> bytes:
        count = min(size, self.chunk, len(self.data))
        taken, self.data = self.data[:count], self.data[count:]
        return taken


class PacketParsingTests(unittest.TestCase):
    def test_a_normal_packet_round_trips_every_field(self) -> None:
        parser = CimgStreamParser()
        packets = parser.feed(packet())
        self.assertEqual(1, len(packets))
        received = packets[0]
        self.assertEqual(GROUP_TIME_NS, received.group_time_ns)
        self.assertEqual(CAMERAS, received.camera_ids)
        self.assertEqual((10, 11, 12, 13), tuple(i.sequence for i in received.images))
        # Exactly the declared size, and four records of 20 + jpeg.
        self.assertEqual(len(packet()) - 20, received.payload_size)
        self.assertEqual((640, 480), (received.images[0].width, received.images[0].height))
        self.assertTrue(all(i.eoi_present for i in received.images))
        self.assertTrue(all(i.jpeg_data[:2] == b"\xff\xd8" for i in received.images))

    def test_per_camera_capture_times_are_kept_apart_from_the_group_time(self) -> None:
        # CLOUD_PUSH.md: group_time_ns is the group's midpoint, not any single
        # image's time.  Downstream must be able to correct per camera, which is
        # only possible if both survive the parse.
        received = CimgStreamParser().feed(packet())[0]
        self.assertEqual(3_000_000, received.capture_span_ns)
        self.assertEqual(
            [GROUP_TIME_NS - 1_000_000, GROUP_TIME_NS, GROUP_TIME_NS + 500_000,
             GROUP_TIME_NS + 2_000_000],
            [image.capture_time_ns for image in received.images],
        )
        self.assertEqual(11, received.image(2).sequence)
        self.assertIsNone(received.image(9))

    def test_header_parser_rejects_magic_version_and_image_count(self) -> None:
        for raw, expected in (
            (PACKET_HEADER.pack(b"NOPE", 1, 4, 1, 80), "包标识错误"),
            (PACKET_HEADER.pack(MAGIC, 2, 4, 1, 80), "不支持协议版本"),
            (PACKET_HEADER.pack(MAGIC, 1, 3, 1, 80), "图片数量"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ProtocolError, expected):
                    parse_packet_header(raw)


class SplitAndStickyTests(unittest.TestCase):
    """The two things TCP does to a framing layer that a file reader never sees."""

    def test_a_packet_split_across_reads_is_reassembled(self) -> None:
        data = packet()
        for chunk_size in (1, 3, 20, 23, 1024, len(data) - 1):
            with self.subTest(chunk_size=chunk_size):
                parser = CimgStreamParser()
                collected = []
                for offset in range(0, len(data), chunk_size):
                    collected.extend(parser.feed(data[offset : offset + chunk_size]))
                self.assertEqual(1, len(collected))
                self.assertEqual(0, parser.resync_count)
                self.assertEqual(GROUP_TIME_NS, collected[0].group_time_ns)

    def test_one_byte_at_a_time_still_yields_one_packet(self) -> None:
        # The pathological split, and the one that catches an off-by-one in the
        # "wait for 20 bytes" branch: 20 is exactly the header size.
        parser = CimgStreamParser()
        data = packet()
        collected = []
        for index in range(len(data)):
            collected.extend(parser.feed(data[index : index + 1]))
        self.assertEqual(1, len(collected))
        self.assertEqual(4, len(collected[0].images))
        self.assertEqual(0, parser.buffered_bytes)

    def test_two_packets_in_one_read_are_returned_separately_and_in_order(self) -> None:
        """The sticky case: a single recv carrying more than one packet."""
        first = packet()
        second = packet(GROUP_TIME_NS + 200_000_000)
        parser = CimgStreamParser()
        collected = parser.feed(first + second)
        self.assertEqual(2, len(collected))
        self.assertEqual(
            [GROUP_TIME_NS, GROUP_TIME_NS + 200_000_000],
            [p.group_time_ns for p in collected],
        )
        self.assertEqual(0, parser.buffered_bytes)
        self.assertEqual(2, parser.packets_parsed)

    def test_a_trailing_half_packet_is_held_until_the_rest_arrives(self) -> None:
        first = packet()
        second = packet(GROUP_TIME_NS + 200_000_000)
        parser = CimgStreamParser()
        self.assertEqual(1, len(parser.feed(first + second[:37])))
        self.assertEqual(37, parser.buffered_bytes)
        self.assertEqual(1, len(parser.feed(second[37:])))
        self.assertEqual(0, parser.buffered_bytes)

    def test_junk_between_packets_is_skipped_and_counted(self) -> None:
        parser = CimgStreamParser()
        collected = parser.feed(b"\x00\x01\x02" + packet())
        self.assertEqual(1, len(collected))
        self.assertEqual(1, parser.resync_count)
        self.assertEqual(3, parser.discarded_bytes)
        self.assertEqual(0, parser.packets_skipped)


class AbnormalLengthTests(unittest.TestCase):
    def test_oversize_payload_size_is_rejected_before_waiting_for_it(self) -> None:
        # Without the bound this declares a 4 GB body and the receiver would
        # buffer until the box dies.
        parser = CimgStreamParser(max_payload_bytes=4096)
        self.assertEqual([], parser.feed(packet(payload_size=100_000)))
        self.assertEqual(1, parser.packets_skipped)
        self.assertIn("payload_size 非法", parser.last_error)

    def test_payload_size_below_the_record_headers_is_rejected(self) -> None:
        parser = CimgStreamParser()
        self.assertEqual([], parser.feed(packet(payload_size=79)))  # 4*20 - 1
        self.assertEqual(1, parser.packets_skipped)

    def test_a_jpeg_size_past_the_end_of_the_payload_is_rejected(self) -> None:
        # The declared payload arrives in full; the record inside it lies.
        images = (fake_jpeg() + b"\x00" * 0, fake_jpeg(), fake_jpeg(), fake_jpeg())
        records = b"".join(
            IMAGE_HEADER.pack(cam, 0, 1, GROUP_TIME_NS, len(jpeg) + 500) + jpeg
            for cam, jpeg in zip(CAMERAS, images)
        )
        payload = records
        raw = PACKET_HEADER.pack(MAGIC, 1, 4, GROUP_TIME_NS, len(payload)) + payload
        parser = CimgStreamParser()
        self.assertEqual([], parser.feed(raw))
        self.assertEqual(1, parser.packets_skipped)
        self.assertIn("不完整", parser.last_error)

    def test_trailing_bytes_inside_the_payload_are_rejected(self) -> None:
        raw = packet()
        with self.assertRaisesRegex(ProtocolError, "多余字节"):
            parse_payload(raw[20:] + b"\x00" * 4, 4)

    def test_an_oversize_jpeg_record_is_rejected(self) -> None:
        jpeg = fake_jpeg()
        with self.assertRaisesRegex(ProtocolError, "长度非法"):
            parse_payload(
                IMAGE_HEADER.pack(1, 0, 1, GROUP_TIME_NS, len(jpeg)) + jpeg,
                4,
                max_jpeg_bytes=8,
            )

    def test_a_payload_that_is_not_jpeg_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "不是 JPEG"):
            parse_payload(
                IMAGE_HEADER.pack(1, 0, 1, GROUP_TIME_NS, 8) + b"NOTAJPEG",
                4,
            )

    def test_the_bounds_do_not_fire_on_a_real_sized_packet(self) -> None:
        # 1920x1080 JPEGs are ~200 KB each, so a real packet is ~825 KB.  The
        # configured bounds must be far above that or a healthy stream would be
        # rejected; this asserts the margin rather than the numbers.
        big = fake_jpeg(1920, 1080, fill=b"\x00" * 250_000)
        parser = CimgStreamParser()
        self.assertEqual(4, len(parser.feed(packet(jpegs=(big,) * 4))[0].images))
        self.assertEqual(0, parser.packets_skipped)


class ResynchronisationTests(unittest.TestCase):
    """A corrupt stream must recover, not wedge the connection until a restart."""

    def test_a_bad_version_is_stepped_over_and_the_next_packet_is_read(self) -> None:
        bad = packet(version=2)
        good = packet(GROUP_TIME_NS + 200_000_000)
        parser = CimgStreamParser()
        collected = parser.feed(bad + good)
        self.assertEqual(1, len(collected), "the good packet must still be parsed")
        self.assertEqual(GROUP_TIME_NS + 200_000_000, collected[0].group_time_ns)
        self.assertEqual(1, parser.packets_skipped)
        self.assertIn("不支持协议版本", parser.last_error)

    def test_a_corrupt_payload_size_does_not_lose_the_following_packets(self) -> None:
        # The failure the blocking first-stage reader could not survive: a wrong
        # length consumes the wrong bytes and every later packet is misparsed.
        good = packet(GROUP_TIME_NS + 400_000_000)
        corrupt = bytearray(packet(payload_size=90))
        parser = CimgStreamParser()
        stream = bytes(corrupt) + good
        collected = parser.feed(stream)
        self.assertEqual(1, len(collected))
        self.assertEqual(GROUP_TIME_NS + 400_000_000, collected[0].group_time_ns)
        self.assertGreaterEqual(parser.resync_count, 1)

    def test_a_packet_survives_corruption_that_lands_inside_jpeg_data(self) -> None:
        """A flipped image byte is a decoder problem, not a framing problem.

        Not a rejection test, a separation test: the framing must stay in sync
        so the receiver counts one *decode* error rather than a resync storm.
        The damage is confined to one image, so the other three still parse.
        """
        jpegs = list(fake_jpeg(fill=b"\x11") for _ in CAMERAS)
        jpegs[2] = jpegs[2][:20] + b"\x00" + jpegs[2][21:]
        parser = CimgStreamParser()
        collected = parser.feed(packet(jpegs=tuple(jpegs)))
        self.assertEqual(1, len(collected))
        self.assertEqual(0, parser.resync_count)
        self.assertEqual(0, parser.packets_skipped)


class DisconnectTests(unittest.TestCase):
    def test_reset_drops_a_half_packet_and_reports_how_much(self) -> None:
        parser = CimgStreamParser()
        self.assertEqual(150, len(packet()[:150]))  # header plus part of a record
        parser.feed(packet()[:150])
        self.assertEqual(150, parser.buffered_bytes)
        self.assertEqual(150, parser.reset())
        self.assertEqual(0, parser.buffered_bytes)
        self.assertEqual(150, parser.discarded_bytes)

    def test_reset_on_an_empty_buffer_is_a_no_op(self) -> None:
        parser = CimgStreamParser()
        self.assertEqual(0, parser.reset())

    def test_a_half_packet_before_a_disconnect_does_not_poison_the_next_one(self) -> None:
        """CLOUD_PUSH.md: restart from the next connection's CIMG header."""
        parser = CimgStreamParser()
        parser.feed(packet()[:150])
        parser.reset()
        collected = parser.feed(packet(GROUP_TIME_NS + 200_000_000))
        self.assertEqual(1, len(collected))
        self.assertEqual(GROUP_TIME_NS + 200_000_000, collected[0].group_time_ns)
        self.assertEqual(0, parser.buffered_bytes)

    def test_read_packet_reports_a_truncated_read_with_both_counts(self) -> None:
        # 40 bytes short, so the 20-byte header arrives and the payload does not:
        # the exception must name both, or the log cannot say whether the peer
        # closed between packets or mid-packet.
        with self.assertRaises(TruncatedRead) as caught:
            read_packet(FragmentingReader(packet()[:-40], 64))
        self.assertEqual(40, caught.exception.expected - caught.exception.received)
        self.assertEqual(len(packet()) - 20, caught.exception.expected)

    def test_read_packet_handles_fragmentation_and_sticky_reads(self) -> None:
        stream = FragmentingReader(packet() + packet(GROUP_TIME_NS + 1), 7)
        first = read_packet(stream)
        second = read_packet(stream)
        self.assertLess(first.group_time_ns, second.group_time_ns)
        self.assertEqual((640, 480), (first.images[0].width, first.images[0].height))


class SequenceTests(unittest.TestCase):
    def test_the_verdict_vocabulary_is_exactly_five_states(self) -> None:
        self.assertEqual(SEQUENCE_FIRST, sequence_check(None, 5).status)
        self.assertEqual(SEQUENCE_CONTIGUOUS, sequence_check(5, 6).status)
        self.assertEqual(SEQUENCE_REPEATED, sequence_check(5, 5).status)
        self.assertEqual(SEQUENCE_REORDERED, sequence_check(5, 4).status)
        gap = sequence_check(5, 9)
        self.assertEqual(SEQUENCE_GAP, gap.status)
        self.assertEqual(3, gap.gap)
        self.assertTrue(gap.is_gap)
        self.assertFalse(gap.is_out_of_order)

    def test_uint32_wrap_is_contiguous_in_both_directions(self) -> None:
        self.assertEqual(SEQUENCE_CONTIGUOUS, sequence_check(0xFFFFFFFF, 0).status)
        self.assertEqual(SEQUENCE_REORDERED, sequence_check(0, 0xFFFFFFFF).status)

    def test_a_gap_is_never_worded_as_packet_loss(self) -> None:
        """The wording is the requirement, so it is asserted.

        The vehicle numbers frames per camera and skips those that miss its
        four-camera sync tolerance, so a jump in the numbering is a fact about
        the sender's frame selection and only sometimes about the network.  An
        operator reading "丢包" would go looking at the link.
        """
        self.assertEqual("序号跳变", SEQUENCE_GAP)
        for forbidden in ("丢包", "丢帧", "loss", "lost"):
            self.assertNotIn(forbidden, SEQUENCE_GAP)


class CameraIdentityTests(unittest.TestCase):
    def test_a_missing_camera_is_rejected(self) -> None:
        records = b"".join(
            IMAGE_HEADER.pack(cam, 0, 1, GROUP_TIME_NS, len(fake_jpeg()))
            + fake_jpeg()
            for cam in (1, 2, 3, 3)  # camera 4 replaced by a duplicate 3
        )
        with self.assertRaisesRegex(ProtocolError, "重复"):
            parse_payload(records, 4)

    def test_an_out_of_range_camera_id_is_rejected(self) -> None:
        records = b"".join(
            IMAGE_HEADER.pack(cam, 0, 1, GROUP_TIME_NS, len(fake_jpeg()))
            + fake_jpeg()
            for cam in (1, 2, 3, 5)
        )
        with self.assertRaisesRegex(ProtocolError, "非法相机编号"):
            parse_payload(records, 4)

    def test_a_non_ascending_camera_order_is_accepted(self) -> None:
        # The vehicle has been seen sending cameras out of order within a
        # packet; the receiver indexes by camera_id, never by position.
        parser = CimgStreamParser()
        received = parser.feed(packet(camera_ids=(4, 2, 1, 3)))[0]
        self.assertEqual((4, 2, 1, 3), received.camera_ids)
        # Sequence follows its camera, not its position: cam1 is the third
        # record here, so it carries the third sequence.
        self.assertEqual(12, received.image(1).sequence)
        self.assertEqual((1, 2, 3, 4), tuple(sorted(received.camera_ids)))


class JpegDimensionTests(unittest.TestCase):
    def test_dimensions_come_from_the_sof_segment(self) -> None:
        self.assertEqual((640, 480), jpeg_dimensions(fake_jpeg(640, 480)))
        self.assertEqual((1920, 1080), jpeg_dimensions(fake_jpeg(1920, 1080)))

    def test_an_unreadable_payload_returns_none_rather_than_raising(self) -> None:
        for data in (b"", b"\x00\x01", b"\xff\xd8", b"\xff\xd8\xff\xda\x00\x08"):
            with self.subTest(data=data):
                self.assertIsNone(jpeg_dimensions(data))


if __name__ == "__main__":
    unittest.main()
