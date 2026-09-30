"""CIMG wire protocol: four JPEGs pushed by the vehicle over one TCP stream.

Stdlib only.  Nothing here opens a socket, imports numpy,
torch or cv2, or knows what a frame buffer is, so the whole protocol can be
parsed and tested in a bare interpreter - which is what ``tests/test_cimg_protocol.py``
does.  The receiver that owns the socket lives in ``camera.cimg_source`` and the
window assembly above it in ``camera.sync``; the dependency runs one way.

Transport contract (F:\\CLOUD_PUSH.md)
    Every packet is a fixed 20-byte header followed by ``payload_size`` bytes of
    four image records, each a 20-byte image header followed by its JPEG::

        packet  = header(20) + record(1) + record(2) + record(3) + record(4)
        record  = image_header(20) + jpeg(jpeg_size)
        payload_size = sum(20 + jpeg_size)

    All multi-byte integers are network byte order (big endian).

Why this module is a stateful parser and not a ``recv_exact`` loop
    TCP is a byte stream: one ``recv()`` may return half a packet or three
    packets.  The first-stage validation tool reads with a blocking
    ``recv_exact``, which handles both of those but cannot recover from a
    stream that has lost its framing: a wrong ``payload_size`` consumes the
    wrong number of bytes and every later packet is misparsed.  This parser
    keeps the bytes it has, extracts every complete packet, and on a bad magic
    or a bad header re-seeks the next ``CIMG`` marker instead of giving up on
    the connection - a mobile link dropping a connection over one corrupt
    packet is exactly the manual-restart fragility the CIMG link is meant to
    remove.  Every resync is counted, never silent.

What is a protocol error and what is not
    Sizes and the outer framing are protocol errors: they mean the bytes
    cannot be trusted.  The JPEG payload itself is validated only as far as
    SOI/SOF scanning (``jpeg_dimensions``), which yields diagnostics; whether
    the image really decodes is the decoder's verdict, reported separately by
    the receiver.  Keeping those two counters apart is what lets an operator
    tell a broken link from a broken encoder on the vehicle.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b"CIMG"
PROTOCOL_VERSION = 1
EXPECTED_IMAGE_COUNT = 4
EXPECTED_CAMERA_IDS = (1, 2, 3, 4)

# ``!4sHHQI`` = magic, version, image_count, group_time_ns, payload_size.
PACKET_HEADER = struct.Struct("!4sHHQI")
# ``!HHIQI`` = camera_id, reserved, sequence, capture_time_ns, jpeg_size.
IMAGE_HEADER = struct.Struct("!HHIQI")
PACKET_HEADER_SIZE = PACKET_HEADER.size  # 20
IMAGE_HEADER_SIZE = IMAGE_HEADER.size  # 20

# Bounds that keep a corrupt or hostile length field from allocating the box
# out of memory.  A real packet is ~825 KB (four 1080p JPEGs at quality 80);
# these are two orders of magnitude above that, so they never fire on a healthy
# stream and always fire before a hostile one can hurt.
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_JPEG_BYTES = 16 * 1024 * 1024

# Sequence comparison verdicts.  ``SEQUENCE_GAP`` is deliberately *not* called
# packet loss: the vehicle numbers frames per camera and skips those that miss
# its four-camera sync tolerance, so a gap is a fact about the sender's frame
# selection and only sometimes about the network.  See ``sequence_check``.
SEQUENCE_FIRST = "首次"
SEQUENCE_CONTIGUOUS = "连续"
SEQUENCE_REPEATED = "重复"
SEQUENCE_GAP = "序号跳变"
SEQUENCE_REORDERED = "乱序"

_UINT32_MASK = 0xFFFFFFFF


class ProtocolError(ValueError):
    """Bytes that cannot be safely parsed as CIMG."""


class TruncatedRead(EOFError):
    """A read ended in the middle of something that is never resumed.

    Raised by the receiver on disconnect, and by ``read_packet``'s blocking
    helper.  Carries how much of the expected bytes arrived so the log can say
    whether the peer closed cleanly between packets or dropped mid-packet.
    """

    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"需要 {expected} 字节，实际只收到 {received} 字节")
        self.expected = expected
        self.received = received


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CimgImage:
    """One JPEG belonging to one camera, exactly as it arrived on the wire.

    ``capture_time_ns`` is the *vehicle's* acquisition timestamp for this
    single image, ``sequence`` is that camera's frame number.  Both are kept
    verbatim: the four images of a packet share a ``group_time_ns``, but they
    are not simultaneous, and downstream code is allowed to correct per camera
    on the individual timestamps.
    """

    camera_id: int
    sequence: int
    capture_time_ns: int
    jpeg_size: int
    jpeg_data: bytes = field(repr=False, compare=False, default=b"")
    #: From the SOF scan.  Diagnostics only - ``None`` when the scan failed to
    #: find one, which is not by itself a reason to reject the image.
    width: int | None = None
    height: int | None = None
    #: Whether the payload ended with EOI.  Reported, not enforced.
    eoi_present: bool = True


@dataclass(frozen=True)
class CimgPacket:
    """One received four-camera group.

    ``group_time_ns`` is the vehicle's unified UTC timestamp for the group
    (midpoint of the four capture times, see CLOUD_PUSH.md); ``received_ns`` is
    filled in by the receiver when the last byte of the packet arrived, and is
    the only field on the server's own clock - the difference between the two
    is the measured vehicle-to-server clock offset plus link latency.
    """

    group_time_ns: int
    payload_size: int
    images: tuple[CimgImage, ...]
    received_ns: int = 0

    @property
    def camera_ids(self) -> tuple[int, ...]:
        return tuple(image.camera_id for image in self.images)

    def image(self, camera_id: int) -> CimgImage | None:
        for image in self.images:
            if image.camera_id == camera_id:
                return image
        return None

    @property
    def capture_span_ns(self) -> int:
        """Newest minus oldest acquisition time in the group.

        The vehicle only sends a group whose span is within its
        ``cloud_sync_tolerance_ms``, so a large value here means the packet
        was assembled by something that does not honour that contract.
        """
        if not self.images:
            return 0
        times = [image.capture_time_ns for image in self.images]
        return max(times) - min(times)


# --------------------------------------------------------------------------- #
# JPEG framing scan
# --------------------------------------------------------------------------- #

_SOF_MARKERS = frozenset(
    {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
)


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` from the SOF segment, or ``None``.

    Best effort by design.  The vehicle encodes with a plain JPEG encoder, so
    this always answers on real data, but a payload the scanner cannot read is
    handed to the decoder rather than rejected here: the decoder is the
    authority on decodability, and refusing an image over a marker-scan
    limitation would turn a cosmetic difference into a total outage.
    """
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    pos = 2
    size = len(data)
    while pos < size:
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < size and data[pos] == 0xFF:
            pos += 1
        if pos >= size:
            return None
        marker = data[pos]
        pos += 1
        if marker == 0xDA:  # start of scan: no more headers
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # standalone markers
            continue
        if pos + 2 > size:
            return None
        segment_length = struct.unpack_from("!H", data, pos)[0]
        if segment_length < 2 or pos + segment_length > size:
            return None
        if marker in _SOF_MARKERS:
            if segment_length < 7:
                return None
            height, width = struct.unpack_from("!HH", data, pos + 3)
            if width <= 0 or height <= 0:
                return None
            return width, height
        pos += segment_length
    return None


# --------------------------------------------------------------------------- #
# Sequence numbering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SequenceCheck:
    """One camera's sequence comparison against its previous packet.

    ``gap`` counts how many numbers between the two are unaccounted for.  It is
    information, not a verdict: the caller reports ``status`` and never turns a
    gap into the word "loss".
    """

    status: str
    gap: int = 0

    @property
    def is_gap(self) -> bool:
        return self.status == SEQUENCE_GAP

    @property
    def is_out_of_order(self) -> bool:
        return self.status == SEQUENCE_REORDERED


def sequence_check(previous: int | None, current: int) -> SequenceCheck:
    """Compare two uint32 sequence numbers from the same camera.

    Handles wrap-around (``0xFFFFFFFF`` -> ``0`` is contiguous).  A backward
    step is reported as ``SEQUENCE_REORDERED`` rather than an error: the four
    cameras are numbered independently and the vehicle has been observed
    sending a lower number on one camera than on the previous group.
    """
    if previous is None:
        return SequenceCheck(SEQUENCE_FIRST)
    delta = (int(current) - int(previous)) & _UINT32_MASK
    if delta == 0:
        return SequenceCheck(SEQUENCE_REPEATED)
    if delta == 1:
        return SequenceCheck(SEQUENCE_CONTIGUOUS)
    if delta < 0x80000000:
        return SequenceCheck(SEQUENCE_GAP, delta - 1)
    return SequenceCheck(SEQUENCE_REORDERED)


# --------------------------------------------------------------------------- #
# Packet parsing
# --------------------------------------------------------------------------- #


def parse_payload(
    payload: bytes,
    image_count: int,
    *,
    max_jpeg_bytes: int = DEFAULT_MAX_JPEG_BYTES,
    expected_camera_ids: tuple[int, ...] = EXPECTED_CAMERA_IDS,
) -> tuple[CimgImage, ...]:
    """Split one ``payload_size``-byte payload into its image records.

    Every check here is a length or an identity check, which is what makes a
    desynchronised parser detectable: random bytes essentially never produce
    exactly the expected set of camera ids with consistent record lengths.
    """
    records: list[CimgImage] = []
    offset = 0
    seen: set[int] = set()
    total = len(payload)
    allowed = set(expected_camera_ids)

    for index in range(image_count):
        if total - offset < IMAGE_HEADER_SIZE:
            raise ProtocolError(f"第 {index + 1} 条图片记录头不完整")
        (
            camera_id,
            _reserved,
            sequence,
            capture_time_ns,
            jpeg_size,
        ) = IMAGE_HEADER.unpack_from(payload, offset)
        offset += IMAGE_HEADER_SIZE
        if camera_id not in allowed:
            raise ProtocolError(
                f"非法相机编号 {camera_id}（应为 {sorted(allowed)}）"
            )
        if camera_id in seen:
            raise ProtocolError(f"相机编号 {camera_id} 重复")
        if jpeg_size <= 0 or jpeg_size > max_jpeg_bytes:
            raise ProtocolError(f"相机 {camera_id} JPEG 长度非法: {jpeg_size}")
        if jpeg_size > total - offset:
            raise ProtocolError(
                f"相机 {camera_id} JPEG 不完整: 声明 {jpeg_size} 字节，"
                f"剩余 {total - offset} 字节"
            )
        jpeg = payload[offset : offset + jpeg_size]
        offset += jpeg_size
        if jpeg[:2] != b"\xff\xd8":
            raise ProtocolError(f"相机 {camera_id} 数据不是 JPEG（缺少 SOI）")
        dimensions = jpeg_dimensions(jpeg)
        records.append(
            CimgImage(
                camera_id=int(camera_id),
                sequence=int(sequence),
                capture_time_ns=int(capture_time_ns),
                jpeg_size=int(jpeg_size),
                jpeg_data=jpeg,
                width=dimensions[0] if dimensions else None,
                height=dimensions[1] if dimensions else None,
                eoi_present=jpeg[-2:] == b"\xff\xd9",
            )
        )
        seen.add(int(camera_id))

    if offset != total:
        raise ProtocolError(f"负载尾部存在 {total - offset} 个多余字节")
    if seen != allowed:
        raise ProtocolError(
            f"相机集合不完整: 收到 {sorted(seen)}，应为 {sorted(allowed)}"
        )
    return tuple(records)


def parse_packet_header(raw: bytes) -> tuple[int, int, int, int]:
    """``(version, image_count, group_time_ns, payload_size)`` from 20 bytes."""
    magic, version, image_count, group_time_ns, payload_size = (
        PACKET_HEADER.unpack(raw)
    )
    if magic != MAGIC:
        raise ProtocolError(f"包标识错误: {magic!r}，期望 {MAGIC!r}")
    if version != PROTOCOL_VERSION:
        # A newer vehicle must be rejected loudly rather than misparsed: the
        # record layout is versioned and a v2 body would desynchronise us.
        raise ProtocolError(
            f"不支持协议版本 {version}，当前只接受版本 {PROTOCOL_VERSION}"
        )
    if image_count != EXPECTED_IMAGE_COUNT:
        raise ProtocolError(
            f"图片数量为 {image_count}，协议要求固定为 {EXPECTED_IMAGE_COUNT}"
        )
    return int(version), int(image_count), int(group_time_ns), int(payload_size)


def read_packet(
    conn,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_jpeg_bytes: int = DEFAULT_MAX_JPEG_BYTES,
    expected_camera_ids: tuple[int, ...] = EXPECTED_CAMERA_IDS,
    received_ns: int = 0,
) -> CimgPacket:
    """Blocking read of one packet from a socket-like object.

    Kept for tools, harnesses and tests that want the simple shape; the live
    receiver uses :class:`CimgStreamParser` instead, which can resynchronise.
    """
    raw_header = _recv_exact(conn, PACKET_HEADER_SIZE)
    _, image_count, group_time_ns, payload_size = parse_packet_header(raw_header)
    minimum = image_count * IMAGE_HEADER_SIZE
    if payload_size < minimum or payload_size > max_payload_bytes:
        raise ProtocolError(
            f"payload_size 非法: {payload_size}（允许 {minimum}..{max_payload_bytes}）"
        )
    payload = _recv_exact(conn, payload_size)
    return CimgPacket(
        group_time_ns=group_time_ns,
        payload_size=payload_size,
        images=parse_payload(
            payload,
            image_count,
            max_jpeg_bytes=max_jpeg_bytes,
            expected_camera_ids=expected_camera_ids,
        ),
        received_ns=received_ns,
    )


def _recv_exact(conn, size: int) -> bytes:
    """Read exactly ``size`` bytes, tolerating TCP fragmentation."""
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise TruncatedRead(size, len(data))
        data.extend(chunk)
    return bytes(data)


# --------------------------------------------------------------------------- #
# Incremental framing
# --------------------------------------------------------------------------- #


class CimgStreamParser:
    """Turn a byte stream into complete packets, resynchronising when it must.

    The parser owns a buffer and makes no I/O.  Feed it whatever ``recv()``
    returned; it returns every packet that is now complete and keeps the rest.
    One read may complete zero, one or many packets, which is the whole
    difference between a stream and a datagram and the reason this class exists.

    Resynchronisation is deliberate and counted.  A stream that lost framing
    cannot be repaired by trusting the next length field, so the parser:
      * seeks the next ``CIMG`` when the buffer does not start with one;
      * steps past a header it cannot accept (bad version, impossible size) and
        seeks again from the following byte;
      * reports how many bytes it skipped and how many packets it accepted, so
        "the link is corrupt" is a number in /health rather than a guess.
    """

    def __init__(
        self,
        *,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_jpeg_bytes: int = DEFAULT_MAX_JPEG_BYTES,
        expected_camera_ids: tuple[int, ...] = EXPECTED_CAMERA_IDS,
    ) -> None:
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_jpeg_bytes = int(max_jpeg_bytes)
        self.expected_camera_ids = tuple(expected_camera_ids)
        self.image_count = len(self.expected_camera_ids)
        self._buffer = bytearray()
        self._packets_parsed = 0
        self._packets_skipped = 0
        self._bytes_discarded = 0
        self._resync_count = 0
        self._last_error: str | None = None

    # -- introspection ---------------------------------------------------- #

    @property
    def buffered_bytes(self) -> int:
        """Bytes held for a packet that has not fully arrived."""
        return len(self._buffer)

    @property
    def packets_parsed(self) -> int:
        return self._packets_parsed

    @property
    def packets_skipped(self) -> int:
        """Headers that were rejected whole (bad version, impossible size)."""
        return self._packets_skipped

    @property
    def resync_count(self) -> int:
        """Times the parser had to seek the next ``CIMG`` marker."""
        return self._resync_count

    @property
    def discarded_bytes(self) -> int:
        return self._bytes_discarded

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # -- lifecycle -------------------------------------------------------- #

    def reset(self) -> int:
        """Drop a partially received packet; returns how many bytes went.

        Called on disconnect.  CLOUD_PUSH.md is explicit that a partial packet
        is abandoned and parsing restarts from the next connection's header -
        the bytes held here belong to a packet that will never be completed.
        """
        discarded = len(self._buffer)
        self._buffer.clear()
        if discarded:
            self._bytes_discarded += discarded
        return discarded

    def feed(self, chunk: bytes) -> list[CimgPacket]:
        """Add ``chunk`` and return every packet now complete, oldest first.

        ``_next_packet`` returning None means "no packet this pass" for two
        different reasons - the rest of the packet has not arrived, or something
        was rejected and the parser wants another pass from a new position.
        Those are told apart by whether the buffer shrank.  Breaking on both
        would leave the *rest of the same read* unprocessed, so a single corrupt
        header would hide every good packet behind it until the next ``recv``
        - and if the vehicle then went quiet, forever.
        """
        if chunk:
            self._buffer.extend(chunk)
        packets: list[CimgPacket] = []
        while True:
            buffered_before = len(self._buffer)
            packet = self._next_packet()
            if packet is not None:
                packets.append(packet)
                continue
            if len(self._buffer) >= buffered_before:
                break  # nothing was discarded: this read is fully consumed
        return packets

    # -- the loop body ---------------------------------------------------- #

    def _next_packet(self) -> CimgPacket | None:
        # 1. Frame on the magic.  Anything before the first CIMG cannot be
        #    trusted, and a trailing 3-byte fragment may be the start of one.
        if len(self._buffer) < PACKET_HEADER_SIZE:
            if len(self._buffer) >= len(MAGIC):
                self._drop_until_magic(keep_trailing=True)
            return None
        if bytes(self._buffer[:4]) != MAGIC:
            self._drop_until_magic(keep_trailing=True)
            if len(self._buffer) < PACKET_HEADER_SIZE:
                return None

        # 2. Header.  A rejection here means the payload length is unknown, so
        #    the only safe move is to step past this magic and seek the next.
        try:
            _, image_count, group_time_ns, payload_size = parse_packet_header(
                bytes(self._buffer[:PACKET_HEADER_SIZE])
            )
        except ProtocolError as exc:
            self._note_rejected(exc)
            del self._buffer[: len(MAGIC)]
            return None

        minimum = image_count * IMAGE_HEADER_SIZE
        if payload_size < minimum or payload_size > self.max_payload_bytes:
            self._note_rejected(
                ProtocolError(
                    f"payload_size 非法: {payload_size}"
                    f"（允许 {minimum}..{self.max_payload_bytes}）"
                )
            )
            del self._buffer[: len(MAGIC)]
            return None

        # 3. Body.  Incomplete is not an error - that is an ordinary split.
        total = PACKET_HEADER_SIZE + payload_size
        if len(self._buffer) < total:
            return None

        payload = bytes(self._buffer[PACKET_HEADER_SIZE:total])
        try:
            images = parse_payload(
                payload,
                image_count,
                max_jpeg_bytes=self.max_jpeg_bytes,
                expected_camera_ids=self.expected_camera_ids,
            )
        except ProtocolError as exc:
            self._note_rejected(exc)
            del self._buffer[: len(MAGIC)]
            return None

        del self._buffer[:total]
        self._packets_parsed += 1
        return CimgPacket(
            group_time_ns=group_time_ns,
            payload_size=payload_size,
            images=images,
        )

    def _drop_until_magic(self, *, keep_trailing: bool) -> None:
        """Discard bytes before the next ``CIMG``, keeping a partial one."""
        index = self._buffer.find(MAGIC)
        if index < 0:
            # No marker: keep only the tail that could still become one.
            keep = len(MAGIC) - 1 if keep_trailing else 0
            drop = max(0, len(self._buffer) - keep)
            if drop:
                del self._buffer[:drop]
                self._bytes_discarded += drop
                self._resync_count += 1
            return
        if index:
            del self._buffer[:index]
            self._bytes_discarded += index
            self._resync_count += 1

    def _note_rejected(self, exc: ProtocolError) -> None:
        self._packets_skipped += 1
        self._last_error = str(exc)
        self._resync_count += 1


__all__ = [
    "DEFAULT_MAX_JPEG_BYTES",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "EXPECTED_CAMERA_IDS",
    "EXPECTED_IMAGE_COUNT",
    "IMAGE_HEADER",
    "IMAGE_HEADER_SIZE",
    "MAGIC",
    "PACKET_HEADER",
    "PACKET_HEADER_SIZE",
    "PROTOCOL_VERSION",
    "SEQUENCE_CONTIGUOUS",
    "SEQUENCE_FIRST",
    "SEQUENCE_GAP",
    "SEQUENCE_REORDERED",
    "SEQUENCE_REPEATED",
    "CimgImage",
    "CimgPacket",
    "CimgStreamParser",
    "ProtocolError",
    "SequenceCheck",
    "TruncatedRead",
    "jpeg_dimensions",
    "parse_packet_header",
    "parse_payload",
    "read_packet",
    "sequence_check",
]
