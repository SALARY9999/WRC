"""The CIMG receiver: decoding, the queue policy, atomicity, shutdown.

Two levels here, and the split matters:

* **Ingest-level** tests drive ``CimgCameraSource.ingest_bytes`` with an
  injected decoder.  They cover framing-to-buffer behaviour - camera mapping,
  the group-atomic write, the bounded queue's drop-oldest policy, backlog
  collapse, decode and timestamp failures - deterministically and with no
  socket, no OpenCV and no sleeping.
* **Socket-level** tests bind a real listener on an ephemeral port and connect
  a real client, because the things they assert only exist in the socket path:
  that the bind happens on the reader thread and not in ``__init__``, that a
  disconnect abandons a half packet and a reconnect resumes, and that
  ``stop()`` is prompt and releases the port.

The decoder is injected rather than stubbed because the interesting behaviour
is what the receiver does *around* a decode - drop the whole group, count the
error, keep receiving - not what a decoder does.  ``camera.cimg_source``
imports cv2 lazily for exactly this reason; ``camera.sync`` (needed here for the
real ``FrameBuffer``) does not, so it is imported under ``cv2_stub()``.
"""

from __future__ import annotations

import socket
import os
import struct
import sys
import threading
import time
import unittest
import unittest.mock

import numpy as np

from _cv2_isolation import cv2_stub


def _fake_resize(frame, size, interpolation=None):
    """Nearest-neighbour resize, enough to prove shape and channel handling."""
    width, height = int(size[0]), int(size[1])
    rows = (np.arange(height) * frame.shape[0] // height).clip(0, frame.shape[0] - 1)
    cols = (np.arange(width) * frame.shape[1] // width).clip(0, frame.shape[1] - 1)
    return frame[rows][:, cols]


def _fake_cvt_color(frame, code):
    return frame[:, :, ::-1] if code == 4 else frame  # 4 = COLOR_BGR2RGB


# ``camera.sync`` -> ``import cv2``.  Stub, import, un-stub: the stub must not
# outlive this block, or a later test module sees an empty cv2 instead of
# ``ModuleNotFoundError`` (see tests/_cv2_isolation.py).  Every import of that
# module belongs *inside* the block - a ``from camera.sync import ...`` below it
# would re-execute the module against an absent cv2 and raise, which is exactly
# what an earlier version of this file did.
#
# Where cv2 is genuinely absent the stub gets the two functions the window
# builder calls, so the integration test below can exercise the real
# ``_build`` path (BGR -> RGB conversion, camera-index ordering) on a developer
# machine.  Where cv2 is installed the real one is used and nothing is patched.
with cv2_stub() as _CV2_STUBBED:
    if _CV2_STUBBED:
        import cv2 as _cv2_module

        _cv2_module.INTER_LINEAR = 1
        _cv2_module.COLOR_BGR2RGB = 4
        _cv2_module.resize = _fake_resize
        _cv2_module.cvtColor = _fake_cvt_color
    from camera.sync import CameraSynchronizer, CameraWindowProvider, FrameBuffer
    # Same class object ``CameraWindowProvider`` checks.  The stub evicts
    # ``camera.cimg_source`` on the way out, and the import below reloads it;
    # a config built from that second copy would fail the provider's type check.
    from camera.cimg_source import CimgConfig as _ProviderCimgConfig

from camera.cimg import IMAGE_HEADER, MAGIC, PACKET_HEADER
from camera.cimg_source import CimgCameraSource, CimgConfig
from camera.models import FrameEntry
from time_sync import TimestampManager

CAMERAS = (1, 2, 3, 4)
# Sixty seconds ago, aligned to a second.  Camera time is capture_time_ns, so
# a fixed instant in 2027 is rejected once it is no longer "now", and a test
# that adds a few seconds of group spacing still stays inside the future bound.
BASE_NS = (time.time_ns() // 1_000_000_000 - 60) * 1_000_000_000


# --------------------------------------------------------------------------- #
# Wire helpers (written from CLOUD_PUSH.md, not from the parser's constants)
# --------------------------------------------------------------------------- #


def jpeg_bytes(camera_id: int, sequence: int, size: int = 400) -> bytes:
    """A JPEG-ish payload whose bytes identify the camera and the sequence.

    The injected decoder below reads the camera id back out of it, so a frame
    arriving in the wrong buffer is detectable rather than merely suspicious.
    """
    body = bytes([camera_id]) + struct.pack("!I", sequence)
    return b"\xff\xd8" + body + b"\x00" * max(0, size - len(body) - 4) + b"\xff\xd9"


def packet(
    group_ns: int,
    *,
    sequences: dict[int, int] | None = None,
    capture_ns: dict[int, int] | None = None,
    jpegs: dict[int, bytes] | None = None,
    version: int = 1,
    image_count: int = 4,
    payload_size: int | None = None,
) -> bytes:
    sequences = sequences or {cam: 100 for cam in CAMERAS}
    capture_ns = capture_ns or {cam: group_ns + cam * 1000 for cam in CAMERAS}
    jpegs = jpegs or {cam: jpeg_bytes(cam, sequences[cam]) for cam in CAMERAS}
    records = b"".join(
        IMAGE_HEADER.pack(cam, 0, sequences[cam], capture_ns[cam], len(jpegs[cam]))
        + jpegs[cam]
        for cam in CAMERAS
    )
    declared = len(records) if payload_size is None else payload_size
    return PACKET_HEADER.pack(
        MAGIC, version, image_count, group_ns, declared
    ) + records


def decode_stub(jpeg: bytes, size: tuple[int, int]) -> np.ndarray | None:
    """Stand-in decoder: encodes (camera_id, sequence) into the pixels.

    Returns ``None`` for a payload tagged undecodable, which is how the
    decode-failure tests ask for a failure without needing a real broken JPEG.
    """
    if jpeg[:2] != b"\xff\xd8" or len(jpeg) < 8:
        return None
    camera_id = jpeg[2]
    (sequence,) = struct.unpack("!I", jpeg[3:7])
    if camera_id == 0 or sequence == 0xDEAD:
        return None
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:, :, 0] = camera_id
    frame[:, :, 1] = sequence & 0xFF
    return frame


def manager() -> TimestampManager:
    return TimestampManager()


def stepping_clock(step_ns: int = 100_000):
    """A monotonic clock that always advances by ``step_ns``.

    The receive-time clock is what the arrival timestamp mode reads, and a real
    one has microsecond resolution: two groups ingested back to back in a test
    loop can land in the *same* microsecond, which the receiver correctly reads
    as "the timestamp did not advance" and drops.  That is a property of the
    test, not of the link (the vehicle sends 200 ms apart), so tests that are
    not about that rule get a clock that cannot collide.
    """
    counter = [0]

    def tick() -> int:
        counter[0] += step_ns
        return counter[0]

    return tick


def build_source(
    *,
    config: CimgConfig | None = None,
    timestamp_manager: TimestampManager | None = None,
    decoder=decode_stub,
    monotonic_ns=None,
) -> tuple[CimgCameraSource, list[FrameBuffer], threading.RLock]:
    config = config or CimgConfig(port=0)
    lock = threading.RLock()
    buffers = [
        FrameBuffer(max_size=config.queue_frames, lock=lock)
        for _ in range(config.camera_count)
    ]
    source = CimgCameraSource(
        buffers,
        config=config,
        timestamp_manager=timestamp_manager or manager(),
        decoder=decoder,
        group_lock=lock,
        monotonic_ns=monotonic_ns or stepping_clock(),
    )
    return source, buffers, lock


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_for_port(port: int, timeout_s: float = 5.0) -> socket.socket:
    """Connect to the reader once it is listening; fail if it never is."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            conn = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            conn.settimeout(5.0)
            return conn
        except OSError:
            time.sleep(0.02)
    raise AssertionError(f"listener never came up on port {port}")


class LiveReader:
    """A reader thread on an ephemeral port, with a client socket attached."""

    def __init__(self, **kwargs) -> None:
        self.port = free_port()
        self.config = CimgConfig(host="127.0.0.1", port=self.port, **kwargs)
        self.source, self.buffers, self.lock = build_source(config=self.config)

    def __enter__(self) -> "LiveReader":
        self.source.start()
        self.conn = wait_for_port(self.port)
        # Wait for the accept so stats reflect a live session rather than a
        # listener that has not yet seen the client.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not self.source.stats()["connected"]:
            time.sleep(0.01)
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            self.conn.close()
        except OSError:
            pass
        self.stop()

    def stop(self) -> None:
        self.source.stop()
        self.source.join(3.0)


# --------------------------------------------------------------------------- #
# Ingest level
# --------------------------------------------------------------------------- #


class IngestTests(unittest.TestCase):
    def test_synchronizer_reads_four_buffers_under_the_atomic_group_lock(self) -> None:
        lock = threading.RLock()
        first_snapshot = threading.Event()
        writer_attempted = threading.Event()
        writer_acquired = threading.Event()
        acquired_during_snapshot: list[bool] = []

        class PausingBuffer(FrameBuffer):
            def snapshot(self):
                first_snapshot.set()
                self.assert_writer_attempted = writer_attempted.wait(1.0)
                time.sleep(0.02)
                acquired_during_snapshot.append(writer_acquired.is_set())
                return super().snapshot()

        buffers = [PausingBuffer(lock=lock)] + [FrameBuffer(lock=lock) for _ in range(3)]
        old_times = (100, 100, 100, 100)
        for camera_id, (buf, timestamp) in enumerate(zip(buffers, old_times), 1):
            buf.put(FrameEntry(timestamp, np.zeros((1, 1, 3), dtype=np.uint8),
                               camera_id=camera_id, group_time_us=1))

        def write_new_group() -> None:
            first_snapshot.wait(1.0)
            writer_attempted.set()
            with lock:
                writer_acquired.set()
                for camera_id, buf in enumerate(buffers, 1):
                    buf.put(FrameEntry(101, np.zeros((1, 1, 3), dtype=np.uint8),
                                       camera_id=camera_id, group_time_us=2))

        writer = threading.Thread(target=write_new_group)
        writer.start()
        synchronizer = CameraSynchronizer(
            buffers,
            max_target_error_us=20,
            max_cross_camera_skew_us=20,
            group_lock=lock,
        )
        group = synchronizer.get_group(100, timeout_s=1.0)
        writer.join(1.0)

        self.assertIsNotNone(group)
        self.assertEqual({1}, {entry.group_time_us for entry in group.packets.values()})
        self.assertEqual([False], acquired_during_snapshot)
        self.assertFalse(writer.is_alive())

    def test_a_normal_packet_fills_all_four_cameras(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(packet(BASE_NS, sequences={1: 10, 2: 11, 3: 12, 4: 13}))
        self.assertEqual([1, 1, 1, 1], [b.depth() for b in buffers])
        self.assertEqual(
            [10, 11, 12, 13], [b.latest.sequence for b in buffers]
        )
        self.assertEqual(
            [1, 2, 3, 4], [b.latest.camera_id for b in buffers]
        )

    def test_camera_id_maps_through_the_model_index_to_the_right_buffer(self) -> None:
        """The production mapping: vehicle 1=front, 2=right, 3=left, 4=back.

        Alpamayo indexes front=1, right=2, left=0, back=4, so the vehicle's
        camera_id 3 (left) must reach the buffer the model reads as index 0.
        Getting this wrong is invisible in every frame count and silently swaps
        two of the four images in the model's input.
        """
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(packet(BASE_NS))
        # The decoded pixel carries the camera_id the vehicle sent, so this
        # checks each payload landed in the buffer the mapping chose.
        self.assertEqual([1, 2, 3, 4], [int(b.latest.camera_id) for b in buffers])
        self.assertEqual([1, 2, 3, 4], [int(b.latest.bgr[0, 0, 0]) for b in buffers])
        self.assertEqual(
            [1, 2, 0, 4],
            [row["camera_index"] for row in source.stats()["cameras"]],
        )
        # The identity carries the model index, so camera 3's frame - left, model
        # index 0 - is the one the inference package will call "cam0".
        self.assertEqual(
            ["cam1:s100", "cam2:s100", "cam0:s100", "cam4:s100"],
            [b.latest.frame_id for b in buffers],
        )

    def test_a_custom_index_mapping_is_honoured(self) -> None:
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, camera_indices=(4, 3, 2, 1))
        )
        source.ingest_bytes(packet(BASE_NS))
        self.assertEqual(
            [4, 3, 2, 1],
            [row["camera_index"] for row in source.stats()["cameras"]],
        )
        self.assertEqual("cam4:s100", buffers[0].latest.frame_id)

    def test_provenance_survives_into_the_frame_entry(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(
            packet(BASE_NS, sequences={1: 40, 2: 38, 3: 36, 4: 39},
                   capture_ns={cam: BASE_NS + cam for cam in CAMERAS})
        )
        entry = buffers[2].latest
        self.assertEqual(3, entry.camera_id)
        self.assertEqual(36, entry.sequence)
        self.assertEqual(BASE_NS // 1000, entry.group_time_us)
        # frame_id uses the *model* index, matching what camera.sync puts in the
        # inference package.
        self.assertEqual("cam0:s36", entry.frame_id)
        self.assertIsNone(entry.pts)

    def test_a_group_is_written_whole_and_never_partially(self) -> None:
        """All four buffers must agree on the group, or the window is warped.

        Two parts: at rest after each ingest, and - the part that actually
        tests the lock - observed from another thread that takes the same
        shared lock the writer holds across its four ``put`` calls.  If the
        writer released the lock between cameras, the observer would catch the
        four buffers holding different groups.
        """
        source, buffers, lock = build_source(config=CimgConfig(port=0))
        stop = threading.Event()
        tears: list[tuple[int | None, ...]] = []

        def observe() -> None:
            while not stop.is_set():
                with lock:
                    seen = tuple(b.latest.group_time_us if b.latest else None
                                 for b in buffers)
                if len(set(seen)) > 1:
                    tears.append(seen)

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            for index in range(200):
                source.ingest_bytes(packet(BASE_NS + index * 1_000_000))
        finally:
            stop.set()
            observer.join(3.0)

        self.assertEqual([], tears, "a group was visible in pieces")
        self.assertEqual(
            1, len({b.latest.group_time_us for b in buffers})
        )
        # And the same holds with no lock at all, once the writer has stopped:
        # a torn state is never left behind.
        seen = tuple(b.latest.group_time_us for b in buffers)
        self.assertEqual(1, len(set(seen)))

    def test_every_packet_of_a_fragmented_stream_lands_in_order(self) -> None:
        """Framing across read boundaries, with the backlog policy out of the way.

        ``max_backlog_groups`` is raised because the chunks here are small and
        uneven: a chunk that completes three packets is a backlog by the
        receiver's definition, and this test is about the framing, not about
        how a backlog is handled (covered separately below).
        """
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, max_backlog_groups=16)
        )
        stream = b"".join(
            packet(BASE_NS + index * 1_000_000,
                   sequences={cam: 100 + index for cam in CAMERAS})
            for index in range(5)
        )
        for offset in range(0, len(stream), 777):
            source.ingest_bytes(stream[offset : offset + 777])
        for buf in buffers:
            self.assertEqual([100, 101, 102, 103, 104],
                             [f.sequence for f in buf.snapshot()])

    def test_queue_drops_the_oldest_and_keeps_the_newest(self) -> None:
        """The latency rule: a full buffer costs the oldest frame, never the newest."""
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, queue_frames=3)
        )
        for index in range(6):
            source.ingest_bytes(
                packet(BASE_NS + index * 1_000_000,
                       sequences={cam: index for cam in CAMERAS})
            )
        for buf in buffers:
            self.assertEqual(3, buf.depth())
            self.assertEqual(3, buf.dropped)
            self.assertEqual([3, 4, 5], [f.sequence for f in buf.snapshot()])
        stats = source.stats()
        self.assertEqual(3, stats["cameras"][0]["queue_depth"])
        self.assertEqual(3, stats["cameras"][0]["queue_dropped"])

    def test_a_backlog_is_collapsed_to_the_newest_group(self) -> None:
        """A burst that leaves several complete packets is not replayed.

        Replaying it would be accumulated lag by another route: by the time the
        older groups were ingested their frames would already be the age the
        queue policy exists to avoid.
        """
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, max_backlog_groups=2)
        )
        stream = b"".join(
            packet(BASE_NS + index * 1_000_000,
                   sequences={cam: index for cam in CAMERAS})
            for index in range(5)
        )
        source.ingest_bytes(stream)  # one read, five complete packets
        stats = source.stats()
        self.assertEqual(4, stats["groups_dropped"]["backlog"])
        # All five were received - the vehicle sent them - and one was used.
        # ``receive_fps`` is derived from the received count, so a backlog drop
        # does not disguise itself as a slower vehicle.
        self.assertEqual(5, stats["groups_total"])
        self.assertEqual(1, stats["groups_ingested"])
        for buf in buffers:
            self.assertEqual(1, buf.depth(), "only the newest group is kept")
            self.assertEqual(4, buf.latest.sequence)

    def test_real_sized_fragmented_socket_backlog_keeps_latest_complete_group(self) -> None:
        """64 KiB recv chunks still expose an 800 KiB/group backlog safely."""
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, max_backlog_groups=2)
        )
        large_jpegs = {
            cam: jpeg_bytes(cam, 0, size=200_000) for cam in CAMERAS
        }
        complete = b"".join(
            packet(
                BASE_NS + index * 100_000_000,
                sequences={cam: index for cam in CAMERAS},
                jpegs={
                    cam: jpeg_bytes(cam, index, size=len(large_jpegs[cam]))
                    for cam in CAMERAS
                },
            )
            for index in range(5)
        )
        trailing = packet(
            BASE_NS + 500_000_000,
            sequences={cam: 5 for cam in CAMERAS},
            jpegs={cam: jpeg_bytes(cam, 5, size=200_000) for cam in CAMERAS},
        )
        stream = complete + trailing[:123]

        class QueuedSocket:
            def __init__(self, data: bytes) -> None:
                self.data = data

            def setblocking(self, _enabled: bool) -> None:
                pass

            def settimeout(self, _timeout: float) -> None:
                pass

            def recv(self, size: int) -> bytes:
                if not self.data:
                    raise BlockingIOError
                chunk, self.data = self.data[:size], self.data[size:]
                return chunk

        first, rest = stream[: 1 << 16], stream[1 << 16 :]
        parsed = source._drain_available(QueuedSocket(rest), first)
        source._consume(parsed)

        self.assertEqual(5, source.stats()["groups_received"])
        self.assertEqual(1, source.stats()["groups_ingested"])
        self.assertEqual(4, buffers[0].latest.sequence)
        self.assertEqual(123, source.parser.buffered_bytes)

    def test_backlog_within_the_allowance_is_replayed_in_full(self) -> None:
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, max_backlog_groups=3)
        )
        source.ingest_bytes(
            b"".join(
                packet(BASE_NS + index * 1_000_000,
                       sequences={cam: index for cam in CAMERAS})
                for index in range(3)
            )
        )
        self.assertEqual(0, source.stats()["groups_dropped"]["backlog"])
        self.assertEqual([0, 1, 2], [f.sequence for f in buffers[0].snapshot()])

    def test_the_receiver_keeps_up_when_nothing_reads_the_buffers(self) -> None:
        """No producer/consumer handshake: a stalled consumer cannot stall the link.

        The buffers are never drained here.  A design that applied back-pressure
        to the socket would stall; this one must keep accepting and keep the
        newest data, at a bounded depth.
        """
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, queue_frames=4)
        )
        for index in range(500):
            source.ingest_bytes(
                packet(BASE_NS + index * 1_000_000,
                       sequences={cam: index for cam in CAMERAS})
            )
        for buf in buffers:
            self.assertEqual(4, buf.depth())
            self.assertEqual(499, buf.latest.sequence)
            self.assertLessEqual(buf.depth(), buf.max_size)


class FailureTests(unittest.TestCase):
    def test_a_decode_failure_drops_the_whole_group(self) -> None:
        """Not just the one image: the other three must not advance.

        A group that reached only three cameras would pair those three frames
        with the previous group's fourth - a time-warped window in which every
        individual frame is fresh, so no freshness gate could catch it.
        """
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(packet(BASE_NS, sequences={cam: 1 for cam in CAMERAS}))
        self.assertEqual(1, buffers[0].latest.sequence)

        broken = {cam: jpeg_bytes(cam, 1) for cam in CAMERAS}
        broken[4] = jpeg_bytes(0, 0xDEAD)  # decoder returns None for this
        source.ingest_bytes(packet(BASE_NS + 1_000_000, sequences={cam: 2 for cam in CAMERAS}, jpegs=broken))

        stats = source.stats()
        self.assertEqual(1, stats["decode_errors"])
        self.assertEqual(2, stats["groups_received"])
        self.assertEqual(1, stats["groups_ingested"])
        self.assertEqual(8, stats["images_received"])
        self.assertEqual(4, stats["images_total"])
        for buf in buffers:
            self.assertEqual(1, buf.depth(), "the group must not be half-written")
            self.assertEqual(1, buf.latest.sequence)

    def test_a_decoder_exception_is_one_frame_not_a_dead_thread(self) -> None:
        def exploding(jpeg, size):
            raise RuntimeError("decoder blew up")

        source, buffers, _ = build_source(
            config=CimgConfig(port=0), decoder=exploding
        )
        source.ingest_bytes(packet(BASE_NS))
        self.assertEqual(1, source.stats()["decode_errors"])
        self.assertEqual(0, buffers[0].depth())

    def test_a_wrong_protocol_version_is_reported_and_the_link_recovers(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        stream = packet(BASE_NS, version=2) + packet(BASE_NS + 1_000_000,
                                                     sequences={cam: 7 for cam in CAMERAS})
        source.ingest_bytes(stream)
        self.assertEqual(1, source.stats()["protocol_errors"]["packets_skipped"])
        self.assertGreaterEqual(source.stats()["protocol_errors"]["resyncs"], 1)
        self.assertIn("不支持协议版本", source.stats()["protocol_errors"]["last_error"])
        self.assertEqual(7, buffers[0].latest.sequence)

    def test_an_absurd_payload_size_is_rejected_without_buffering_it(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(
            packet(BASE_NS, payload_size=200_000_000)
            + packet(BASE_NS + 1_000_000, sequences={cam: 9 for cam in CAMERAS})
        )
        self.assertEqual(1, source.stats()["protocol_errors"]["packets_skipped"])
        self.assertEqual(9, buffers[0].latest.sequence)

    def test_any_capture_time_ns_becomes_the_frame_clock(self) -> None:
        """The vehicle always sends capture_time_ns; it is converted, not judged."""
        source, buffers, _ = build_source(
            config=CimgConfig(port=0), timestamp_manager=manager()
        )
        source.ingest_bytes(
            packet(BASE_NS, capture_ns={cam: 1_000 for cam in CAMERAS})
        )
        stats = source.stats()
        self.assertEqual(0, stats["timestamp_errors"])
        self.assertEqual(1, stats["groups_ingested"])
        self.assertEqual(1, buffers[0].latest.timestamp_us)
        self.assertNotIn("timestamp_fallback", stats)

    def test_a_valid_vehicle_timestamp_becomes_the_frame_clock(self) -> None:
        source, buffers, _ = build_source(
            config=CimgConfig(port=0), timestamp_manager=manager()
        )
        now_ns = time.time_ns()
        source.ingest_bytes(
            packet(now_ns // 1_000_000 * 1_000_000,
                   capture_ns={1: now_ns, 2: now_ns + 1, 3: now_ns + 2, 4: now_ns + 3})
        )
        self.assertEqual(now_ns // 1000, buffers[0].latest.timestamp_us)
        self.assertEqual((now_ns + 3) // 1000, buffers[3].latest.timestamp_us)
        self.assertEqual(0, source.stats()["timestamp_errors"])
        self.assertNotIn("timestamp_fallback", source.stats())

    def test_a_group_whose_timestamps_do_not_advance_is_dropped_whole(self) -> None:
        """Camera-to-camera disagreement must not be papered over.

        ``timestamp_us <= latest`` on any camera means the group's four clocks
        are not all ahead of what the buffers hold; writing the other three
        would leave the four buffers on different groups, which is what the
        atomic write exists to prevent.  It is not an error worth alarming on -
        the vehicle has been seen sending a lower number on one camera - so it
        is counted and the next group is used.
        """
        # Frame clocks are the packet's capture times, so one camera can
        # disagree with the others about the group's time.
        source, buffers, _ = build_source(
            config=CimgConfig(port=0), timestamp_manager=manager()
        )
        now_ns = time.time_ns()
        source.ingest_bytes(
            packet(now_ns // 1_000_000 * 1_000_000,
                   capture_ns={1: now_ns, 2: now_ns + 1, 3: now_ns + 2, 4: now_ns + 3})
        )
        self.assertEqual(1, buffers[0].depth())
        # Second group: camera 2 repeats its capture time, so the group is not
        # wholly ahead of what is buffered.
        source.ingest_bytes(
            packet(now_ns // 1_000_000 * 1_000_000 + 1_000_000_000,
                   capture_ns={1: now_ns + 4, 2: now_ns + 1, 3: now_ns + 5, 4: now_ns + 6})
        )
        stats = source.stats()
        self.assertEqual(1, stats["groups_dropped"]["out_of_order"])
        self.assertEqual(1, stats["cameras"][1]["out_of_order"])
        for buf in buffers:
            self.assertEqual(1, buf.depth())

    def test_a_sequence_gap_is_logged_as_a_jump_not_as_packet_loss(self) -> None:
        """The log wording is a requirement, so it is asserted here too.

        ``CimgCameraSource`` logs through the module logger; capturing it is
        the only way to check what an operator would actually read.
        """
        import logging

        records: list[str] = []

        class Catcher(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        logger = logging.getLogger("camera.cimg_source")
        handler = Catcher()
        logger.addHandler(handler)
        level = logger.level
        logger.setLevel(logging.INFO)
        try:
            source, buffers, _ = build_source(config=CimgConfig(port=0))
            source.ingest_bytes(packet(BASE_NS, sequences={cam: 10 for cam in CAMERAS}))
            source.ingest_bytes(
                packet(BASE_NS + 1_000_000, sequences={cam: 14 for cam in CAMERAS})
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)

        joined = "\n".join(records)
        self.assertIn("序号跳变", joined)
        self.assertIn("+3", joined)
        # The message says "not network loss" on purpose, so a bare substring
        # search for 丢包 would fail on the disclaimer itself.  What must not
        # appear is the *claim*: any wording that tells an operator the network
        # dropped packets.
        self.assertNotIn("网络丢包", joined.replace("不等于网络丢包", ""))
        for forbidden in ("丢帧", "packet loss", "packets lost"):
            self.assertNotIn(forbidden, joined)
        self.assertEqual(1, source.stats()["cameras"][0]["sequence_gaps"])

    def test_a_repeated_sequence_is_counted_but_not_called_a_gap(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(packet(BASE_NS, sequences={cam: 5 for cam in CAMERAS}))
        source.ingest_bytes(
            packet(BASE_NS + 1_000_000, sequences={cam: 5 for cam in CAMERAS})
        )
        self.assertEqual(1, source.stats()["cameras"][0]["repeated_sequences"])
        self.assertEqual(0, source.stats()["cameras"][0]["sequence_gaps"])

    def test_uint32_sequence_wrap_is_not_a_gap(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(
            packet(BASE_NS, sequences={cam: 0xFFFFFFFF for cam in CAMERAS})
        )
        source.ingest_bytes(
            packet(BASE_NS + 1_000_000, sequences={cam: 0 for cam in CAMERAS})
        )
        self.assertEqual(0, source.stats()["cameras"][0]["sequence_gaps"])
        self.assertEqual(0, source.stats()["cameras"][0]["out_of_order"])


class StatsTests(unittest.TestCase):
    def test_the_required_health_fields_are_present_and_per_camera(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        stats = source.stats()
        # Connection state, receive fps, newest timestamps, per-camera sequence,
        # frame age, queue depth, and the two error families kept apart.
        for key in (
            "connection_state", "connected", "peer", "sessions", "receive_fps",
            "groups_total", "images_total", "images_received", "bytes_total", "frame_age_ms",
            "last_group_time_ns", "last_group_latency_ms", "last_capture_span_ms",
            "protocol_errors", "decode_errors", "timestamp_errors",
            "read_timeouts", "groups_dropped",
        ):
            self.assertIn(key, stats)
        self.assertEqual(4, len(stats["cameras"]))
        for row in stats["cameras"]:
            for key in (
                "source", "camera_id", "camera_index", "health", "sequence",
                "frame_age_ms", "queue_depth", "queue_dropped", "frames_total",
                "sequence_gaps", "last_capture_time_ns", "last_timestamp_us",
            ):
                self.assertIn(key, row)

    def test_each_camera_row_carries_the_keys_check_chain_reads(self) -> None:
        """The keys ``tools/check_chain.sh`` indexes by name, not by ``get``.

        That tool is what the server workflow runs to decide whether the chain
        is healthy, and it reads ``restarts`` and ``last_error`` off every
        reader row.  Publishing the CIMG link as four per-camera rows keeps the
        decision layer and the tool working unchanged - but only if the rows
        really are the same shape, and a missing key there is a KeyError inside
        the operator's diagnostic rather than a red line on it.
        """
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source.ingest_bytes(packet(BASE_NS))
        required = {
            "source", "health", "frames_total", "frame_age_ms", "restarts",
            "last_error",
        }
        report = source.stats()
        for row in report["cameras"]:
            with self.subTest(camera_id=row["camera_id"]):
                self.assertLessEqual(required, set(row))
        # The link-level values those two keys hold on a healthy link, so a row
        # that reports "4 restarts" before anything has restarted is caught.
        self.assertIsInstance(report["restarts"], int)
        self.assertIsNone(report["last_error"], "no failure has happened yet")

    def test_a_link_failure_is_visible_on_every_camera_row(self) -> None:
        # One socket, so the reason is shared: the row is where a red camera is
        # read, and "connection refused" is the answer a reader needs there.
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        source._note_failure("连接被拒绝", log=False)
        rows = source.stats()["cameras"]
        self.assertEqual(["连接被拒绝"] * 4, [row["last_error"] for row in rows])
        self.assertEqual([1] * 4, [row["consecutive_failures"] for row in rows])

    def test_health_is_starting_before_any_frame_and_healthy_after(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        self.assertEqual("STARTING", source.health)
        source.ingest_bytes(packet(BASE_NS))
        self.assertEqual("HEALTHY", source.health)
        self.assertEqual(["HEALTHY"] * 4,
                         [row["health"] for row in source.stats()["cameras"]])

    def test_a_stalled_stream_goes_stale(self) -> None:
        source, buffers, _ = build_source(
            config=CimgConfig(port=0, stale_after_s=0.05),
            # The real clock: staleness is age against wall time, so a test that
            # advances a fake clock per call could never age anything.
            monotonic_ns=time.monotonic_ns,
        )
        source.ingest_bytes(packet(BASE_NS))
        self.assertEqual("HEALTHY", source.health)
        time.sleep(0.08)
        self.assertEqual("STALE", source.health)
        self.assertEqual(
            ["STALE"] * 4,
            [row["health"] for row in source.stats()["cameras"]],
        )

    def test_frame_age_is_measured_on_the_server_clock_not_the_vehicle_clock(self) -> None:
        """A skewed vehicle clock must not be able to make frames look fresh.

        ``frame_age_ms`` is the one field the decision layer's staleness gate
        reads, so it is derived from the arrival (monotonic) time; the vehicle's
        timestamps are for alignment, not for deciding whether it is safe.
        """
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        old_ns = time.time_ns() - 3_600 * 1_000_000_000
        source.ingest_bytes(packet(old_ns))
        row = source.stats()["cameras"][0]
        self.assertIsNotNone(row["frame_age_ms"])
        self.assertLess(row["frame_age_ms"], 5_000)
        self.assertEqual(old_ns, row["last_group_time_ns"])
        self.assertGreater(old_ns, 1_000_000_000_000_000_000)

    def test_summary_frame_age_is_the_oldest_camera_age(self) -> None:
        now_us = 10_000_000
        source, _buffers, _ = build_source(
            config=CimgConfig(port=0, stale_after_s=20),
            monotonic_ns=lambda: now_us * 1000,
        )
        for camera_id, age_ms in zip(CAMERAS, (100, 200, 300, 900)):
            source._cameras[camera_id].last_arrival_us = now_us - age_ms * 1000
        self.assertEqual(900.0, source.stats()["frame_age_ms"])
        self.assertEqual("HEALTHY", source.health)

    def test_receive_fps_reflects_the_arrival_rate(self) -> None:
        source, buffers, _ = build_source(config=CimgConfig(port=0))
        self.assertEqual(0.0, source.receive_fps())
        for index in range(8):
            source.ingest_bytes(packet(BASE_NS + index * 1_000_000))
            time.sleep(0.01)
        self.assertGreater(source.receive_fps(), 10.0)

    def test_a_burst_in_one_clock_tick_still_reports_a_rate(self) -> None:
        """The arrival clock can be too coarse to measure this link.

        ``monotonic_ns`` is 15.6 ms on Windows, and a reconnect or a backlog
        drain legitimately delivers several groups in one read, so every sample
        in the window can share a tick.  Reporting 0.0 there reads as "nothing
        is arriving" on a link that is carrying frames, so the vehicle's group
        times stand in as the denominator.
        """
        # One tick: the clock never advances, as on a coarse platform.
        source, buffers, _ = build_source(
            config=CimgConfig(port=0), monotonic_ns=lambda: 0
        )
        for index in range(5):
            source.ingest_bytes(packet(BASE_NS + index * 200_000_000))
        self.assertAlmostEqual(5.0, source.receive_fps(), places=6)

    def test_the_group_time_fallback_cannot_invent_a_rate(self) -> None:
        # A sender clock that does not move has nothing to say about rate, and
        # neither then does this field: better a zero an operator can question
        # than a number derived from a clock that is not running.
        source, buffers, _ = build_source(
            config=CimgConfig(port=0), monotonic_ns=lambda: 0
        )
        for _ in range(5):
            source.ingest_bytes(packet(BASE_NS))
        self.assertEqual(0.0, source.receive_fps())


# --------------------------------------------------------------------------- #
# Socket level
# --------------------------------------------------------------------------- #


class SocketTests(unittest.TestCase):
    def test_no_socket_is_opened_until_the_thread_starts(self) -> None:
        """The bind belongs on the reader thread, not in the constructor.

        A provider that bound during construction would hold port 30106 for the
        whole process lifetime, including a startup that later fails.
        """
        port = free_port()
        source, buffers, _ = build_source(config=CimgConfig(host="127.0.0.1", port=port))
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)
        self.assertFalse(source.is_alive())

    def test_a_stream_over_a_real_socket_fills_the_buffers(self) -> None:
        with LiveReader() as live:
            for index in range(4):
                live.conn.sendall(
                    packet(BASE_NS + index * 1_000_000,
                           sequences={cam: index for cam in CAMERAS})
                )
                # One packet per round trip.  Four sends with no pause can
                # coalesce into one read, and a batch of four *is* a backlog -
                # the receiver would be right to collapse it, and this test
                # would then be asserting the wrong rule.
                deadline = time.monotonic() + 5.0
                while (time.monotonic() < deadline
                       and live.buffers[0].depth() < index + 1):
                    time.sleep(0.01)
            self.assertEqual([0, 1, 2, 3],
                             [f.sequence for f in live.buffers[0].snapshot()])
            stats = live.source.stats()
            self.assertTrue(stats["connected"])
            self.assertEqual(1, stats["sessions"])
            self.assertEqual(4, stats["groups_total"])
            self.assertEqual(16, stats["images_total"])
            self.assertEqual(0, stats["groups_dropped"]["backlog"])
            self.assertIn("127.0.0.1:", stats["peer"])

    def test_a_half_packet_at_disconnect_is_abandoned_and_reconnect_resumes(self) -> None:
        """CLOUD_PUSH.md: a partial packet is dropped, the next connection starts at a header."""
        with LiveReader() as live:
            live.conn.sendall(packet(BASE_NS, sequences={cam: 3 for cam in CAMERAS}))
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and live.buffers[0].depth() < 1:
                time.sleep(0.01)
            partial = packet(BASE_NS + 1_000_000, sequences={cam: 4 for cam in CAMERAS})
            live.conn.sendall(partial[:120])
            time.sleep(0.2)
            live.conn.close()

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and live.source.stats()["connected"]:
                time.sleep(0.01)
            stats = live.source.stats()
            self.assertEqual(1, stats["truncated_reads"])
            self.assertEqual(1, stats["groups_dropped"]["partial"])
            self.assertEqual(1, stats["groups_total"], "the half packet is not a group")

            with socket.create_connection(("127.0.0.1", live.port), timeout=2.0) as again:
                again.sendall(
                    packet(BASE_NS + 2_000_000, sequences={cam: 5 for cam in CAMERAS})
                )
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and live.buffers[0].depth() < 2:
                    time.sleep(0.01)
            self.assertEqual([3, 5], [f.sequence for f in live.buffers[0].snapshot()])
            self.assertEqual(2, live.source.stats()["sessions"])
            self.assertEqual(2, live.source.restart_count)

    def test_a_stalled_link_times_out_so_the_vehicle_reconnects(self) -> None:
        """The idle budget is semantic: no complete packet, no connection.

        A silent TCP connection looks identical to a healthy one from the
        server's side, so without this a vehicle that stopped encoding would
        leave four cameras "connected" and stale indefinitely.
        """
        with LiveReader(read_timeout_s=0.3) as live:
            live.conn.sendall(packet(BASE_NS, sequences={cam: 1 for cam in CAMERAS}))
            time.sleep(0.1)
            live.conn.sendall(packet(BASE_NS, sequences={cam: 2 for cam in CAMERAS})[:80])
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and live.source.stats()["read_timeouts"] < 1:
                time.sleep(0.02)
            stats = live.source.stats()
            self.assertEqual(1, stats["read_timeouts"])
            self.assertEqual(1, stats["groups_dropped"]["partial"])
            self.assertEqual("listening", stats["connection_state"])

    def test_a_bad_packet_over_a_socket_does_not_end_the_session(self) -> None:
        with LiveReader() as live:
            live.conn.sendall(packet(BASE_NS, version=3))
            live.conn.sendall(
                packet(BASE_NS + 1_000_000, sequences={cam: 42 for cam in CAMERAS})
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and live.buffers[0].depth() < 1:
                time.sleep(0.01)
            self.assertEqual(42, live.buffers[0].latest.sequence)
            stats = live.source.stats()
            self.assertEqual(1, stats["sessions"], "the session must survive")
            self.assertEqual(1, stats["protocol_errors"]["packets_skipped"])

    def test_stop_is_prompt_and_releases_the_port(self) -> None:
        live = LiveReader()
        live.__enter__()
        try:
            live.conn.sendall(packet(BASE_NS, sequences={cam: 1 for cam in CAMERAS}))
            started = time.monotonic()
            live.source.stop()
            live.source.join(3.0)
            elapsed = time.monotonic() - started
        finally:
            try:
                live.conn.close()
            except OSError:
                pass
        self.assertFalse(live.source.is_alive(), "the reader thread must exit")
        self.assertLess(elapsed, 3.0, "shutdown must not wait out the recv poll")
        # The port is free again, which is what lets a restart bind it.
        probe = socket.socket()
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", live.port))
        finally:
            probe.close()
        self.assertEqual("STOPPED", live.source.health)

    def test_stop_called_twice_is_harmless(self) -> None:
        live = LiveReader()
        live.__enter__()
        try:
            live.source.stop()
            live.source.stop()
            live.source.join(3.0)
        finally:
            try:
                live.conn.close()
            except OSError:
                pass
        self.assertFalse(live.source.is_alive())

    def test_a_bind_conflict_is_reported_loudly_rather_than_silently(self) -> None:
        """A port already in use must be visible in /health.

        A swallowed "Address already in use" would look like a vehicle that
        never connected.

        The failure is injected rather than provoked with a real second bind,
        because a real one does not fail on Windows: ``SO_REUSEADDR`` there
        permits a second socket to take a port that is already listening, which
        is a property of the platform and not of this receiver.  The behaviour
        under test is the *handling* - loud, recorded, retried, never silent.
        """
        port = free_port()
        source, buffers, _ = build_source(
            config=CimgConfig(host="127.0.0.1", port=port)
        )
        real_bind = socket.socket.bind

        def refuse(self, address):
            if address == ("127.0.0.1", port):
                raise OSError(48, "Address already in use")
            return real_bind(self, address)

        with unittest.mock.patch.object(socket.socket, "bind", refuse):
            source.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and source.connection_state != "bind_failed":
                time.sleep(0.02)
            self.assertEqual("FAILED", source.health)
            self.assertEqual("bind_failed", source.connection_state)
            self.assertIn("listen failed", source.stats()["last_error"])
            self.assertGreaterEqual(source.consecutive_failures, 1)
            source.stop()
            source.join(3.0)
        self.assertFalse(source.is_alive())

    def test_a_restart_after_a_bind_failure_clears_the_failed_state(self) -> None:
        """FAILED must not be sticky, or a fixed port would need a process restart."""
        port = free_port()
        source, buffers, _ = build_source(
            config=CimgConfig(host="127.0.0.1", port=port)
        )
        real_bind = socket.socket.bind
        calls = [0]

        def refuse_once(self, address):
            calls[0] += 1
            if calls[0] == 1:
                raise OSError(48, "Address already in use")
            return real_bind(self, address)

        with unittest.mock.patch.object(socket.socket, "bind", refuse_once):
            source.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and source.connection_state != "listening":
                time.sleep(0.05)
            self.assertEqual("listening", source.connection_state)
            self.assertEqual("STARTING", source.health)
            source.stop()
            source.join(3.0)


class ConfigurationTests(unittest.TestCase):
    def test_the_buffer_count_must_match_the_camera_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "4 个相机缓存"):
            CimgCameraSource(
                [FrameBuffer() for _ in range(3)],
                config=CimgConfig(),
                timestamp_manager=manager(),
            )

    def test_the_default_mapping_is_the_production_one(self) -> None:
        config = CimgConfig()
        self.assertEqual((1, 2, 0, 4), config.camera_indices)
        self.assertEqual(30106, config.port)
        self.assertEqual(4, config.camera_count)
        self.assertEqual((576, 320), config.target_size)

    def test_a_non_default_camera_count_is_supported_on_the_reader_side(self) -> None:
        """The reader is not hard-coded to four; the protocol's 4 is asserted in camera.cimg."""
        config = CimgConfig(camera_indices=(0, 1))
        lock = threading.RLock()
        buffers = [FrameBuffer(lock=lock) for _ in range(2)]
        source = CimgCameraSource(
            buffers, config=config, timestamp_manager=manager(),
            decoder=decode_stub, group_lock=lock,
        )
        self.assertEqual(2, len(source.stats()["cameras"]))
        self.assertEqual([0, 1], [row["camera_index"] for row in source.stats()["cameras"]])


class DecoderTests(unittest.TestCase):
    """``decode_bgr`` itself, against a fake cv2.

    The rest of this file injects a decoder, which is right for testing what the
    receiver does around a decode - but it means the one function that runs in
    production was never called on a machine without OpenCV.  A fake cv2 pins
    what this module is responsible for: the flag that decides the channel
    order, the shape guarantees the buffer and the window builder rely on, and
    "return None, never raise" on garbage.
    """

    class _FakeCv2:
        """Records how it was called and returns an array of the size asked."""

        IMREAD_COLOR = 1
        INTER_LINEAR = 2

        def __init__(self, *, frame=None, resized_dims=None) -> None:
            self.frame = frame
            self.resized_dims = resized_dims
            self.imdecode_calls: list = []
            self.resize_calls: list = []

        def imdecode(self, array, flags):
            self.imdecode_calls.append((len(array), flags))
            if self.frame is None:
                return None
            if self.resized_dims is not None:
                self.frame = np.zeros(
                    (self.resized_dims[1], self.resized_dims[0], 3), dtype=np.uint8
                )
            return self.frame

        def resize(self, frame, size, interpolation=None):
            self.resize_calls.append((size, interpolation))
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def _decode(self, fake, jpeg=b"\xff\xd8jpeg\xff\xd9", size=(576, 320)):
        """Call ``decode_bgr`` with ``fake`` standing in for the cv2 module.

        The function imports cv2 itself, so the substitution goes through
        ``sys.modules`` for the duration of the call.  Guarded by a restore in
        ``finally`` because a leaked fake would make every later test that
        touches cv2 silently pass against it.
        """
        import camera.cimg_source as module

        had = "cv2" in sys.modules
        previous = sys.modules.get("cv2")
        sys.modules["cv2"] = fake
        try:
            return module.decode_bgr(jpeg, size)
        finally:
            if had:
                sys.modules["cv2"] = previous
            else:
                sys.modules.pop("cv2", None)

    def test_a_decodable_frame_comes_back_in_bgr_at_the_model_size(self) -> None:
        fake = self._FakeCv2(frame=np.zeros((320, 576, 3), dtype=np.uint8))
        frame = self._decode(fake)
        self.assertIsNotNone(frame)
        self.assertEqual((320, 576, 3), frame.shape)
        # IMREAD_COLOR is the whole colour contract: it is what makes the result
        # BGR, which is what camera.sync reverses to RGB for the model.
        self.assertEqual([(len(b"\xff\xd8jpeg\xff\xd9"), fake.IMREAD_COLOR)],
                         fake.imdecode_calls)
        self.assertEqual([], fake.resize_calls, "already at target size")

    def test_a_1080p_frame_is_resized_and_not_merely_accepted(self) -> None:
        # 6.2 MB per frame at 1080p, so the resize is what keeps 60 buffered
        # frames near 130 MB instead of 1.5 GB.
        fake = self._FakeCv2(
            frame=np.zeros((1080, 1920, 3), dtype=np.uint8), resized_dims=(1920, 1080)
        )
        frame = self._decode(fake, size=(576, 320))
        self.assertEqual((320, 576, 3), frame.shape)
        self.assertEqual([((576, 320), fake.INTER_LINEAR)], fake.resize_calls)

    def test_undecodable_bytes_return_none_rather_than_raising(self) -> None:
        # cv2's own failure signal, and the one the receiver counts as a decode
        # error.  Raising here would take the receive thread down instead.
        self.assertIsNone(self._decode(self._FakeCv2(frame=None)))

    def test_a_grayscale_or_two_channel_result_is_refused(self) -> None:
        # cv2 can return 2-D or 4-channel arrays for some inputs; passing one on
        # would fail far away, inside the window builder's cvtColor.
        for shape in ((320, 576), (320, 576, 4), (320, 576, 1)):
            with self.subTest(shape=shape):
                fake = self._FakeCv2(frame=np.zeros(shape, dtype=np.uint8))
                self.assertIsNone(self._decode(fake))


# --------------------------------------------------------------------------- #
# End to end: receiver -> buffers -> the window the model is actually fed
# --------------------------------------------------------------------------- #


def _unix_now_us() -> int:
    return time.time_ns() // 1000


class WindowIntegrationTests(unittest.TestCase):
    """The whole CIMG link assembled, driven with the packets the vehicle sends.

    Everything below the window - framing, the four-way camera mapping, the
    atomic group write - has its own tests.  What is only checkable here is that
    the assembled parts agree: that the CIMG receiver feeds the *existing*
    history window (requirement 7) rather than a parallel one, and that a frame
    survives the trip with its identity and its channel order intact, which is
    what requirement 6 is about and what a stub decoder cannot prove on its own.

    ``CameraWindowProvider`` builds its reader without a decoder injection
    point, because production has exactly one decoder.  The test reaches in and
    swaps the bound decoder for the pixel-tagging stub for the same reason the
    ingest tests inject one: what is under test is the wiring around the decode.
    """

    CIMG_INDICES = (1, 2, 0, 4)

    def test_ten_hz_input_keeps_a_ten_hz_window_grid(self) -> None:
        provider = CameraWindowProvider(
            _ProviderCimgConfig(
                port=0,
                camera_indices=self.CIMG_INDICES,
                group_fps=10.0,
            ),
            frames_per_camera=2,
            target_fps=10.0,
            timestamp_manager=manager(),
        )
        self.assertEqual(100_000, provider.period_us)
        # 300 ms skew policy is wider than half a 10 Hz period (50 ms).
        self.assertEqual(300_000, provider.synchronizer.max_target_error_us)
        self.assertIs(provider._group_lock, provider.synchronizer.group_lock)

    def _provider(self, frames_per_camera: int = 2) -> CameraWindowProvider:
        provider = CameraWindowProvider(
            _ProviderCimgConfig(port=0, camera_indices=self.CIMG_INDICES),
            frames_per_camera=frames_per_camera,
            target_fps=5.0,
            timestamp_manager=manager(),
        )
        self.assertEqual(list(self.CIMG_INDICES), provider.camera_indices)
        self.assertEqual(1, len(provider.readers))
        provider.readers[0]._decode = decode_stub
        return provider

    def _feed(
        self, provider: CameraWindowProvider, count: int, *, base_sequence: int = 500
    ) -> None:
        """Ingest ``count`` groups 200 ms apart, ending at roughly now.

        Oldest first, so the frame timestamps advance in the order the vehicle
        would send them: the receiver refuses a group whose timestamp did not
        move, so a test that fed them newest-first would measure that rule
        instead of the window it is here to check.
        """
        reader = provider.readers[0]
        now_us = _unix_now_us()
        for step in reversed(range(count)):
            group_us = now_us - step * 200_000
            # ``step`` counts down while time counts up, so the sequence has to
            # be derived from the position in the loop rather than from ``step``
            # - taking it from ``step`` numbers the older group higher, which
            # the entry provenance makes visible and nothing else would.
            sequence = base_sequence + (count - 1 - step)
            reader.ingest_bytes(
                packet(
                    group_us * 1000,
                    sequences={cam: sequence for cam in CAMERAS},
                    capture_ns={
                        cam: group_us * 1000 + cam * 1000 for cam in CAMERAS
                    },
                )
            )

    def test_the_receiver_feeds_the_existing_history_window(self) -> None:
        provider = self._provider(frames_per_camera=2)
        self._feed(provider, 2)
        window = provider.latest_window(sync_timeout_s=1.0)
        self.assertIsNotNone(window, "the CIMG link must produce a model window")

        # (cameras, T, channels, H, W) at the model size, not the decode size:
        # the frames were decoded at target_size, so _build must not resize
        # them again on the way in.
        self.assertEqual((4, 2, 3, 320, 576), tuple(window.frames.shape))
        self.assertEqual(2, len(window.timestamps_us))
        self.assertIsNotNone(window.frame_age_us)
        # A window reached through no refusal at all - not a stale one, not an
        # age one, not a repeat of an earlier window.  Any of those would mean
        # the window above was produced by luck after invalidating the cache.
        self.assertEqual({}, dict(provider.refusals))

    def test_camera_indices_order_the_window_and_reach_the_model(self) -> None:
        """camera_id 1..4 -> model index 1,2,0,4, and the window is sorted by it.

        The mapping is asserted end to end here, not just on the config: the
        vehicle's cam3 is the model's left camera (index 0), so it must be row 0
        of the window with its own frame ids - a mapping that is right on paper
        and wrong in the buffer would still pass every count-based check.
        """
        provider = self._provider(frames_per_camera=2)
        self._feed(provider, 2)
        window = provider.latest_window(sync_timeout_s=1.0)
        self.assertIsNotNone(window)

        self.assertEqual([0, 1, 2, 4], list(window.camera_indices))
        # Row order is the model index order, so row 0 is camera_index 0, which
        # the mapping puts on vehicle camera 3.  The frame id says cam0 because
        # it names the model camera; the pixels below say vehicle camera 3.
        self.assertEqual(["cam0:s500", "cam0:s501"], window.frame_ids[0])
        self.assertEqual(["cam4:s500", "cam4:s501"], window.frame_ids[3])
        # The pixels agree with the ids, and they also pin the channel order.
        # decode_stub tags the id in the *blue* channel of a BGR array, so in
        # the model's RGB tensor it has to come out as the last channel - an
        # extra or a missing conversion both land it somewhere else, and the
        # model would be reading a colour-shifted frame with no other symptom.
        for row, camera_index in enumerate(window.camera_indices):
            expected_camera_id = self.CIMG_INDICES.index(int(camera_index)) + 1
            self.assertEqual(
                expected_camera_id,
                int(window.frames[row, -1, 2, 0, 0]),
                f"row {row} carries the wrong camera",
            )
            # Channel 0 is red: the decoder wrote 0 there and the id in blue,
            # so a window that skipped the conversion would show the camera id
            # in channel 0 instead - a colour shift with no other symptom.
            self.assertEqual(0, int(window.frames[row, -1, 0, 0, 0]))

    def test_the_health_view_reports_the_cimg_link(self) -> None:
        provider = self._provider(frames_per_camera=2)
        self._feed(provider, 2)
        # Keyed by vehicle camera id, because that is the number the operator
        # reads off the vehicle's own log.  Read before the window is built:
        # the sync layer consumes what it hands to the model, so a depth of 0
        # afterwards is normal and must not be mistaken for a starved link -
        # which is why the frame age, not the depth, is the freshness signal.
        self.assertEqual(
            {"cam1": 2, "cam2": 2, "cam3": 2, "cam4": 2},
            provider.cimg_health_view()["queue_depth"],
        )
        provider.latest_window(sync_timeout_s=1.0)

        view = provider.cimg_health_view()
        # No thread and no socket here - ingest_bytes is driven directly - so
        # the link must not claim to be connected just because frames arrived.
        self.assertFalse(view["connected"])
        self.assertEqual(0, view["sessions"])
        self.assertEqual("capture_time_ns", view["timestamp_mode"])
        self.assertEqual(2, view["groups_ingested"])
        self.assertEqual(2, view["groups_total"])
        self.assertEqual(8, view["images_total"])
        # Protocol errors are reported as counts plus the last reason, not a
        # bare total: "the link is resyncing" and "the vehicle sent a version we
        # do not speak" need different answers.
        self.assertEqual(
            {
                "packets_skipped": 0,
                "resyncs": 0,
                "discarded_bytes": 0,
                "last_error": None,
            },
            view["protocol_errors"],
        )
        self.assertEqual(0, view["decode_errors"])
        self.assertEqual(
            {"cam1": 0, "cam2": 0, "cam3": 0, "cam4": 0},
            view["queue_depth"],
            "the window build consumed the frames it was handed",
        )
        # Capacity comes from CimgConfig.queue_frames (default 60 = 6 s at the
        # vehicle's 10 groups/s), not from the two groups this test sent.
        self.assertEqual(60, view["queue_max_frames"])
        self.assertEqual(60, provider.buffers[0].max_size)
        self.assertLess(view["frame_age_ms"], 1000)
        self.assertEqual(4, len(view["cameras"]))
        self.assertEqual([1, 2, 3, 4], [c["camera_id"] for c in view["cameras"]])
        self.assertTrue(all(c["health"] == "HEALTHY" for c in view["cameras"]))

        # The old flat shape has to keep working: tools/check_chain.sh diffs
        # frames_total per reader to derive fps, and the takeover layer counts
        # the rows.  Four cameras under CIMG is still four rows.
        stats = provider.stats()
        self.assertEqual(4, len(stats["readers"]))
        # Vehicle slot order here, model order in the window: this list is the
        # mapping itself (camera_id 1..4 -> model index), and sorting it would
        # destroy the only place the correspondence is written down.
        self.assertEqual(list(self.CIMG_INDICES), stats["cimg"]["camera_indices"])
        self.assertEqual(
            list(self.CIMG_INDICES), [row["camera_index"] for row in stats["readers"]]
        )
        self.assertEqual([1, 2, 3, 4], [row["camera_id"] for row in stats["readers"]])
        self.assertEqual(
            ["cam1", "cam2", "cam3", "cam4"],
            list(stats["cimg"]["queues"]),
        )
        self.assertEqual(4, len(provider.camera_health()))
        self.assertEqual((), provider.unhealthy_readers())

    def test_a_decode_failure_drops_the_group_and_still_yields_a_window(self) -> None:
        """One bad camera in one group must not stall the window for good.

        The vehicle re-sends all four cameras every group, so dropping the
        group costs 200 ms; holding the group instead would leave three buffers
        one group ahead of the fourth forever, and the sync gate would refuse
        every window after that.
        """
        provider = self._provider(frames_per_camera=2)
        reader = provider.readers[0]
        sequences = {cam: 499 for cam in CAMERAS}
        jpegs = {cam: jpeg_bytes(cam, 499) for cam in CAMERAS}
        jpegs[4] = jpeg_bytes(0, 499)  # camera_id 0 -> the stub decoder fails
        reader.ingest_bytes(
            packet((_unix_now_us() - 400_000) * 1000, sequences=sequences, jpegs=jpegs)
        )
        self.assertEqual(1, reader.stats()["decode_errors"])
        self.assertEqual([0, 0, 0, 0], [b.depth() for b in provider.buffers])

        self._feed(provider, 2)
        self.assertIsNotNone(provider.latest_window(sync_timeout_s=1.0))
        self.assertEqual(1, reader.stats()["decode_errors"])


if __name__ == "__main__":
    unittest.main()
