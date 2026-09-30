"""The CIMG receiver: one TCP connection carrying four cameras as JPEG.

This is the camera input the backend runs.  The vehicle pushes synchronised
JPEGs.  Each image is decoded with ``decode_bgr`` (``cv2.imdecode``).  The
packet's ``capture_time_ns`` is the vehicle's UTC clock, and ``sequence`` is
the vehicle's per-camera frame number.

Four things this module is careful about, because each of them has bitten this
link before in some form:

**A group is written whole or not at all.**  The four images of one packet are
decoded first and only then written, under one shared lock across the four
buffers.  A half-written group would let the synchronizer mix camera 1's frame
from one group with camera 4's from the previous one - a time-warped model
input that no freshness gate can detect, because every individual frame is
fresh.

**The receiver never blocks on inference.**  It decodes on its own thread and
writes into bounded buffers; the inference loop only ever reads them.  When the
buffers fill, the *oldest* frame is evicted - never the newest - so lag cannot
accumulate.  A burst that leaves more than ``max_backlog_groups`` complete
packets waiting is collapsed to the newest one, because replaying a backlog is
the same accumulated lag by another route.

**Frames are downscaled on arrival, not in the window builder.**  At 1920x1080 a
buffered frame is 6.2 MB; four cameras x 60 frames of that is 1.5 GB.  Decoding
straight to the model size (576x320 by default) keeps the buffer at ~130 MB
and matches what
``camera.sync._build`` already expects to receive.

**Decode is in BGR.**  ``cv2.imdecode(..., IMREAD_COLOR)`` returns BGR, the same
order ``cv2.VideoCapture.read`` produces, so
``camera.sync``'s existing BGR->RGB conversion is unchanged and the model sees
the same channel order it always has.

Nothing here imports ``camera.sync``: the buffers are passed in.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from camera.cimg import (
    DEFAULT_MAX_JPEG_BYTES,
    DEFAULT_MAX_PAYLOAD_BYTES,
    SEQUENCE_REORDERED,
    SEQUENCE_REPEATED,
    CimgImage,
    CimgPacket,
    CimgStreamParser,
    sequence_check,
)
from camera.models import FrameEntry
from config.loader import default_endpoint
from time_sync import TimestampManager

#: Resolved once, at import: ``network.cimg``'s default host and port.
_DEFAULT_ENDPOINT = default_endpoint("cimg")

logger = logging.getLogger(__name__)

#: Rows kept for the rolling receive-rate estimate.  64 groups is 6.4 s at the
#: vehicle's 10 groups/s, which is long enough to be steady and short enough to
#: react to a stall.
_FPS_WINDOW = 64

#: How often the socket is polled.  The recv loop wakes this often to notice
#: ``stop()``; the semantic idle timeout is ``read_timeout_s`` below and is
#: accumulated across polls.  Kept well under the shutdown reap budget so a
#: stopping receiver is never reported as a component that failed to stop.
_POLL_S = 0.25

#: Current vehicle input cadence.  This is the camera acquisition/push rate,
#: not the model inference output rate.
DEFAULT_GROUP_FPS = 10.0


@dataclass(frozen=True)
class CimgConfig:
    """Everything the receiver needs that is not the buffers.

    Frozen and defaulted so tests, tools and the pipeline can all build one
    without a config file.
    """

    #: The listener the vehicle pushes frames to.  Defaulted from the one
    #: endpoint table (``network.cimg`` in config/config.json) rather than from
    #: a second copy of the number here: ``pipeline._cimg_config`` always passes
    #: ``settings.cimg_host``/``cimg_port``, so this default is only ever the
    #: starting point for a test or a tool, and a test or a tool that binds it
    #: should be binding the port the receiver actually uses.
    host: str = _DEFAULT_ENDPOINT.host
    port: int = _DEFAULT_ENDPOINT.port
    #: Model camera index for ``camera_id`` 1..4, in that order.  The default is
    #: the production mapping: the vehicle numbers its cameras front, right,
    #: left, back, and Alpamayo indexes front=1, right=2, left=0, back=4.
    camera_indices: tuple[int, ...] = (1, 2, 0, 4)
    #: Idle budget: no complete packet for this long and the connection is
    #: dropped so the vehicle re-establishes it.  Matches the validation tool's
    #: default and is far above the vehicle's 200 ms cadence.
    read_timeout_s: float = 15.0
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_jpeg_bytes: int = DEFAULT_MAX_JPEG_BYTES
    #: Frames held per camera before the oldest is evicted.  At the current
    #: 10 groups/s, 60 frames is six seconds of bounded history.
    queue_frames: int = 60
    #: Complete packets allowed to be waiting in one read before the older ones
    #: are dropped and only the newest is ingested.
    max_backlog_groups: int = 2
    #: Expected CIMG camera acquisition/push cadence.
    group_fps: float = DEFAULT_GROUP_FPS
    #: Size every decoded frame is reduced to before it is buffered.  This is
    #: the model input size.
    frame_width: int = 576
    frame_height: int = 320
    #: Seconds without a new frame before that camera is reported STALE.
    stale_after_s: float = 3.0
    #: Accept-socket backlog.
    listen_backlog: int = 4

    @property
    def camera_count(self) -> int:
        return len(self.camera_indices)

    @property
    def target_size(self) -> tuple[int, int]:
        return (int(self.frame_width), int(self.frame_height))

    @property
    def backend(self) -> str:
        return f"cimg://{self.host}:{self.port}"


# --------------------------------------------------------------------------- #
# JPEG decoding
# --------------------------------------------------------------------------- #


def decode_bgr(jpeg: bytes, size: tuple[int, int]) -> np.ndarray | None:
    """Decode one JPEG to a BGR uint8 array resized to ``size``.

    Returns ``None`` when the bytes are not a decodable image, which is a
    *decode* error and is counted separately from a protocol error - an
    operator needs to know whether to look at the link or at the vehicle's
    encoder.

    ``cv2`` is imported here rather than at module scope on purpose: this is
    the only function in the camera reader path that needs it, so the module
    stays importable (and the framing, queueing and lifecycle logic testable)
    without OpenCV installed.  It is a hard dependency in production - every
    other camera path already uses it.

    ``IMREAD_COLOR`` is not a detail to change casually: it is what makes the
    result BGR, the order ``camera.sync._build`` converts to RGB for the model.
    """
    import cv2  # noqa: PLC0415 - deliberate: see docstring

    array = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        return None
    if (int(frame.shape[1]), int(frame.shape[0])) != (int(size[0]), int(size[1])):
        frame = cv2.resize(
            frame,
            (int(size[0]), int(size[1])),
            interpolation=cv2.INTER_LINEAR,
        )
    return frame


# --------------------------------------------------------------------------- #
# Per-camera accounting
# --------------------------------------------------------------------------- #


@dataclass
class _CameraState:
    """What the receiver knows about one ``camera_id``."""

    camera_id: int
    camera_index: int
    slot: int
    sequence: int | None = None
    frames_total: int = 0
    frames_dropped: int = 0
    sequence_gaps: int = 0
    out_of_order: int = 0
    repeated_sequences: int = 0
    last_capture_time_ns: int | None = None
    last_timestamp_us: int | None = None
    last_arrival_us: int | None = None
    last_group_time_ns: int | None = None

    def health(self, now_monotonic_ns: int, stale_after_us: int) -> str:
        if self.last_arrival_us is None:
            return "STARTING"
        age_us = now_monotonic_ns // 1000 - int(self.last_arrival_us)
        return "STALE" if age_us > stale_after_us else "HEALTHY"


# --------------------------------------------------------------------------- #
# The receiver
# --------------------------------------------------------------------------- #


class CimgCameraSource(threading.Thread):
    """Listens for the vehicle's CIMG push and fills one buffer per camera.

    One thread does everything: it owns the listening socket, frames the
    stream, decodes the JPEGs and writes the buffers.  There is no queue
    *between* those steps, because a queue there would be a place for latency
    to hide; the only queue is the bounded per-camera buffer the window builder
    reads, and its overflow policy is "drop the oldest".

    The public surface matches what ``camera.sync.CameraWindowProvider`` uses
    on a reader - ``source``, ``health``, ``stats()``, ``stop()``,
    ``restart_count``, ``join()``, ``is_alive()`` - so the provider needs no
    special case for lifecycle, only for the fact that this one reader reports
    *four* cameras.
    """

    def __init__(
        self,
        buffers: list,
        *,
        config: CimgConfig,
        timestamp_manager: TimestampManager,
        decoder: Callable[[bytes, tuple[int, int]], np.ndarray | None] | None = None,
        group_lock: threading.RLock | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__(daemon=True, name=f"cimg-{config.port}")
        if len(buffers) != config.camera_count:
            raise ValueError(
                f"需要 {config.camera_count} 个相机缓存，实际收到 {len(buffers)} 个"
            )
        self.config = config
        self.buffers = list(buffers)
        self.timestamp_manager = timestamp_manager
        self.timestamp_mode = "capture_time_ns"
        self._decode = decoder or decode_bgr
        self._group_lock = group_lock if group_lock is not None else threading.RLock()
        self._clock_ns = clock_ns
        self._monotonic_ns = monotonic_ns

        self.source = config.backend
        self.parser = CimgStreamParser(
            max_payload_bytes=config.max_payload_bytes,
            max_jpeg_bytes=config.max_jpeg_bytes,
            expected_camera_ids=tuple(range(1, config.camera_count + 1)),
        )
        # camera_id -> slot.  camera_id is 1-based on the wire; the slot is the
        # position in ``config.camera_indices``, which is also the position in
        # the buffer list the window builder reads.
        self._slot_by_camera: dict[int, int] = {
            camera_id: slot
            for slot, camera_id in enumerate(range(1, config.camera_count + 1))
        }
        self._cameras: dict[int, _CameraState] = {
            camera_id: _CameraState(
                camera_id=camera_id,
                camera_index=int(config.camera_indices[slot]),
                slot=slot,
            )
            for camera_id, slot in self._slot_by_camera.items()
        }

        # -- connection / lifetime ----------------------------------------- #
        self._cancel = threading.Event()
        self._health_lock = threading.Lock()
        self._health = "STARTING"
        self._connection_state = "starting"
        self._last_error = ""
        self.restart_count = 0
        self.consecutive_failures = 0
        self.peer: tuple[str, int] | None = None
        self._sessions = 0

        # -- packet counters ----------------------------------------------- #
        #: Complete packets off the wire.  What ``receive_fps`` and
        #: ``/health``'s ``groups_total`` mean, and the number to compare
        #: against the vehicle's own send rate.
        self.groups_received = 0
        # Images in complete protocol-valid packets, before decode/timestamp
        # validation.  Kept separate from successfully buffered images_total.
        self.images_received = 0
        #: Packets that reached the buffers - ``groups_received`` minus the
        #: backlog drops.  Lower than it only when this thread fell behind.
        self.groups_ingested = 0
        self.images_total = 0
        self.bytes_total = 0
        self.groups_dropped_backlog = 0
        self.groups_dropped_partial = 0
        self.groups_dropped_stale_time = 0
        self.groups_dropped_out_of_order = 0
        self.decode_errors = 0
        self.timestamp_errors = 0
        self.truncated_reads = 0
        self.read_timeouts = 0
        self.resyncs = 0
        self._group_arrivals: deque[int] = deque(maxlen=_FPS_WINDOW)
        # The vehicle's own group times for the same window, used only as the
        # denominator when the arrival clock cannot supply one.
        self._group_times: deque[int] = deque(maxlen=_FPS_WINDOW)
        self.last_group_time_ns: int | None = None
        self.last_group_received_ns: int | None = None
        self.last_group_latency_ms: float | None = None
        self.last_capture_span_ms: float | None = None
        self._last_gap_log_at = 0.0
        self._last_error_log_at = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        """Set the stop flag.  Never blocks; the recv loop polls every
        ``_POLL_S`` so the thread is gone well inside the reap budget."""
        self._cancel.set()

    @property
    def health(self) -> str:
        """Worst health across the four cameras, or FAILED if unlistenable.

        Aggregated so the provider's reader-level checks (``unhealthy_readers``)
        keep their meaning: on this link the four cameras arrive together, so a
        single stale camera means the whole packet stream stalled.

        Derived on read, not cached.  The event that matters here is a stream
        that *stopped*: it produces no frames, so nothing would ever call a
        refresh, and a cached value would report HEALTHY indefinitely for a
        camera that went quiet minutes ago.  The two lifecycle states that are
        not a function of frame age - FAILED (cannot listen) and STOPPED - are
        the only stored ones.
        """
        with self._health_lock:
            stored = self._health
        if stored in ("FAILED", "STOPPED"):
            return stored
        return self._derived_health()

    def _derived_health(self) -> str:
        stale_us = int(self.config.stale_after_s * 1_000_000)
        now_ns = self._monotonic_ns()
        states = [cam.health(now_ns, stale_us) for cam in self._cameras.values()]
        if states and all(state == "HEALTHY" for state in states):
            return "HEALTHY"
        if any(state == "STALE" for state in states):
            return "STALE"
        return "STARTING"

    @property
    def connection_state(self) -> str:
        with self._health_lock:
            return self._connection_state

    def _set_health(self, health: str, state: str | None = None) -> None:
        with self._health_lock:
            self._health = health
            if state is not None:
                self._connection_state = state

    def _refresh_health(self, state: str | None = None) -> None:
        """Store the derived health, and optionally the connection state.

        A stalled stream keeps a perfectly healthy TCP connection, so health
        and connection state are separate answers: "connected" says a socket
        exists, "HEALTHY" says frames are arriving, and the recovery step for
        each is different.
        """
        self._set_health(self._derived_health(), state)

    # ------------------------------------------------------------------ #
    # The thread
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        backoff_s = 2.0
        while not self._cancel.is_set():
            server = None
            try:
                server = self._listen()
            except OSError as exc:
                # A port already in use is not transient and must be loud.
                self._note_failure(f"listen failed on {self.source}: {exc}")
                self._set_health("FAILED", "bind_failed")
                self._cancel.wait(backoff_s)
                backoff_s = min(backoff_s * 2.0, 30.0)
                continue
            backoff_s = 2.0
            self._set_health("STARTING", "listening")
            try:
                while not self._cancel.is_set():
                    conn, peer = self._accept(server)
                    if conn is None:
                        continue
                    try:
                        self._handle_connection(conn, peer)
                    except Exception as exc:  # noqa: BLE001 - one bad session
                        self._note_failure(f"connection handler error: {exc!r}")
                    finally:
                        try:
                            conn.close()
                        except OSError:
                            pass
                        self._refresh_health()
            finally:
                if server is not None:
                    try:
                        server.close()
                    except OSError:
                        pass
        self._set_health("STOPPED", "stopped")

    def _listen(self) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # SO_REUSEADDR so a restart is not blocked by the previous process's
            # TIME_WAIT sockets.  On Linux - the deployment target - it does not
            # let a second process take a port that is actively listening, so a
            # genuine conflict still raises.
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.config.host, self.config.port))
            server.listen(self.config.listen_backlog)
            # Short accept timeout so stop() is noticed promptly.
            server.settimeout(_POLL_S)
        except OSError:
            # Close it here: the caller's retry loop would otherwise leak a
            # descriptor per attempt, and a bind conflict retries forever.
            server.close()
            raise
        self._log(
            f"listening for CIMG on {self.config.host}:{self.config.port} "
            f"(cameras {sorted(self._cameras)}, "
            f"indices {list(self.config.camera_indices)}, "
            f"queue_frames={self.config.queue_frames})"
        )
        return server

    def _accept(self, server: socket.socket) -> tuple[socket.socket | None, tuple]:
        while not self._cancel.is_set():
            try:
                return server.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                self._note_failure(f"accept failed: {exc}")
                return None, ("", 0)
        return None, ("", 0)

    def _handle_connection(self, conn: socket.socket, peer: tuple) -> None:
        self.peer = (str(peer[0]), int(peer[1]))
        self._sessions += 1
        self.restart_count = self._sessions
        self.consecutive_failures = 0
        self._set_health("STARTING", "connected")
        self._log(
            f"vehicle connected peer={self.peer[0]}:{self.peer[1]} "
            f"session={self._sessions}"
        )
        # A fresh connection starts at a header: whatever was buffered belongs
        # to a packet the old connection will never finish (CLOUD_PUSH.md).
        discarded = self.parser.reset()
        if discarded:
            self._log(f"丢弃上一连接残留的 {discarded} 字节半包")
        conn.settimeout(_POLL_S)

        idle_s = 0.0
        try:
            while not self._cancel.is_set():
                try:
                    chunk = conn.recv(1 << 16)
                except socket.timeout:
                    idle_s += _POLL_S
                    if idle_s >= self.config.read_timeout_s:
                        self._note_timeout()
                        break
                    continue
                except OSError as exc:
                    self._note_failure(f"recv failed: {exc}")
                    break
                if not chunk:
                    partial = self.parser.reset()
                    if partial:
                        # The vehicle closed mid-packet.  Those bytes can never
                        # be completed, so the half packet is dropped and the
                        # next connection is expected to begin at a header.
                        self.groups_dropped_partial += 1
                        self.truncated_reads += 1
                    self._log(
                        f"车辆断开 peer={self.peer[0]}:{self.peer[1]} "
                        f"累计组数={self.groups_received}"
                        + (f" 丢弃未完成数据包 {partial} 字节" if partial else "")
                    )
                    break
                idle_s = 0.0
                self._consume(self._drain_available(conn, chunk))
        finally:
            self.peer = None
            # Back to "listening" the moment the session ends, so /health never
            # shows a connected vehicle that is no longer there.
            self._refresh_health("listening")

    def _drain_available(self, conn: socket.socket, first_chunk: bytes) -> list[CimgPacket]:
        """Parse the first read and all bytes already queued by TCP.

        A real group is much larger than the 64 KiB recv chunk.  Draining before
        decode exposes complete queued groups to the bounded backlog policy,
        while CimgStreamParser retains any final half-packet intact.
        """
        packets: list[CimgPacket] = []

        def feed(chunk: bytes) -> None:
            self.bytes_total += len(chunk)
            packets.extend(self.parser.feed(chunk))

        feed(first_chunk)
        conn.setblocking(False)
        try:
            while not self._cancel.is_set():
                try:
                    chunk = conn.recv(1 << 16)
                except (BlockingIOError, InterruptedError):
                    break
                if not chunk:
                    break
                feed(chunk)
        finally:
            conn.settimeout(_POLL_S)
        return packets

    def _note_timeout(self) -> None:
        """No complete packet within the idle budget: drop the connection.

        A half-received packet is abandoned rather than carried over, because
        the bytes that would complete it are not coming.
        """
        self.read_timeouts += 1
        partial = self.parser.reset()
        if partial:
            self.groups_dropped_partial += 1
            self._log(
                f"接收超时（{self.config.read_timeout_s:g}s 无完整数据包），"
                f"丢弃未完成数据包 {partial} 字节，关闭连接等待车辆重连"
            )
        else:
            self._log(
                f"接收超时（{self.config.read_timeout_s:g}s 无完整数据包），关闭连接"
            )
        self._note_failure("read timeout", log=False)

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #

    def _consume(self, packets: list[CimgPacket]) -> None:
        """Ingest what one read produced, collapsing any backlog.

        A single read returning several complete packets means this thread is
        behind.  The older packets are already old, so replaying them would
        only push the model's input further into the past; the newest one
        carries all the information they do.

        Every packet counted here was *received*; ``_ingest`` counts the subset
        that was *used*.  The two are reported separately because they answer
        different questions - the vehicle's send rate, and how much of it
        reached the model - and because a backlog drop is not a link fault, so
        it must not depress the number that diagnoses the link.
        """
        if not packets:
            return
        self.groups_received += len(packets)
        self.images_received += sum(len(packet.images) for packet in packets)
        now_ns = self._monotonic_ns()
        self._group_arrivals.extend([now_ns] * len(packets))
        self._group_times.extend(int(p.group_time_ns) for p in packets)
        if len(packets) > self.config.max_backlog_groups:
            skipped = len(packets) - 1
            self.groups_dropped_backlog += skipped
            self._log_throttled(
                f"积压 {len(packets)} 组，丢弃前 {skipped} 组只处理最新一组",
                "backlog",
            )
            packets = packets[-1:]
        for packet in packets:
            self._ingest(packet)

    def _ingest(self, packet: CimgPacket) -> None:
        received_ns = self._clock_ns()
        arrival_us = self._monotonic_ns() // 1000
        self.last_group_time_ns = int(packet.group_time_ns)
        self.last_group_received_ns = received_ns
        self.last_group_latency_ms = (
            received_ns - int(packet.group_time_ns)
        ) / 1e6
        self.last_capture_span_ms = packet.capture_span_ns / 1e6

        # Decode every image before writing any of them: the group is the unit
        # of consistency, so a group with one bad image must not appear in the
        # buffers at all.
        # The image travels with the frame: ``_CameraState.sequence`` is the
        # *previous* number until ``_record_frame`` advances it, so building the
        # entry from it would stamp every frame with its predecessor's sequence
        # (and the first frame of a session with None) - an off-by-one that
        # would be invisible in the frame counts and wrong in the record.
        decoded: list[tuple[_CameraState, CimgImage, np.ndarray, int]] = []
        for image in packet.images:
            state = self._cameras.get(image.camera_id)
            if state is None:
                self.decode_errors += 1
                return
            timestamp_us = self._resolve_timestamp(image)
            frame = self._decode_safely(image.jpeg_data)
            if frame is None:
                self.decode_errors += 1
                self._log_throttled(
                    f"相机 {image.camera_id} JPEG 解码失败 "
                    f"(seq={image.sequence}, {image.jpeg_size} 字节)",
                    "decode",
                )
                return
            decoded.append((state, image, frame, timestamp_us))

        out_of_order: list[_CameraState] = []
        for state, _image, _frame, timestamp_us in decoded:
            latest = self.buffers[state.slot].latest
            if latest is not None and timestamp_us <= int(latest.timestamp_us):
                out_of_order.append(state)
        if out_of_order:
            # Writing the other three would leave the four buffers disagreeing
            # about which group they hold, which is exactly the state the
            # atomic write exists to prevent.
            self.groups_dropped_out_of_order += 1
            for state in out_of_order:
                state.frames_dropped += 1
                state.out_of_order += 1
            self._log_throttled(
                "丢弃整组："
                + ", ".join(
                    f"相机 {s.camera_id} 时间戳未前进"
                    f"(seq={s.sequence})"
                    for s in out_of_order
                ),
                "out-of-order",
            )
            return

        with self._group_lock:
            for state, image, frame, timestamp_us in decoded:
                entry = FrameEntry(
                    timestamp_us=timestamp_us,
                    bgr=frame,
                    # No PES PTS on this link.  The vehicle's own frame number
                    # takes its place in the frame id, which is what the
                    # inference package records per frame.
                    pts=None,
                    arrival_us=arrival_us,
                    frame_id=f"cam{state.camera_index}:s{image.sequence}",
                    sequence=int(image.sequence),
                    camera_id=state.camera_id,
                    group_time_us=int(packet.group_time_ns) // 1000,
                    jpeg_bytes=bytes(image.jpeg_data),
                )
                self.buffers[state.slot].put(entry)

        # Count only complete groups that passed timestamp/decode validation
        # and reached all four buffers.
        self.groups_ingested += 1
        self.images_total += len(decoded)
        for state, image, _frame, timestamp_us in decoded:
            self._record_frame(state, packet, image, timestamp_us, arrival_us)

    def _resolve_timestamp(self, image) -> int:
        """The frame clock: CIMG ``capture_time_ns`` as Unix microseconds."""
        return self.timestamp_manager.resolve_camera(int(image.capture_time_ns))

    def _decode_safely(self, jpeg: bytes) -> np.ndarray | None:
        try:
            return self._decode(jpeg, self.config.target_size)
        except Exception as exc:  # noqa: BLE001 - a decoder bug is one frame
            self._log_throttled(f"JPEG 解码异常: {exc!r}", "decode-exception")
            return None

    def _record_frame(
        self,
        state: _CameraState,
        packet: CimgPacket,
        image: CimgImage,
        timestamp_us: int,
        arrival_us: int,
    ) -> None:
        check = sequence_check(state.sequence, image.sequence)
        if check.is_gap:
            # Deliberately NOT "packet loss": the vehicle numbers frames per
            # camera and skips the ones that miss its four-camera sync
            # tolerance, so a jump is often the sender's frame selection.
            state.sequence_gaps += 1
            self._log_throttled(
                f"相机 {state.camera_id} 序号跳变(+{check.gap}) "
                f"{state.sequence}->{image.sequence} "
                f"（序号缺口不等于网络丢包）",
                "sequence-gap",
            )
        elif check.status == SEQUENCE_REORDERED:
            state.out_of_order += 1
            self._log_throttled(
                f"相机 {state.camera_id} 序号乱序 "
                f"{state.sequence}->{image.sequence}",
                "sequence-order",
            )
        elif check.status == SEQUENCE_REPEATED:
            state.repeated_sequences += 1
        if not check.is_out_of_order:
            state.sequence = int(image.sequence)
        state.frames_total += 1
        state.last_timestamp_us = int(timestamp_us)
        state.last_arrival_us = int(arrival_us)
        state.last_capture_time_ns = int(image.capture_time_ns)
        state.last_group_time_ns = int(packet.group_time_ns)
        self._refresh_health()
        if self.groups_received % 25 == 1:
            self._log(
                f"组#{self.groups_received} 接收完成 "
                f"fps={self.receive_fps():.2f} "
                f"capture_span={self.last_capture_span_ms:.1f}ms "
                f"latency={self.last_group_latency_ms:.1f}ms "
                f"health={self.health}"
            )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log(self, message: str) -> None:
        logger.info("[%s] %s", self.name, message)

    def _log_throttled(self, message: str, key: str, interval_s: float = 5.0) -> None:
        now = time.monotonic()
        stamps = getattr(self, "_throttle", None)
        if stamps is None:
            stamps = self._throttle = {}
        if now - stamps.get(key, -interval_s) < interval_s:
            return
        stamps[key] = now
        self._log(message)

    def _note_failure(self, reason: str, *, log: bool = True) -> None:
        self.consecutive_failures += 1
        with self._health_lock:
            self._last_error = reason
        if log:
            self._log(f"接收失败（{reason}），连续 {self.consecutive_failures} 次")

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def receive_fps(self) -> float:
        """Groups per second over the recent window.

        Measured on arrival times, falling back to the vehicle's own group
        times when arrivals cannot supply a span.  Two arrivals in the same
        clock tick is not a hypothetical: ``monotonic_ns`` is 15.6 ms on
        Windows and this link legitimately delivers several groups in one read
        (that is the backlog case, and a reconnect drains like that too).  All
        the samples landing in one tick would otherwise report 0.0 - "nothing
        is arriving" for a link that is carrying frames, which is worse than no
        number at all.  The group times are the right second opinion: they are
        when the vehicle captured the frames, so a burst of ten groups 200 ms
        apart reports 5/s rather than dividing by zero.

        A rate from the vehicle clock is still a rate, but it is not evidence
        that frames *arrived*: ``groups_total`` and ``frame_age_ms`` are the
        fields for that, and they cannot be satisfied by a skewed sender clock.
        """
        arrivals = self._group_arrivals
        if len(arrivals) < 2:
            return 0.0
        span_s = (arrivals[-1] - arrivals[0]) / 1e9
        if span_s <= 0:
            if len(self._group_times) < 2:
                return 0.0
            span_s = (self._group_times[-1] - self._group_times[0]) / 1e9
            if span_s <= 0:
                return 0.0
        return (len(arrivals) - 1) / span_s

    def _camera_stats(self) -> list[dict]:
        stale_us = int(self.config.stale_after_s * 1_000_000)
        now_ns = self._monotonic_ns()
        with self._health_lock:
            last_error = self._last_error or None
        rows: list[dict] = []
        for state in self._cameras.values():
            age_ms = (
                round((now_ns // 1000 - state.last_arrival_us) / 1000.0, 1)
                if state.last_arrival_us is not None
                else None
            )
            rows.append(
                {
                    # The keys the decision layer and tools/check_chain.sh
                    # already read, in the shape they read them - the whole
                    # point of publishing four per-camera rows rather than one
                    # row per connection.  check_chain.sh indexes ``restarts``
                    # and ``last_error`` by name, so a row without them is a
                    # KeyError in the operator's own tool.
                    "source": f"{self.config.backend}#cam{state.camera_id}",
                    "camera_id": state.camera_id,
                    "camera_index": state.camera_index,
                    "health": state.health(now_ns, stale_us),
                    "frames_total": state.frames_total,
                    "frame_age_ms": age_ms,
                    # Both are link-level facts.  There is one socket, so a
                    # restart or a read error is not attributable to any single
                    # camera - but the row is where an operator reads a red
                    # camera, so the link's answer belongs there rather than
                    # nowhere.
                    "restarts": self.restart_count,
                    "last_error": last_error,
                    "consecutive_failures": self.consecutive_failures,
                    "timestamp_mode": self.timestamp_mode,
                    "sequence": state.sequence,
                    "sequence_gaps": state.sequence_gaps,
                    "out_of_order": state.out_of_order,
                    "repeated_sequences": state.repeated_sequences,
                    "frames_dropped": state.frames_dropped,
                    "queue_depth": self.buffers[state.slot].depth(),
                    "queue_dropped": self.buffers[state.slot].dropped,
                    "last_capture_time_ns": state.last_capture_time_ns,
                    "last_timestamp_us": state.last_timestamp_us,
                    "last_group_time_ns": state.last_group_time_ns,
                }
            )
        return rows

    def stats(self) -> dict:
        """Per-camera rows plus the connection-level counters.

        ``cameras`` is the list the window provider publishes as ``readers``:
        it has exactly one entry per camera, so the decision layer's
        ``expected_camera_count`` check still sees four.
        """
        with self._health_lock:
            state = self._connection_state
            last_error = self._last_error
        return {
            "source": self.config.backend,
            # Through the property, so a caller reading stats gets the same
            # freshness verdict as one reading ``health``.
            "health": self.health,
            "connection_state": state,
            "connected": self.peer is not None,
            "peer": None if self.peer is None else f"{self.peer[0]}:{self.peer[1]}",
            "sessions": self._sessions,
            "restarts": self.restart_count,
            "consecutive_failures": self.consecutive_failures,
            "last_error": last_error or None,
            # Requirement: connection state, receive fps, newest timestamps,
            # per-camera sequence, frame age, queue depth, and the two error
            # families kept apart.
            "receive_fps": round(self.receive_fps(), 3),
            "groups_total": self.groups_received,
            "groups_received": self.groups_received,
            "groups_ingested": self.groups_ingested,
            "images_total": self.images_total,
            "images_received": self.images_received,
            "bytes_total": self.bytes_total,
            "frames_total": self.images_total,
            "frame_age_ms": max(
                (
                    row["frame_age_ms"]
                    for row in self._camera_stats()
                    if row["frame_age_ms"] is not None
                ),
                default=None,
            ),
            "last_group_time_ns": self.last_group_time_ns,
            "last_group_latency_ms": (
                None
                if self.last_group_latency_ms is None
                else round(self.last_group_latency_ms, 1)
            ),
            "last_capture_span_ms": (
                None
                if self.last_capture_span_ms is None
                else round(self.last_capture_span_ms, 3)
            ),
            "protocol_errors": {
                "packets_skipped": self.parser.packets_skipped,
                "resyncs": self.parser.resync_count,
                "discarded_bytes": self.parser.discarded_bytes,
                "last_error": self.parser.last_error,
            },
            "decode_errors": self.decode_errors,
            "timestamp_errors": self.timestamp_errors,
            "timestamp_mode": self.timestamp_mode,
            "truncated_reads": self.truncated_reads,
            "read_timeouts": self.read_timeouts,
            "groups_dropped": {
                "backlog": self.groups_dropped_backlog,
                "partial": self.groups_dropped_partial,
                "out_of_order": self.groups_dropped_out_of_order,
                "stale_time": self.groups_dropped_stale_time,
            },
            "parser_buffered_bytes": self.parser.buffered_bytes,
            "cameras": self._camera_stats(),
        }

    # ------------------------------------------------------------------ #
    # Test / tooling helpers
    # ------------------------------------------------------------------ #

    def ingest_bytes(self, chunk: bytes) -> None:
        """Feed raw stream bytes through the live ingest path.

        Used by tests to drive framing, decoding and the queue policy without a
        socket.  Identical to what the recv loop does with each read, including
        the backlog collapse - a caller that wants every packet kept must feed
        one packet at a time or raise ``max_backlog_groups``.
        """
        self.bytes_total += len(chunk)
        self._consume(self.parser.feed(chunk))


__all__ = [
    "DEFAULT_GROUP_FPS",
    "CimgCameraSource",
    "CimgConfig",
    "decode_bgr",
]
