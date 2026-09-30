"""Camera synchronization and window assembly ported from the tested
zhoudong backend pipeline (app/services/camera/*).

Differences from upstream:
* timestamps are the CIMG packet's capture_time_ns, converted to microseconds;
* the caller is also the inference loop, so this provider exposes a
  ``latest_window`` behaviour that skips stale frames after slow inference.

The final model-visible window matches the tested format:
``(N_camera, T, 3, H, W)`` uint8 RGB tensors plus per-camera Alpamayo indices.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from camera.cimg_source import (
    CimgCameraSource,
    CimgConfig,
)
from camera.models import CameraWindow, FrameEntry
from time_sync import TimestampManager

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Bounded, time-ordered buffer for one camera.

    ``maxlen`` is the *only* back-pressure policy in the camera path: a full
    buffer evicts its oldest frame on the next ``put``.  That is deliberate for
    the CIMG link, where the vehicle pushes at a fixed rate and inference takes
    ten seconds - the newest frames are the only ones worth keeping, and an
    unbounded queue would instead accumulate a lag the model would then be fed.
    ``dropped`` counts those evictions so /health can show them.

    ``lock`` exists so the four cameras of one CIMG packet can be written under
    one shared lock, making a group atomic for every reader: a partially
    written group would let the synchronizer pair camera 1's frame from group N
    with camera 4's frame from group N-1, which is a silently time-warped model
    input.  It must be a re-entrant lock (``threading.RLock``) because the
    writer holds it and then calls ``put``, which acquires it again.  The
    default is a private non-reentrant lock, which is enough for one writer.
    """

    def __init__(
        self,
        max_size: int = 240,
        lock: threading.RLock | None = None,
    ) -> None:
        self._queue: deque[FrameEntry] = deque(maxlen=max_size)
        self._lock = lock if lock is not None else threading.Lock()
        self._dropped = 0

    def put(self, packet: FrameEntry) -> None:
        with self._lock:
            if self._queue and packet.timestamp_us < self._queue[-1].timestamp_us:
                raise ValueError("frame timestamps must be non-decreasing")
            if (
                self._queue.maxlen is not None
                and len(self._queue) == self._queue.maxlen
            ):
                self._dropped += 1
            self._queue.append(packet)

    @property
    def dropped(self) -> int:
        """Frames evicted because the buffer was full."""
        with self._lock:
            return self._dropped

    @property
    def max_size(self) -> int | None:
        """Capacity, for reporting queue depth as ``n/max`` in /health."""
        return self._queue.maxlen

    def depth(self) -> int:
        """How many frames are buffered right now."""
        with self._lock:
            return len(self._queue)

    def snapshot(self) -> list[FrameEntry]:
        with self._lock:
            return list(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def discard_older_than(self, timestamp_us: int) -> None:
        with self._lock:
            while self._queue and self._queue[0].timestamp_us < timestamp_us:
                self._queue.popleft()

    def consume_through(self, timestamp_us: int) -> None:
        with self._lock:
            while self._queue:
                packet = self._queue.popleft()
                if packet.timestamp_us == timestamp_us:
                    break

    @property
    def oldest(self) -> FrameEntry | None:
        with self._lock:
            return self._queue[0] if self._queue else None

    @property
    def latest(self) -> FrameEntry | None:
        with self._lock:
            return self._queue[-1] if self._queue else None


@dataclass
class SynchronizedGroup:
    target_timestamp_us: int
    packets: dict[int, FrameEntry]
    max_skew_us: int


class CameraSynchronizer:
    """Select the closest frame in each camera for a common target timestamp.

    Ported from app/services/camera/synchronizer.py in the tested backend.
    """

    def __init__(
        self,
        buffers: list[FrameBuffer],
        *,
        max_target_error_us: int = 300_000,
        max_cross_camera_skew_us: int = 300_000,
        poll_interval_s: float = 0.002,
        group_lock: threading.RLock | None = None,
    ) -> None:
        self.buffers = buffers
        self.max_target_error_us = max_target_error_us
        self.max_cross_camera_skew_us = max_cross_camera_skew_us
        self.poll_interval_s = poll_interval_s
        self.group_lock = group_lock

    def wait_for_initial_target(
        self, timeout_s: float, back_off_us: int = 0
    ) -> int:
        """First target time: the newest instant every camera already covers.

        Anchoring at each buffer's *oldest* frame (the previous behaviour)
        starts the assembler ~11 s in the past (240 frames at 22 fps).  Each
        failed target only advances 0.1 s and costs the full sync timeout, so
        the pipeline crawled forward for minutes and then rejected the result
        as "too old" - measured 2026-09-11: 20 minutes, four healthy cameras,
        zero windows, zero inferences.  ``back_off_us`` moves the anchor back
        far enough that the whole T-frame window is already buffered.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.group_lock if self.group_lock is not None else nullcontext():
                newest = [buf.latest for buf in self.buffers]
            if all(p is not None for p in newest):
                return (
                    min(int(p.timestamp_us) for p in newest)  # type: ignore[misc]
                    - int(back_off_us)
                )
            time.sleep(self.poll_interval_s)
        raise TimeoutError("timed out waiting for initial camera frames")

    def get_group(
        self,
        target_timestamp_us: int,
        *,
        timeout_s: float,
    ) -> SynchronizedGroup | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # The CIMG writer uses the same lock for its four-buffer commit.
            # Hold it for all snapshots, selection and consumption so a new
            # group cannot land halfway through this four-camera read.
            with self.group_lock if self.group_lock is not None else nullcontext():
                candidates: dict[int, FrameEntry] = {}
                all_reached = True
                for cam_idx, buf in enumerate(self.buffers):
                    buf.discard_older_than(
                        target_timestamp_us - self.max_target_error_us
                    )
                    frames = buf.snapshot()
                    if not frames:
                        all_reached = False
                        break
                    if frames[-1].timestamp_us < target_timestamp_us:
                        all_reached = False
                        break
                    nearest = min(
                        frames,
                        key=lambda item: abs(item.timestamp_us - target_timestamp_us),
                    )
                    if (
                        abs(nearest.timestamp_us - target_timestamp_us)
                        > self.max_target_error_us
                    ):
                        all_reached = False
                        break
                    candidates[cam_idx] = nearest
                if all_reached and len(candidates) == len(self.buffers):
                    stamps = [p.timestamp_us for p in candidates.values()]
                    max_skew = max(stamps) - min(stamps)
                    if max_skew <= self.max_cross_camera_skew_us:
                        for cam_idx, packet in candidates.items():
                            self.buffers[cam_idx].consume_through(
                                packet.timestamp_us
                            )
                        return SynchronizedGroup(
                            target_timestamp_us=target_timestamp_us,
                            packets=candidates,
                            max_skew_us=max_skew,
                        )
            time.sleep(self.poll_interval_s)
        return None


class CameraWindowProvider:
    """End-to-end window provider: threads -> buffers -> sync -> T-group window.

    Follows MultiCameraPipeline/MultiCameraWindowAssembler from the tested
    backend but adds the ``latest_window`` behaviour needed when inference is
    slower than the camera rate.
    """

    def __init__(
        self,
        cimg: CimgConfig,
        *,
        frames_per_camera: int = 4,
        snapshot_keep: int = 600,
        target_fps: float = 10.0,
        max_camera_skew_us: int = 300_000,
        window_max_age_ms: int = 2000,
        timestamp_manager: TimestampManager | None = None,
    ) -> None:
        if not isinstance(cimg, CimgConfig):
            raise ValueError("CIMG camera input requires a CimgConfig")
        if cimg.group_fps <= 0:
            raise ValueError("CIMG group_fps must be greater than zero")
        self.cimg = cimg
        self.frames_per_camera = frames_per_camera
        self.target_fps = target_fps
        self.period_us = round(1_000_000 / target_fps)
        self.max_camera_skew_us = max_camera_skew_us
        self.timestamp_manager = timestamp_manager or TimestampManager()
        self.window_max_age_us = max(
            int(window_max_age_ms) * 1000,
            200_000,
        )
        self._last_stale_log_at = 0.0
        self._snapshot_dir = os.environ.get("ALPAMAYO_SNAPSHOT_DIR", "").strip()
        self._snapshot_keep = int(snapshot_keep)
        self._snapshot_interval = float(
            os.environ.get("ALPAMAYO_SNAPSHOT_INTERVAL", "1.0")
        )
        self._snapshot_labels = [
            x.strip()
            for x in os.environ.get(
                "ALPAMAYO_CAMERA_LABELS",
                "front_cam,right_cam,left_cam,back_cam",
            ).split(",")
            if x.strip()
        ]
        self._last_snapshot_at = 0.0

        self.window_builds = 0
        self.refusals: Counter[str] = Counter()
        self.last_window_built_at = 0.0
        self._stats_interval_s = max(
            float(os.environ.get("ALPAMAYO_CAMERA_STATS_INTERVAL_S", "10")), 0.0
        )
        self._stats_thread: threading.Thread | None = None

        # Four buffers, one shared re-entrant lock.  The lock is what makes
        # a packet's four frames land together: without it the window
        # builder can read camera 1 from group N and camera 4 from group
        # N-1, and every freshness check still passes because both frames
        # are individually new.
        self.camera_indices = [int(i) for i in cimg.camera_indices]
        # The reader decodes at the config's size, so the window builder
        # must agree with it - otherwise _build resizes every frame again
        # and memory scales with the decode size, not the model size.
        self.target_size = cimg.target_size
        # Do not sample faster than the camera input can supply.  This caps
        # only the model-window sampling grid; inference output cadence is
        # governed independently by the pipeline loop.
        group_period_us = round(1_000_000 / cimg.group_fps)
        self.period_us = max(self.period_us, group_period_us)
        self._group_lock = threading.RLock()
        self.buffers = [
            FrameBuffer(max_size=cimg.queue_frames, lock=self._group_lock)
            for _ in range(cimg.camera_count)
        ]
        self.readers = [
            CimgCameraSource(
                self.buffers,
                config=cimg,
                timestamp_manager=self.timestamp_manager,
                group_lock=self._group_lock,
            )
        ]
        self.synchronizer = CameraSynchronizer(
            self.buffers,
            max_target_error_us=self._target_error_us(max_camera_skew_us),
            max_cross_camera_skew_us=self._target_error_us(max_camera_skew_us),
            group_lock=self._group_lock,
        )
        self._groups: deque[SynchronizedGroup] = deque(maxlen=frames_per_camera)
        self._next_target_us: int | None = None
        self._last_returned_key: tuple[int, ...] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _target_error_us(self, max_camera_skew_us: int) -> int:
        """How far a target may sit from a frame and still match.

        Derived from the configured CIMG input cadence, and only on that link.
        Half an input period covers the worst phase offset between the model
        sampling grid and real camera groups.

        Half the group period is the right size: large enough that every target
        has a frame (the worst case is landing midway between two groups), and
        still below the period itself, so a target can never be within
        tolerance of two *different* groups.  The cross-camera skew gate keeps
        its own meaning because all four frames of a group are written in one
        atomic step, so a matched group really is one acquisition.
        """
        contained = int(round(500_000.0 / self.cimg.group_fps))
        return max(max_camera_skew_us, contained)

    def _snap_target_us(self, target_us: int) -> int:
        """CIMG only: aim at a real group instead of a grid instant.

        A fixed sampling grid can drift in phase against the vehicle push.
        When a target lands near the midpoint between two groups, the
        per-camera "nearest frame" search can settle on group N for one camera
        and group N+1 for another, because each frame inside a group has its
        own capture time and so shifts the midpoint by a few milliseconds.  The
        cross-camera skew gate catches that and refuses - which is the right
        outcome, but a refusal costs a whole sync timeout.

        Aiming at a timestamp that is actually buffered removes the drift: the
        nearest frame is then the same group for all four cameras, at a
        distance no larger than the intra-group capture span (a few ms).
        """
        if not self.buffers:
            return target_us
        stamps = [int(frame.timestamp_us) for frame in self.buffers[0].snapshot()]
        if not stamps:
            return target_us
        return min(stamps, key=lambda stamp: abs(stamp - target_us))

    def _check_reader_sessions(self) -> bool:
        sessions = tuple(getattr(reader, "restart_count", 0) for reader in self.readers)
        previous = getattr(self, "_reader_sessions", sessions)
        self._reader_sessions = sessions
        if sessions != previous:
            self._invalidate_stale("decoder session changed; discarding assembled groups")
            return False
        return True

    def start(self) -> None:
        self._stop.clear()
        for reader in self.readers:
            reader.start()
        if self._stats_interval_s > 0 and self._stats_thread is None:
            self._stats_thread = threading.Thread(
                target=self._stats_logger,
                daemon=True,
                name="camera-stats",
            )
            self._stats_thread.start()

    def stop(self) -> None:
        """Set every stop flag.  Never blocks.

        This is shutdown's phase-A call, and it is deliberately only flags: the
        drain has a budget and this method must fit in none of it.  Joining the
        CIMG reader is the later step, because that one has a cost.
        """
        self._stop.set()
        for reader in self.readers:
            reader.stop()

    def join(self, timeout_s: float = 0.0) -> None:
        """Wait for the reader threads and the stats thread, bounded.

        Per-reader timeout rather than one shared pool, so a single stuck reader
        cannot consume the whole budget and leave the others unjoined.  Returns
        after ``timeout_s`` per component whatever happens: the threads are
        daemons and the process exit is the backstop.

        A reader whose thread was never started is skipped rather than joined:
        ``Thread.join()`` raises ``RuntimeError`` for it, and shutdown reaches
        that state whenever ``start()`` did not run - a startup that failed
        before ``cameras.start()``, or an ASGI server that gave up during it.
        Catching the raise would still be wrong here: the blanket ``except``
        below logs it at ERROR, which fills the shutdown log with one traceback
        per camera for a provider that was simply never running.
        """
        stats_thread = self._stats_thread
        if stats_thread is not None:
            if stats_thread.ident is not None:
                stats_thread.join(timeout=min(timeout_s, 0.5))
            self._stats_thread = None
        for reader in self.readers:
            join = getattr(reader, "join", None)
            if join is None:
                continue
            if getattr(reader, "ident", None) is None:
                continue
            try:
                join(timeout_s)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.exception("failed to join reader %s", reader)

    def alive_readers(self) -> tuple[str, ...]:
        """Readers whose thread is still running.  Diagnostics for shutdown."""
        return tuple(
            str(getattr(reader, "source", reader))
            for reader in self.readers
            if getattr(reader, "is_alive", lambda: False)()
        )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def _cimg_stats(self) -> dict:
        """The single CIMG reader's report, with its four cameras flattened.

        ``readers`` must stay a list with one entry per camera: the decision
        layer compares its length against ``expected_camera_count`` and
        declares ``camera_status_unavailable`` otherwise, and
        ``tools/check_chain.sh`` reads ``camera_stats.readers[].frames_total``.
        The one-reader-four-cameras shape of the link is reported alongside it
        under ``cimg``, so nothing existing has to change shape to read the
        connection state.
        """
        reader = self.readers[0]
        report = reader.stats()
        readers = list(report.pop("cameras", []))
        cimg = dict(report)
        cimg["camera_indices"] = [int(i) for i in self.camera_indices]
        cimg["queues"] = {
            f"cam{row['camera_id']}": {
                "depth": row["queue_depth"],
                "max": self.buffers[row["camera_id"] - 1].max_size,
                "dropped": row["queue_dropped"],
            }
            for row in readers
        }
        return {"readers": readers, "cimg": cimg}

    def reported_camera_count(self) -> int:
        """How many rows ``stats()["readers"]`` contains.

        The takeover layer compares this length with ``expected_camera_count``
        and reports ``camera_status_unavailable`` on a mismatch.  CIMG has one
        socket and one row per camera, so the count is ``len(camera_indices)``.
        """
        return len(self.camera_indices)

    def stats(self) -> dict:
        now = time.monotonic()
        report = {
            "window_builds": self.window_builds,
            "refusals": dict(self.refusals),
            "last_window_age_ms": (
                round((now - self.last_window_built_at) * 1000.0, 1)
                if self.last_window_built_at
                else None
            ),
        }
        report.update(self._cimg_stats())
        return report

    def _stats_line(self) -> str:
        now = time.monotonic()
        window_age = (
            f"{(now - self.last_window_built_at) * 1000.0:.0f}ms"
            if self.last_window_built_at
            else "n/a"
        )
        refusals = ",".join(
            f"{k.split(':')[-1].strip()}={v}" for k, v in self.refusals.items()
        )
        return (
            self._cimg_stats_line()
            + f" || windows={self.window_builds} last_window_age={window_age}"
            + (f" refusals[{refusals}]" if refusals else "")
        )

    def cimg_health_view(self) -> dict:
        """The CIMG link as an operator reads it.

        Field names here are the ones this link is *diagnosed* by, in the order
        they are checked: whether the vehicle is connected, whether frames are
        arriving and how fast, how old the newest one is, what is queued, and
        then the two error families that need different people - a protocol
        error is a link or a vehicle-encoder problem, a decode error is a
        payload problem.

        Returned as an always-present key rather than a conditional one so a
        dashboard does not have to distinguish "no CIMG link" from "field
        missing", which look identical in a scrape.  The runner is absent only
        before startup; once the provider exists this view is the live link.
        """
        row = self.readers[0].stats()
        per_camera = row["cameras"]
        return {
            "connection_state": row["connection_state"],
            "connected": row["connected"],
            "peer": row["peer"],
            "sessions": row["sessions"],
            "receive_fps": row["receive_fps"],
            "groups_total": row["groups_total"],
            # Received minus ingested is the backlog the receiver chose not to
            # replay; shown next to the total so the gap is never a mystery.
            "groups_ingested": row["groups_ingested"],
            "images_total": row["images_total"],
            "images_received": row["images_received"],
            "bytes_total": row["bytes_total"],
            "last_group_time_ns": row["last_group_time_ns"],
            "last_group_latency_ms": row["last_group_latency_ms"],
            "last_capture_span_ms": row["last_capture_span_ms"],
            # The stalest camera's frame age: one number that answers "is the
            # model about to be handed stale images".
            "frame_age_ms": row["frame_age_ms"],
            "queue_max_frames": self.buffers[0].max_size,
            "queue_depth": {
                f"cam{entry['camera_id']}": entry["queue_depth"]
                for entry in per_camera
            },
            "protocol_errors": row["protocol_errors"],
            "decode_errors": row["decode_errors"],
            "timestamp_errors": row["timestamp_errors"],
            "timestamp_mode": row["timestamp_mode"],
            "read_timeouts": row["read_timeouts"],
            "groups_dropped": row["groups_dropped"],
            "parser_buffered_bytes": row["parser_buffered_bytes"],
            "cameras": [
                {
                    "camera_id": entry["camera_id"],
                    "camera_index": entry["camera_index"],
                    "health": entry["health"],
                    "sequence": entry["sequence"],
                    "sequence_gaps": entry["sequence_gaps"],
                    "frame_age_ms": entry["frame_age_ms"],
                    "queue_depth": entry["queue_depth"],
                    "queue_dropped": entry["queue_dropped"],
                    "frames_total": entry["frames_total"],
                    "last_capture_time_ns": entry["last_capture_time_ns"],
                    "last_timestamp_us": entry["last_timestamp_us"],
                }
                for entry in per_camera
            ],
        }

    def _cimg_stats_line(self) -> str:
        """One line for the CIMG link: connection first, then each camera.

        Connection state comes first because on this link it is the thing that
        explains everything after it: four stale cameras have one cause, and it
        is usually whether the vehicle is still connected.
        """
        report = self.readers[0].stats()
        connection = (
            f"cimg[state={report['connection_state']},"
            f"peer={report['peer'] or '-'},"
            f"fps={report['receive_fps']},"
            f"groups={report['groups_ingested']}/{report['groups_total']},"
            f"sessions={report['sessions']},"
            f"proto_err={report['protocol_errors']['packets_skipped']}"
            f"/{report['protocol_errors']['resyncs']}resync,"
            f"decode_err={report['decode_errors']},"
            f"ts_err={report['timestamp_errors']},"
            f"dropped={report['groups_dropped']['backlog']}backlog"
            f"/{report['groups_dropped']['partial']}partial]"
        )
        cameras = ",".join(
            f"cam{row['camera_id']}->{row['camera_index']}"
            f"={row['health']},"
            f"age={'n/a' if row['frame_age_ms'] is None else str(row['frame_age_ms']) + 'ms'},"
            f"seq={row['sequence']},gaps={row['sequence_gaps']},"
            f"q={row['queue_depth']}/{self.buffers[row['camera_id'] - 1].max_size}"
            for row in report["cameras"]
        )
        return f" | {connection} | {cameras}"

    def _stats_logger(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._stats_interval_s)
            if self._stop.is_set():
                break
            try:
                print(f"[camera-stats] {self._stats_line()}", flush=True)
            except Exception:  # noqa: BLE001 - diagnostics are best-effort
                pass

    def camera_health(self) -> dict[str, str]:
        """Per-camera health for supervision (``<backend>#cam<id>`` -> state).

        Four entries, keyed by the strings the reader reports as its
        per-camera ``source``.
        """
        return {
            str(row["source"]): str(row["health"])
            for row in self.readers[0].stats().get("cameras", [])
        }

    def unhealthy_readers(self) -> tuple[str, ...]:
        """Sources that lost their stream and are not delivering fresh frames.

        The one definition of the rule: ``_has_unhealthy_reader`` below asks this
        rather than restating it, so the window-refusal decision and readiness
        can never disagree about which readers are bad.
        """
        # Per camera, not per connection: a stalled stream still holds a
        # healthy TCP connection, so "connected" says nothing about whether
        # frames are arriving.
        return tuple(
            str(row["source"])
            for row in self.readers[0].stats().get("cameras", [])
            if str(row["health"]) in ("STALE", "FAILED", "RESTARTING")
        )

    def _has_unhealthy_reader(self) -> bool:
        """Any reader that lost its stream and is not delivering fresh frames."""
        return bool(self.unhealthy_readers())

    def _window_too_old(self) -> bool:
        groups = list(self._groups)
        if not groups:
            return False
        latest_target_us = max(int(g.target_timestamp_us) for g in groups)
        age_us = self._now_us() - latest_target_us
        return age_us > self.window_max_age_us

    def _window_arrival_age_us(self) -> int | None:
        """Age of the stalest camera's newest frame on the arrival clock."""
        if not self._groups:
            return None
        arrivals = [
            getattr(packet, "arrival_us", None)
            for packet in self._groups[-1].packets.values()
        ]
        if any(value is None for value in arrivals):
            return None
        oldest_latest_arrival = min(int(value) for value in arrivals)
        return max(0, time.monotonic_ns() // 1000 - oldest_latest_arrival)

    def _window_key(self) -> tuple[int, ...]:
        """Identity of the newest synchronized group across all cameras."""
        if not self._groups:
            return ()
        latest = self._groups[-1]
        return tuple(
            int(latest.packets[index].timestamp_us)
            for index in sorted(latest.packets)
        )

    def _now_us(self) -> int:
        """Current Unix time, the same domain as capture_time_ns."""
        return time.time_ns() // 1000

    def _invalidate_stale(self, reason: str) -> None:
        # Drop cached groups so that when a camera recovers it cannot be mixed
        # with old frames into a hybrid, time-warped window.
        self._groups.clear()
        self._next_target_us = None
        self.refusals[reason] += 1
        now = time.monotonic()
        if now - self._last_stale_log_at >= 2.0:
            print(f"[camera-sync] {reason}", flush=True)
            self._last_stale_log_at = now

    def _read_group(
        self,
        *,
        timeout_s: float,
        advance_on_miss: bool = True,
    ) -> bool:
        if self._next_target_us is None:
            self._next_target_us = self.synchronizer.wait_for_initial_target(
                timeout_s,
                back_off_us=(self.frames_per_camera - 1) * self.period_us,
            )
        target = self._next_target_us
        if self.cimg is not None:
            target = self._snap_target_us(target)
        group = self.synchronizer.get_group(target, timeout_s=timeout_s)
        if group is None:
            if advance_on_miss:
                self._next_target_us += self.period_us
                # If we are hopelessly behind (slow inference, a stall, or a
                # long unhealthy period), jump to the newest coverable target
                # instead of crawling forward one period at a time.
                newest = [buf.latest for buf in self.buffers]
                if all(p is not None for p in newest):
                    newest_us = min(int(p.timestamp_us) for p in newest)
                    if self._next_target_us < newest_us - self.window_max_age_us:
                        self._next_target_us = (
                            newest_us
                            - (self.frames_per_camera - 1) * self.period_us
                        )
            return False
        self._next_target_us = target + self.period_us
        self._groups.append(group)
        return True

    def _fill_window(self, timeout_s: float) -> bool:
        while len(self._groups) < self.frames_per_camera:
            if not self._read_group(timeout_s=timeout_s):
                return False
        return True

    def latest_window(
        self,
        *,
        sync_timeout_s: float = 0.5,
        catch_up_timeout_s: float = 0.02,
    ) -> CameraWindow | None:
        """Fill a first window, then consume buffered groups up to the newest."""
        if hasattr(self, "readers"):
            self._check_reader_sessions()
        if self._has_unhealthy_reader():
            self._invalidate_stale("refusing cached window: reader unhealthy")
            return None
        if not self._fill_window(sync_timeout_s):
            return None
        # Keep consuming recently available groups non-blocking.
        while len(self._groups) == self.frames_per_camera:
            if not self._read_group(timeout_s=catch_up_timeout_s):
                break
        arrival_age_us = self._window_arrival_age_us()
        if arrival_age_us is not None and arrival_age_us > self.window_max_age_us:
            self._invalidate_stale(
                f"refusing cached window: frame age {arrival_age_us / 1000.0:.0f}ms"
            )
            return None
        if self._window_too_old():
            self._invalidate_stale("refusing cached window: too old")
            return None
        key = self._window_key()
        if key and key == self._last_returned_key:
            self.refusals["refusing cached window: already returned"] += 1
            return None
        window = self._build()
        if window is not None:
            self._last_returned_key = key
        return window

    def _build(self) -> CameraWindow | None:
        if hasattr(self, "readers") and not self._check_reader_sessions():
            return None
        groups = list(self._groups)
        if len(groups) != self.frames_per_camera:
            return None
        if any(packet.timestamp_us < 946_684_800_000_000
               for group in groups for packet in group.packets.values()):
            self._invalidate_stale(
                "refusing non-Unix timestamps in capture_time_ns window"
            )
            return None
        self.window_builds += 1
        self.last_window_built_at = time.monotonic()

        # Consistent ordering: sort cameras by model index.
        order = np.argsort(self.camera_indices)
        indices = [self.camera_indices[i] for i in order]
        cam_frames: list[torch.Tensor] = []
        cam_frame_ids: list[list[str | None]] = []
        cam_frame_times: list[list[int]] = []
        cam_frame_arrivals: list[list[int | None]] = []
        cam_jpegs: list[list[bytes | None]] = []
        actual_latest_us = 0
        target_times: list[int] = []

        for cam_pos in order:
            temporal: list[torch.Tensor] = []
            frame_ids: list[str | None] = []
            frame_times: list[int] = []
            frame_arrivals: list[int | None] = []
            frame_jpegs: list[bytes | None] = []
            camera_index = int(self.camera_indices[cam_pos])
            for group in groups:
                entry = group.packets[cam_pos]
                pts = getattr(entry, "pts", None)
                # A CIMG frame carries the vehicle's own frame number, which is
                # a better identity than a PES PTS - and on that link there is
                # no PTS at all, so without this every frame would be anonymous.
                frame_id = getattr(entry, "frame_id", None)
                frame_ids.append(
                    frame_id
                    if frame_id is not None
                    else (f"cam{camera_index}:{int(pts)}" if pts is not None else None)
                )
                frame_times.append(int(entry.timestamp_us))
                frame_arrivals.append(getattr(entry, "arrival_us", None))
                frame_jpegs.append(getattr(entry, "jpeg_bytes", None))
                # CIMG frames are decoded straight to the model size, so this
                # resize is a no-op for them; skipping it avoids copying every
                # frame of every window a second time.
                frame = entry.bgr
                if (int(frame.shape[1]), int(frame.shape[0])) != self.target_size:
                    frame = cv2.resize(
                        frame,
                        self.target_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                resized = frame
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                temporal.append(
                    torch.from_numpy(np.ascontiguousarray(rgb))
                    .permute(2, 0, 1)
                    .contiguous()
                )
                actual_latest_us = max(actual_latest_us, int(entry.timestamp_us))
            cam_frames.append(torch.stack(temporal, dim=0))
            cam_frame_ids.append(frame_ids)
            cam_frame_times.append(frame_times)
            cam_frame_arrivals.append(frame_arrivals)
            cam_jpegs.append(frame_jpegs)

        target_times = [g.target_timestamp_us for g in groups]
        latest_arrivals = [
            arrivals[-1]
            for arrivals in cam_frame_arrivals
            if arrivals and arrivals[-1] is not None
        ]
        frame_age_us = (
            max(
                0,
                time.monotonic_ns() // 1000
                - min(int(value) for value in latest_arrivals),
            )
            if len(latest_arrivals) == len(cam_frame_arrivals)
            else None
        )
        if self._snapshot_dir:
            self._save_snapshots(groups, order)
        return CameraWindow(
            t0_us=actual_latest_us,
            frames=torch.stack(cam_frames, dim=0),
            camera_indices=indices,
            timestamps_us=target_times,
            frame_ids=cam_frame_ids,
            frame_timestamps_us=cam_frame_times,
            frame_arrivals_us=cam_frame_arrivals,
            frame_age_us=frame_age_us,
            jpeg_bytes=cam_jpegs,
        )

    def _save_snapshots(self, groups: list, order: list[int]) -> None:
        now = time.monotonic()
        if now - self._last_snapshot_at < self._snapshot_interval:
            return
        try:
            latest = groups[-1]
            bucket_us = int(latest.target_timestamp_us)
            # Monotonic receive timestamps are not Unix time; use wall time for
            # a human-readable directory until vehicle timestamp calibration.
            if bucket_us < 1_000_000_000_000_000:
                bucket_us = int(time.time() * 1_000_000)
            bucket_sec, bucket_frac_us = divmod(bucket_us, 1_000_000)
            bucket_dir = os.path.join(
                self._snapshot_dir,
                time.strftime("%Y%m%d_%H%M%S", time.localtime(bucket_sec))
                + f"_{bucket_frac_us:06d}",
            )
            os.makedirs(bucket_dir, exist_ok=True)
            for cam_pos in order:
                entry = latest.packets[cam_pos]
                label = (
                    self._snapshot_labels[cam_pos]
                    if cam_pos < len(self._snapshot_labels)
                    else f"cam_{cam_pos}"
                )
                path = os.path.join(bucket_dir, f"{label}.jpg")
                cv2.imwrite(path, np.ascontiguousarray(entry.bgr))
            self._last_snapshot_at = now
        except Exception as exc:  # noqa: BLE001
            print(f"[camera-sync] snapshot failed: {exc!r}", flush=True)


# CameraWindow and FrameEntry now live in camera.models and are re-exported
# here so the historical ``from camera.sync import CameraWindow, FrameEntry``
# import path keeps working.
__all__ = ["CameraWindow", "CameraWindowProvider", "FrameEntry"]
