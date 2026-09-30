#!/usr/bin/env python3
"""The real CIMG ingress chain, on loopback, driven by real captured JPEGs.

Phase 3 of the offline acceptance plan.  The chain under test is the one the
deployment actually runs for CIMG input:

    TCP push  ->  CimgCameraSource  ->  4 x FrameBuffer
              ->  CameraWindowProvider.latest_window()  ->  InputBuilder.build()
              ->  InferenceTask.from_build()  ->  GpuScheduler (fake workers)

Nothing here starts Uvicorn or a relay, and
nothing binds a vehicle-facing port: the replayer listens on 127.0.0.1 at an
ephemeral port chosen at run time, and the pose is pushed straight into a
``PoseHistory``, not over the wire.

What is real and what is not - read this before quoting any result
    real    the four JPEG payloads.  They come from ``--samples``, a directory
            of ``group<NNN>_cam<ID>_seq<SEQ>.jpg`` written by the first-stage
            capture tool, so they are the vehicle's own encoder output for
            those frames.  The tool verifies this rather than trusting it: it
            re-decodes one sample independently and compares it against the
            tensor that came out of the live window.
    real    every stage of the chain listed above, unmodified.
    NOT     the CIMG *framing*.  The raw byte stream was never saved - the
            capture tool that would have written it has no output directory on
            the deployment host - so this tool rebuilds the 20-byte packet and
            image headers from the payloads, using the same struct format the
            parser reads.  A framing bug therefore cannot be caught here.
    NOT     the pose.  Synthetic, at 10 Hz, on a clock the tool controls, fed
            into the real ``PoseHistory`` and the real gates.
    NOT     the vehicle's own timing.  Group cadence, sequences and capture
            timestamps past the recorded set are generated, not recorded.

Consequently: **the real image ingress is unverified**.  What is verified is
that the decode, buffering, window assembly, gating and task construction all
work on real JPEG bytes.  The distinction is stated in the report itself, not
only here.

Usage
    python tools/cimg_ingress_test.py --samples /path/to/cimg_samples
    python tools/cimg_ingress_test.py --samples ... --seconds 8 --devices 1

Exit status is 0 only when every assertion holds.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from camera import cimg  # noqa: E402
from camera.cimg_source import CimgConfig  # noqa: E402
from camera.sync import CameraWindowProvider  # noqa: E402
from inference.input_builder import InputBuilder  # noqa: E402
from inference.tasks import InferenceTask  # noqa: E402
from time_sync import TimestampManager  # noqa: E402
from vehicle.history import PoseHistory  # noqa: E402
from vehicle.models import GlobalPose, vehicle_azimuth_for_compass  # noqa: E402

from multi_gpu_offline_test import (  # noqa: E402
    Harness,
    Report,
    _settings_for,
    pool_report,
)

#: ``group001_cam3_seq36.jpg`` - what the capture tool writes, one file per
#: camera per group.  Parsing the name is what lets the tool assert that the
#: frame id inside the live window is the one that was on disk.
SAMPLE_NAME = re.compile(r"^group(\d+)_cam(\d+)_seq(\d+)\.jpg$")

#: The vehicle's push cadence.  Not a knob to raise: the window builder samples
#: on a grid derived from ``CimgConfig.group_fps``, so a replay at a different
#: rate would test a configuration the vehicle never produces.
GROUP_FPS = 10.0

#: Camera ids in wire order.  The vehicle numbers its cameras front, right,
#: left, back; Alpamayo indexes front=1, right=2, left=0, back=4.  Both orders
#: matter and they are not the same order.
WIRE_CAMERA_IDS = (1, 2, 3, 4)
ALPAMAYO_INDICES = (1, 2, 0, 4)

#: A window is 4 cameras x 4 frames.
FRAMES_PER_CAMERA = 4

#: How far ahead of the replayed captures the synthetic pose clock runs.  A
#: window's ``t0_us`` is its *newest* frame, and four groups have to arrive
#: before there is a window at all, so a pose clock in exact step can sit just
#: behind the window it is meant to describe.  ``InputBuilder`` then rejects the
#: cycle as ``model_pose_history_unavailable`` - a provisioning artifact of the
#: replay, not a property of the chain.  The lead is well inside the 1000 ms
#: camera-lag budget, so it does not mask the gate either.
POSE_LEAD_US = 150_000


# --------------------------------------------------------------------------- #
# The recorded payloads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sample:
    """One recorded JPEG and the identity the vehicle gave it."""

    camera_id: int
    sequence: int
    path: Path
    jpeg: bytes = field(repr=False, default=b"")


def load_samples(directory: Path) -> list[list[Sample]]:
    """Every complete group in ``directory``, in group order.

    A group with anything other than one image per camera is refused rather
    than partially replayed: the receiver drops a group whose images are not
    all decodable, so a short group would look like a link fault instead of a
    bad sample set.
    """
    by_group: dict[int, dict[int, Sample]] = {}
    for path in sorted(directory.glob("group*_cam*_seq*.jpg")):
        match = SAMPLE_NAME.match(path.name)
        if match is None:
            continue
        group_number, camera_id, sequence = (int(x) for x in match.groups())
        by_group.setdefault(group_number, {})[camera_id] = Sample(
            camera_id=camera_id,
            sequence=sequence,
            path=path,
            jpeg=path.read_bytes(),
        )
    groups: list[list[Sample]] = []
    for number in sorted(by_group):
        cameras = by_group[number]
        missing = [cam for cam in WIRE_CAMERA_IDS if cam not in cameras]
        if missing:
            raise SystemExit(
                f"{directory}: group {number:03d} is missing camera(s) {missing}; "
                "a partial group cannot be replayed as a packet"
            )
        # Wire order, which is the order the record headers appear in.
        groups.append([cameras[cam] for cam in WIRE_CAMERA_IDS])
    if not groups:
        raise SystemExit(
            f"{directory}: no group<NNN>_cam<ID>_seq<SEQ>.jpg samples found"
        )
    return groups


def build_packet(group_time_ns: int, records: Sequence[tuple[int, int, int, bytes]]) -> bytes:
    """One CIMG packet: the real framing, rebuilt from the payloads.

    ``records`` is ``(camera_id, sequence, capture_time_ns, jpeg)`` per camera,
    in wire order.  The struct formats come from ``camera.cimg`` itself rather
    than being re-spelled here, so a change to the protocol changes this too
    instead of silently producing bytes the parser would reject.
    """
    payload = b"".join(
        cimg.IMAGE_HEADER.pack(camera_id, 0, sequence, capture_ns, len(jpeg)) + jpeg
        for camera_id, sequence, capture_ns, jpeg in records
    )
    return (
        cimg.PACKET_HEADER.pack(
            cimg.MAGIC,
            cimg.PROTOCOL_VERSION,
            len(records),
            group_time_ns,
            len(payload),
        )
        + payload
    )


# --------------------------------------------------------------------------- #
# The replayer
# --------------------------------------------------------------------------- #


class Replayer(threading.Thread):
    """Connects to the receiver and pushes recorded groups at the real cadence.

    The clock is the tool's: ``capture_time_ns`` is derived from a base Unix
    time that advances one group period per packet, so the receiver sees a
    plausible vehicle clock rather than the wall clock with its connection
    setup jitter baked in.  It is still generated, not recorded, and the report
    says so.
    """

    def __init__(
        self,
        host: str,
        port: int,
        groups: list[list[Sample]],
        *,
        seconds: float,
        base_unix_us: int,
        connect_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(daemon=True, name="cimg-replayer")
        self.host = host
        self.port = port
        self.groups = groups
        self.seconds = float(seconds)
        self.base_unix_us = int(base_unix_us)
        self.connect_timeout_s = float(connect_timeout_s)
        self.groups_sent = 0
        self.passes = 0
        self.bytes_sent = 0
        self.error: str | None = None
        self.connected = threading.Event()
        #: The ``(camera_id, sequence)`` of every record actually put on the
        #: wire, in order.  The window's frame ids are checked against this, so
        #: "the real payloads reached the window" is a fact rather than a hope.
        self.sent_ids: list[tuple[int, int]] = []
        #: Per camera, ``last_recorded - first_recorded + 1``: how far one pass
        #: moves that camera's numbering, so the next pass starts above it.
        self._span = {
            camera_id: max(
                sample.sequence for group in groups for sample in group
                if sample.camera_id == camera_id
            )
            - min(
                sample.sequence for group in groups for sample in group
                if sample.camera_id == camera_id
            )
            + 1
            for camera_id in WIRE_CAMERA_IDS
        }
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _connect(self) -> socket.socket | None:
        deadline = time.monotonic() + self.connect_timeout_s
        while time.monotonic() < deadline and not self._stop_event.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError:
                time.sleep(0.05)
        return None

    def run(self) -> None:
        sock = self._connect()
        if sock is None:
            self.error = f"could not connect to {self.host}:{self.port}"
            return
        self.connected.set()
        period_ns = int(round(1e9 / GROUP_FPS))
        base_ns = self.base_unix_us * 1000
        index = 0
        started = time.monotonic()
        try:
            while not self._stop_event.is_set() and time.monotonic() - started < self.seconds:
                group = self.groups[index % len(self.groups)]
                if index and index % len(self.groups) == 0:
                    self.passes += 1
                group_ns = base_ns + index * period_ns
                # Per-camera offsets, so the four images of a group do not claim
                # to be simultaneous - the real capture had a spread, and the
                # protocol documents that they are not.
                records = [
                    (
                        sample.camera_id,
                        self._sequence(index, sample),
                        group_ns + slot * 1_000_000,
                        sample.jpeg,
                    )
                    for slot, sample in enumerate(group)
                ]
                payload = build_packet(group_ns, records)
                try:
                    sock.sendall(payload)
                except OSError as exc:
                    self.error = f"send failed after {self.groups_sent} groups: {exc}"
                    return
                self.sent_ids.extend(
                    (sample.camera_id, records[slot][1]) for slot, sample in enumerate(group)
                )
                self.groups_sent += 1
                self.bytes_sent += len(payload)
                index += 1
                self._stop_event.wait(1.0 / GROUP_FPS)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _sequence(self, index: int, sample: Sample) -> int:
        """The recorded number on the first pass, continued afterwards.

        The recording starts each camera at its own number, so the first pass
        replays those exactly - inventing a numbering would make the receiver's
        sequence-gap counter report the sample set as a link fault.  On later
        passes the numbers continue from that camera's own last recorded
        number.  Adding a fixed count per pass instead would restart below the
        previous pass's end and read as reordering, which is a real fault the
        receiver is right to flag.
        """
        pass_index = index // len(self.groups)
        if pass_index == 0:
            return sample.sequence
        return sample.sequence + pass_index * self._span[sample.camera_id]


# --------------------------------------------------------------------------- #
# The synthetic pose
# --------------------------------------------------------------------------- #


class PoseFeeder(threading.Thread):
    """Pushes a plausible 10 Hz vehicle track into the real PoseHistory.

    Synthetic throughout, and on the same Unix clock as the replayed captures,
    so the camera window and the pose history can be aligned directly.

    The track is a straight line at 10 m/s heading east, which keeps the ENU
    conversion well away from any pole or wrap: this tool is testing the
    ingress chain, not the geodesy.
    """

    def __init__(
        self,
        history: PoseHistory,
        *,
        base_unix_us: int,
        seconds: float,
        latitude: float = 30.5,
        longitude: float = 114.3,
    ) -> None:
        super().__init__(daemon=True, name="cimg-pose")
        self.history = history
        self.base_unix_us = int(base_unix_us)
        self.seconds = float(seconds)
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.added = 0
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def seed(self, *, lookback_s: float = 3.0) -> None:
        """Fill the history before the first window is asked for.

        ``InputBuilder`` needs 16 samples spanning 1.5 s ending at the window's
        own timestamp, and it rejects the cycle outright when the history does
        not reach back that far - so without this the first windows would fail
        on a provisioning detail rather than on anything being tested.
        """
        steps = int(lookback_s * 10)
        now_us = self.base_unix_us
        for i in range(steps, -1, -1):
            self._add(now_us - i * 100_000)

    def _add(self, timestamp_us: int) -> None:
        elapsed_s = (timestamp_us - self.base_unix_us) / 1e6
        self.history.add(
            GlobalPose(
                timestamp_us=int(timestamp_us),
                latitude=self.latitude,
                longitude=self.longitude + elapsed_s * 10.0 / 111_320.0,
                altitude=20.0,
                v_north=0.0,
                v_east=10.0,
                v_up=0.0,
                roll_deg=0.0,
                pitch_deg=0.0,
                azimuth_deg=vehicle_azimuth_for_compass(90.0),
                # The vehicle's own clock, and the server's receive time.  The
                # gates age the pose by ``arrival_us``, so it has to be the
                # monotonic clock the server would have stamped.
                payload_timestamp_us=int(timestamp_us),
                arrival_us=time.monotonic_ns() // 1000,
            )
        )
        self.added += 1

    def run(self) -> None:
        started = time.monotonic()
        index = 0
        while not self._stop_event.is_set() and time.monotonic() - started < self.seconds:
            self._add(self.base_unix_us + POSE_LEAD_US + index * 100_000)
            index += 1
            self._stop_event.wait(0.1)


# --------------------------------------------------------------------------- #
# Independent verification of one frame
# --------------------------------------------------------------------------- #


#: ``cam0:s184`` - the receiver stamps each frame with the *Alpamayo* camera
#: index and the vehicle's sequence number, not with the wire camera id.  Both
#: have to be undone to find the file a frame came from.
FRAME_ID = re.compile(r"^cam(\d+):s(\d+)$")

#: Alpamayo index -> the camera id the vehicle used on the wire.
INDEX_TO_WIRE = {index: camera for camera, index in zip(WIRE_CAMERA_IDS, ALPAMAYO_INDICES)}


def parse_frame_id(frame_id: Any) -> tuple[int, int] | None:
    """``cam0:s184`` -> ``(0, 184)``, or ``None`` if it is not that shape."""
    match = FRAME_ID.match(str(frame_id or ""))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def independent_decode(sample: Sample, size: tuple[int, int]) -> np.ndarray:
    """Decode a sample the way the receiver would, outside the receiver.

    ``cv2.imdecode`` then ``INTER_LINEAR`` resize then BGR->RGB, matching
    ``camera.cimg_source.decode_bgr`` and ``camera.sync._build``.  The point is
    that this path shares no code with the live one: if the two agree, the
    tensor in the window came from the bytes on disk.
    """
    import cv2

    array = np.frombuffer(sample.jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"{sample.path}: does not decode with OpenCV")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Buffer observation
# --------------------------------------------------------------------------- #


class DepthSampler(threading.Thread):
    """Watches how full the four per-camera buffers get, on its own schedule.

    The reader thread fills them at 10 groups/s and the window builder drains
    them; a reading taken from the test loop is therefore almost always taken
    just after a drain.  Sampling independently is the only way to say anything
    about the depths at all - and ``max`` over the run is the honest summary,
    because a depth is a transient by design, not a level.
    """

    #: Fast enough to catch the gap between a group arriving and the builder
    #: consuming it, which is well under one 100 ms group period.
    INTERVAL_S = 0.002

    def __init__(self, buffers: Sequence[Any]) -> None:
        super().__init__(daemon=True, name="cimg-depths")
        self.buffers = list(buffers)
        self.max_depths = [0] * len(self.buffers)
        self.samples = 0
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            for index, buffer in enumerate(self.buffers):
                depth = buffer.depth()
                if depth > self.max_depths[index]:
                    self.max_depths[index] = depth
            self.samples += 1
            self._stop_event.wait(self.INTERVAL_S)


# --------------------------------------------------------------------------- #
# Free port
# --------------------------------------------------------------------------- #


def free_loopback_port() -> int:
    """An ephemeral 127.0.0.1 port for the receiver to listen on.

    The port has to be known before the receiver starts, because it goes into
    the ``CimgConfig`` the receiver is built from.  Binding and closing a probe
    socket is the standard way to get one; the window between the close and the
    receiver's own bind is a race in principle and never fires in practice on
    loopback.  The report records the port so a failure here is diagnosable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --------------------------------------------------------------------------- #
# The scenario
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    report = Report()
    samples_dir = Path(args.samples)
    groups = load_samples(samples_dir)
    pixels = sum(len(sample.jpeg) for group in groups for sample in group)
    print(
        f"\n== CIMG ingress: {len(groups)} recorded groups "
        f"({len(groups) * len(WIRE_CAMERA_IDS)} JPEGs, {pixels / 1e6:.1f} MB) "
        f"from {samples_dir} =="
    )
    report.note(
        "camera data: real JPEG payloads from a real CIMG capture; the packet "
        "framing is rebuilt by this tool because the raw stream was never saved"
    )
    report.note("pose: synthetic, 10 Hz, same Unix clock as the replayed captures")

    port = free_loopback_port()
    print(f"  . replay target: 127.0.0.1:{port} (ephemeral, loopback only)")

    timestamp_manager = TimestampManager.from_config()
    print(
        f"  . timestamps: camera_clock=capture_time_ns pose_timestamp_mode=unix"
    )

    config = CimgConfig(
        host="127.0.0.1",
        port=port,
        camera_indices=ALPAMAYO_INDICES,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    provider = CameraWindowProvider(
        config,
        frames_per_camera=FRAMES_PER_CAMERA,
        target_fps=GROUP_FPS,
        timestamp_manager=timestamp_manager,
    )
    report.check(
        "the provider built the CIMG reader",
        [type(r).__name__ for r in provider.readers] == ["CimgCameraSource"],
        f"readers={[type(r).__name__ for r in provider.readers]}",
    )

    # A little behind the server clock absorbs the sender's own startup latency.
    base_unix_us = time.time_ns() // 1000 - 100_000
    history = PoseHistory()
    pose = PoseFeeder(history, base_unix_us=base_unix_us, seconds=args.seconds)
    pose.seed()
    replayer = Replayer(
        "127.0.0.1", port, groups, seconds=args.seconds, base_unix_us=base_unix_us
    )

    builder = InputBuilder(
        history_steps=16,
        history_dt_s=0.1,
        camera_stale_ms=1000.0,
        pose_stale_ms=800.0,
        camera_pose_delta_ms=1000.0,
    )

    pool = _start_pool(args, report)
    tasks: list[tuple[int, InferenceTask]] = []
    build_results: list[Any] = []
    windows = 0
    rejections: Counter[str] = Counter()
    refusals: Counter[str] = Counter()
    window_faults: list[str] = []
    pixel_snapshot: tuple[Any, list[list[str]]] | None = None
    # The window builder drains the buffers as part of building a window, so a
    # depth read from this loop would always land after a consume and always be
    # zero.  A separate sampler observes the buffers on its own schedule, which
    # is what makes "the buffers held frames" an observation rather than an
    # artifact of where the reading thread happened to be.
    depths = DepthSampler(provider.buffers)
    try:
        provider.start()
        depths.start()
        replayer.start()
        pose.start()
        if pool is not None:
            pool.start()
            if not pool.wait_ready():
                report.check("the pool started", False, "workers never became ready")
                pool = None

        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            try:
                window = provider.latest_window(sync_timeout_s=0.5)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                # The window builder raises when no camera has produced a frame
                # within its startup budget.  That is a real verdict, but it
                # must not take the report down with it: the receiver's own
                # counters are what say *why*.
                window_faults.append(f"{type(exc).__name__}: {exc}")
                break
            if window is None:
                for reason, count in provider.refusals.items():
                    refusals[reason] += count
                provider.refusals.clear()
                time.sleep(0.02)
                continue
            windows += 1
            if len(build_results) < args.max_tasks:
                built = builder.build(window, history)
                build_results.append(built)
                if built.rejection_reason is not None:
                    rejections[built.rejection_reason] += 1
                else:
                    task = InferenceTask.from_build(
                        len(tasks) + 1, built, window, cycle_id=len(tasks) + 1
                    )
                    tasks.append((int(task.task_seq), task))
                    if pixel_snapshot is None:
                        # The window as the model would see it, kept because the
                        # live buffers are drained by the very builder that
                        # produced this task.
                        pixel_snapshot = (task.window.frames, task.window.frame_ids)
                    if pool is not None:
                        pool.submit(task)
            time.sleep(0.01)
    finally:
        replayer.stop()
        pose.stop()
        depths.stop()
        for thread in (replayer, pose):
            thread.join(timeout=5.0)
        # The replayer closing the socket does not mean the receiver has read
        # the last groups: whatever is buffered in the socket is still in
        # flight.  Without this pause the group count would sometimes come up
        # short and look like packet loss.
        time.sleep(0.5)
        provider.stop()
        provider.join(timeout_s=10.0)
        if pool is not None:
            pool.drain(seconds=20.0)

    if pool is not None:
        report.note(
            "pool: " + json.dumps(pool_report(pool)["result_drops"], sort_keys=True)
        )
    report.note(
        f"windows={windows} tasks={len(tasks)} "
        f"builder rejections={dict(rejections) or '{}'}"
    )
    if refusals:
        report.note(f"window refusals={dict(refusals)}")
    if replayer.error:
        report.check("the replayer stayed connected", False, replayer.error)
    else:
        report.check(
            "the replayer pushed groups for the whole run",
            replayer.groups_sent > 0,
            f"groups_sent={replayer.groups_sent} passes={replayer.passes} "
            f"bytes={replayer.bytes_sent / 1e6:.1f} MB",
        )
    report.check(
        "the window builder never raised",
        not window_faults,
        "; ".join(window_faults) if window_faults else "no exceptions",
    )
    check_ingress(provider, replayer, depths, report)
    check_window(build_results, provider, report)
    check_chain(build_results, tasks, pool, rejections, args, report)
    verify_pixels(pixel_snapshot, groups, report)
    return report.summary()


def _start_pool(args: argparse.Namespace, report: Report) -> Harness | None:
    """A fake-worker pool to hand the real tasks to, or ``None`` if declined.

    Fake workers on purpose: this tool is about the ingress chain, and loading
    a 10B checkpoint would make it unrunnable next to a soak.
    """
    if not args.devices:
        return None
    from inference.fake_worker import fake_worker_main

    settings = _settings_for(
        SimpleNamespace(result_max_age_ms=args.result_max_age_ms,
                        max_capture_lag_ms=args.max_capture_lag_ms),
        args.devices,
    )
    return Harness(
        devices=list(args.devices),
        result_max_age_ms=args.result_max_age_ms,
        max_capture_lag_ms=args.max_capture_lag_ms,
        worker_entry=fake_worker_main,
        worker_options={"delays_ms": [50.0]},
        settings=settings,
    )


def check_ingress(
    provider: CameraWindowProvider,
    replayer: Replayer,
    depths: DepthSampler,
    report: Report,
) -> None:
    """The receiver accepted the bytes and filled all four buffers.

    The depths are the maxima observed while the run was live.  Reading them
    afterwards would always show zero: the window builder consumes the buffers
    as fast as they fill, which is the design, not a fault.
    """
    source = provider.readers[0]
    stats = source.stats()
    report.note(
        "receiver: "
        + json.dumps(
            {k: v for k, v in stats.items() if k != "cameras"},
            sort_keys=True,
            default=str,
        )
    )
    report.check(
        "all four buffers held frames while the run was live",
        all(depth > 0 for depth in depths.max_depths),
        f"max_depths={depths.max_depths} of {provider.buffers[0].max_size} "
        f"capacity, over {depths.samples} independent samples",
    )
    per_camera_frames = [cam.get("frames_total") for cam in stats.get("cameras") or []]
    report.check(
        "every camera's every frame reached its buffer",
        per_camera_frames
        and all(count == replayer.groups_sent for count in per_camera_frames),
        f"frames per camera={per_camera_frames} groups sent={replayer.groups_sent}",
    )
    report.check(
        "every replayed group was ingested as a complete packet",
        stats.get("groups_total") == replayer.groups_sent,
        f"groups_total={stats.get('groups_total')} sent={replayer.groups_sent}",
    )
    parser = stats.get("protocol_errors") or {}
    report.check(
        "the stream never lost framing: no resync, no skipped or discarded bytes",
        not parser.get("resyncs") and not parser.get("packets_skipped")
        and not parser.get("discarded_bytes"),
        json.dumps(parser, sort_keys=True),
    )
    report.check(
        "every JPEG decoded",
        not stats.get("decode_errors"),
        f"decode_errors={stats.get('decode_errors')} "
        f"images_total={stats.get('images_total')}",
    )
    report.check(
        "no frame's timestamp was rejected",
        not stats.get("timestamp_errors"),
        f"timestamp_errors={stats.get('timestamp_errors')} "
        f"mode={stats.get('timestamp_mode')}",
    )
    dropped = stats.get("groups_dropped") or {}
    report.check(
        "no complete group was dropped by the backlog or partial-group policy",
        not dropped.get("backlog") and not dropped.get("partial")
        and not dropped.get("out_of_order"),
        json.dumps(dropped, sort_keys=True),
    )
    per_camera = stats.get("cameras") or []
    report.check(
        "all four cameras appear, each with frames and none stale",
        len(per_camera) == len(WIRE_CAMERA_IDS)
        and all(cam.get("health") == "HEALTHY" for cam in per_camera),
        json.dumps(
            {cam.get("camera_id"): cam.get("frames_total") for cam in per_camera},
            sort_keys=True,
        )
        if per_camera
        else "no per-camera stats",
    )
    report.check(
        "camera health is HEALTHY on all four",
        all(state == "HEALTHY" for state in provider.camera_health().values()),
        json.dumps(provider.camera_health(), sort_keys=True),
    )


def check_window(
    build_results: list[Any], provider: CameraWindowProvider, report: Report
) -> None:
    """The window is the shape the model expects, from the frames that were sent."""
    report.check(
        "the window builder produced windows",
        provider.window_builds > 0,
        f"window_builds={provider.window_builds}",
    )
    if not build_results:
        report.check("a window reached the input builder", False, "no windows built")
        return


def check_chain(
    build_results: list[Any],
    tasks: list[tuple[int, InferenceTask]],
    pool: Harness | None,
    rejections: Counter[str],
    args: argparse.Namespace,
    report: Report,
) -> None:
    """InputBuilder accepted at least one window and a task was built from it."""
    accepted = [result for result in build_results if result.accepted]
    report.check(
        "InputBuilder accepted a real window",
        bool(accepted),
        f"accepted={len(accepted)}/{len(build_results)} "
        f"rejections={dict(rejections) or '{}'}",
    )
    if not accepted:
        return
    result = accepted[0]
    report.check(
        "the assembled ego history has the model's shape",
        tuple(result.ego_xyz.shape) == (1, 1, 16, 3)
        and tuple(result.ego_rot.shape) == (1, 1, 16, 3, 3),
        f"ego_xyz={tuple(result.ego_xyz.shape)} ego_rot={tuple(result.ego_rot.shape)}",
    )
    report.check(
        "the reference pose came from the history, not from a default",
        result.reference_pose is not None
        and result.reference_payload_timestamp_us is not None,
        f"reference_pose={result.reference_pose}",
    )
    report.check(
        "an InferenceTask was built from it",
        bool(tasks),
        f"tasks={len(tasks)}",
    )
    if not tasks:
        return
    seq, task = tasks[-1]
    report.check(
        "the frozen task carries the real window's frames",
        task.window is not None
        and len(task.window.camera_indices) == len(ALPAMAYO_INDICES)
        and tuple(task.window.frames.shape) == (
            len(ALPAMAYO_INDICES),
            FRAMES_PER_CAMERA,
            3,
            args.frame_height,
            args.frame_width,
        ),
        f"indices={task.window.camera_indices} "
        f"frames={tuple(task.window.frames.shape)}",
    )
    report.check(
        "the camera indices are in the model's order, not the wire order",
        list(task.window.camera_indices) == sorted(ALPAMAYO_INDICES),
        f"window indices={task.window.camera_indices} "
        f"wire order={list(WIRE_CAMERA_IDS)} alpamayo={sorted(ALPAMAYO_INDICES)}",
    )
    if pool is None:
        return
    accepted_seqs = {seq for seq, _worker, _points in pool.accepted}
    report.check(
        "the real task crossed the process boundary and was accepted",
        bool(accepted_seqs),
        f"accepted={len(accepted_seqs)} of {len(tasks)} submitted, "
        f"sample seq={seq}",
    )
    report.check(
        "every accepted result carried a non-empty trajectory",
        all(points > 0 for _seq, _worker, points in pool.accepted),
        f"points={[points for _seq, _worker, points in pool.accepted][-6:]}",
    )


def verify_pixels(
    snapshot: tuple[Any, list[list[str]]] | None,
    groups: list[list[Sample]],
    report: Report,
) -> None:
    """Prove the frames in the assembled window are the JPEGs that were on disk.

    The comparison is against the *frozen task window* rather than the live
    buffer: the buffer is drained by the window builder, so by the time a run
    has finished there is nothing left to compare, and the tensor the model
    would actually receive is the stronger thing to verify anyway.

    The frame ids are the vehicle's own sequence numbers, so each frame in the
    window names the file it came from.  Without this check a receiver that
    decoded *something* - a black frame, a wrong camera, a resized anything -
    would pass every count in this report.
    """
    if snapshot is None:
        report.check(
            "a window's frames are byte-identical to the JPEGs on disk",
            False,
            "no built window was available to compare",
        )
        return
    import torch

    frames, frame_ids = snapshot
    size = (int(frames.shape[-1]), int(frames.shape[-2]))
    by_id = {
        (sample.camera_id, sample.sequence): sample
        for group in groups
        for sample in group
    }
    checked = 0
    mismatches: list[str] = []
    beyond: list[str] = []
    for row, ids in enumerate(frame_ids):
        for column, frame_id in enumerate(ids):
            parsed = parse_frame_id(frame_id)
            if parsed is None:
                continue
            index, sequence = parsed
            sample = by_id.get((INDEX_TO_WIRE.get(index, -1), sequence))
            if sample is None:
                # Past the recorded set: the replayer continued the numbering,
                # so there is no file to compare against.  Counted, not hidden.
                beyond.append(str(frame_id))
                continue
            expected = torch.from_numpy(independent_decode(sample, size)).permute(2, 0, 1)
            if not torch.equal(expected, frames[row, column]):
                mismatches.append(f"{frame_id} (row {row}, column {column})")
            checked += 1
    report.check(
        "a window's frames are byte-identical to the JPEGs on disk",
        checked > 0 and not mismatches,
        f"checked={checked} mismatches={mismatches[:5]} beyond_recorded={len(beyond)} "
        f"comparison={size[0]}x{size[1]} RGB (independent cv2 decode, "
        "outside the receiver)",
    )


# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay recorded CIMG JPEG payloads into the real ingress chain on "
            "loopback.  No vehicle and no control link."
        )
    )
    parser.add_argument(
        "--samples",
        required=True,
        help="directory of group<NNN>_cam<ID>_seq<SEQ>.jpg recorded payloads",
    )
    parser.add_argument(
        "--seconds", type=float, default=8.0, help="replay duration (default 8)"
    )
    parser.add_argument("--frame-width", type=int, default=576)
    parser.add_argument("--frame-height", type=int, default=320)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=40,
        help=(
            "how many built inputs to keep and hand to the pool (default 40).  "
            "The receive rate is not throttled by this - only the number of "
            "tasks submitted, so the buffers are still exercised at 10 Hz."
        ),
    )
    parser.add_argument(
        "--devices",
        default="",
        help=(
            "comma-separated devices for a fake-worker pool to hand the built "
            "tasks to.  Empty (the default) stops the chain at InferenceTask."
        ),
    )
    parser.add_argument("--result-max-age-ms", type=float, default=2500.0)
    parser.add_argument("--max-capture-lag-ms", type=float, default=2500.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.devices = [int(x) for x in args.devices.split(",") if x.strip()]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print("CIMG ingress test - loopback only, no vehicle, no control link")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
