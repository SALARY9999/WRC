"""The counters the inference pool publishes, and how they are defined.

Three frequencies, deliberately separate
    ``raw_inference_hz``    the model finished a call.  Counted the moment a
                            worker's result arrives, *before* the result manager
                            has looked at it - so it includes results that are
                            later dropped as out of order, expired or
                            content-invalid.  It excludes worker errors, CUDA
                            OOMs and pending tasks that were replaced before
                            they ever started, because none of those is an
                            inference.
    ``accepted_result_hz``  the result passed every result-manager check and was
                            allowed into the trajectory / takeover / UDP chain.
    ``udp_output_hz``       ``WaypointSender.send()`` returned.

    The three exist because the single ``inference_count`` they replace could
    not answer the only question that matters when the pool is under load: the
    model may be running at 4 Hz white accepted results trickle out at 1 Hz, and
    one number cannot show that.  ``udp_output_hz`` alone is *also* not enough -
    a zero-waypoint release frame is a legitimate send (see ``pipeline``), so a
    healthy-looking UDP rate can be made of packets that carry no trajectory.
    The packet breakdown below is what separates the two.

Windows
    Each frequency reports a 10 s sliding window, a 60 s sliding window and the
    average since the counter started.  A sliding-window rate divides by the
    window, clipped to the time the counter has actually existed - otherwise a
    process five seconds old would report a third of its real rate and look
    broken during exactly the startup window an operator is watching.

Everything here is a leaf: stdlib only, no torch, no CUDA, no pipeline.  The
clock is injectable so the rates can be tested without sleeping.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

#: The two sliding windows every frequency reports.
SHORT_WINDOW_S = 10.0
LONG_WINDOW_S = 60.0

#: How many per-call samples a :class:`LatencyStats` keeps for its percentiles.
#: Bounded on purpose: the pool runs for hours and an unbounded list of floats
#: is a slow leak.  512 samples is ~8 minutes of history at 1 Hz/worker.
LATENCY_SAMPLES = 512


class Frequency:
    """One event rate: total count, sliding windows and since-start average."""

    def __init__(
        self,
        *,
        short_s: float = SHORT_WINDOW_S,
        long_s: float = LONG_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.short_s = float(short_s)
        self.long_s = float(long_s)
        self._clock = clock
        self._times: deque[float] = deque()
        self.total = 0
        self._started_at: float | None = None

    def add(self, count: int = 1, at: float | None = None) -> None:
        """Record ``count`` events now (or at ``at``, for tests)."""
        if count <= 0:
            return
        now = self._clock() if at is None else float(at)
        if self._started_at is None:
            self._started_at = now
        self.total += int(count)
        for _ in range(int(count)):
            self._times.append(now)
        self._prune(now)

    def _prune(self, now: float) -> None:
        horizon = now - self.long_s
        while self._times and self._times[0] < horizon:
            self._times.popleft()

    def rate(self, window_s: float) -> float:
        """Events per second over the last ``window_s`` seconds.

        Divided by the window clipped to the counter's own age: a counter that
        has existed for 2 s reports its true rate over those 2 s rather than a
        fifth of it.
        """
        now = self._clock()
        self._prune(now)
        if self._started_at is None:
            return 0.0
        elapsed = min(float(window_s), max(0.0, now - self._started_at))
        if elapsed <= 0.0:
            return 0.0
        horizon = now - float(window_s)
        recent = sum(1 for t in self._times if t >= horizon)
        return recent / elapsed

    @property
    def average_hz(self) -> float:
        if self._started_at is None:
            return 0.0
        elapsed = self._clock() - self._started_at
        if elapsed <= 0.0:
            return 0.0
        return self.total / elapsed

    def snapshot(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "hz_10s": round(self.rate(self.short_s), 4),
            "hz_60s": round(self.rate(self.long_s), 4),
            "hz_avg": round(self.average_hz, 4),
        }


class LatencyStats:
    """A bounded sample of milliseconds, with the summary an operator reads."""

    def __init__(self, *, samples: int = LATENCY_SAMPLES) -> None:
        self._values: deque[float] = deque(maxlen=int(samples))
        self.count = 0
        self.total_ms = 0.0
        self.last_ms: float | None = None
        self.max_ms: float | None = None
        self.min_ms: float | None = None

    def add(self, value_ms: float) -> None:
        value = float(value_ms)
        self._values.append(value)
        self.count += 1
        self.total_ms += value
        self.last_ms = value
        self.max_ms = value if self.max_ms is None else max(self.max_ms, value)
        self.min_ms = value if self.min_ms is None else min(self.min_ms, value)

    @property
    def average_ms(self) -> float | None:
        return None if self.count == 0 else self.total_ms / self.count

    def percentile_ms(self, fraction: float) -> float | None:
        """Linear-interpolated percentile over the retained sample."""
        if not self._values:
            return None
        ordered = sorted(self._values)
        if len(ordered) == 1:
            return ordered[0]
        position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "last_ms": _round(self.last_ms),
            "avg_ms": _round(self.average_ms),
            "p50_ms": _round(self.percentile_ms(0.50)),
            "p95_ms": _round(self.percentile_ms(0.95)),
            "min_ms": _round(self.min_ms),
            "max_ms": _round(self.max_ms),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


@dataclass
class UdpCounters:
    """What actually went out on the waypoint socket.

    ``sent_total`` alone is the number this module exists to stop people
    quoting: a release frame (``control=0``, ``waypoint_count=0``) is a send
    and carries no trajectory at all.
    """

    sent_total: int = 0
    sent_with_waypoints: int = 0
    sent_release_frames: int = 0
    send_failures: int = 0
    last_waypoint_count: int = 0
    last_control_flag: int | None = None
    last_seq: int | None = None

    def add_sent(self, *, waypoint_count: int, control_flag: int, seq: int) -> None:
        self.sent_total += 1
        if waypoint_count > 0:
            self.sent_with_waypoints += 1
        else:
            self.sent_release_frames += 1
        self.last_waypoint_count = int(waypoint_count)
        self.last_control_flag = int(control_flag)
        self.last_seq = int(seq)

    def add_failure(self) -> None:
        self.send_failures += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "sent_total": self.sent_total,
            "sent_with_waypoints": self.sent_with_waypoints,
            "sent_release_frames": self.sent_release_frames,
            "send_failures": self.send_failures,
            "last_waypoint_count": self.last_waypoint_count,
            "last_control_flag": self.last_control_flag,
            "last_seq": self.last_seq,
        }


class CounterSet:
    """Named monotonic counters, snapshotted as one immutable dict.

    A dict rather than a field per counter because the set is read as a whole
    by ``/health`` and extended by whoever finds a new drop reason - adding a
    key must not mean editing a dataclass in three places.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def bump(self, name: str, count: int = 1) -> None:
        if count <= 0:
            return
        self._counts[name] = self._counts.get(name, 0) + int(count)

    def get(self, name: str) -> int:
        return self._counts.get(name, 0)

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)

    def __getitem__(self, name: str) -> int:
        return self.get(name)


class InferenceMetrics:
    """Everything the pool counts, in one object with one clock.

    Owned by the pipeline thread, which is the only thread that feeds it: the
    scheduler's ``tick()``, the result manager's verdicts and the UDP send all
    happen on that thread.  Workers never touch it - they report numbers, and
    the parent decides what they mean.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        long_s: float = LONG_WINDOW_S,
    ) -> None:
        self.clock = clock
        self.window_s = float(long_s)
        self.raw_inference = Frequency(clock=clock, long_s=long_s)
        self.accepted_result = Frequency(clock=clock, long_s=long_s)
        self.udp_output = Frequency(clock=clock, long_s=long_s)
        self.udp = UdpCounters()
        # Drop / fault reasons, each its own counter.  ``pending_replaced`` is
        # the latest-only policy's cost and is not an error; the rest are.
        self.drops = CounterSet()
        self.faults = CounterSet()
        #: Time a task waited between being frozen and being dispatched.  Grows
        #: with worker saturation, which is the signal "add a worker".
        self.queue_wait_ms = LatencyStats()
        #: Pure model time inside the worker.
        self.infer_ms = LatencyStats()
        #: Time spent validating one result in the pipeline thread.
        self.validate_ms = LatencyStats()
        #: Task frozen -> result accepted, on the monotonic clock.
        self.end_to_end_ms = LatencyStats()
        #: Task frozen -> datagram handed to the socket.
        self.task_to_send_ms = LatencyStats()

    # -- convenience writers ------------------------------------------------ #

    def note_raw_inference(self, *, infer_ms: float | None = None) -> None:
        self.raw_inference.add()
        if infer_ms is not None:
            self.infer_ms.add(infer_ms)

    def note_accepted(self, *, end_to_end_ms: float | None = None) -> None:
        self.accepted_result.add()
        if end_to_end_ms is not None:
            self.end_to_end_ms.add(end_to_end_ms)

    def note_udp_sent(
        self,
        *,
        waypoint_count: int,
        control_flag: int,
        seq: int,
        task_to_send_ms: float | None = None,
    ) -> None:
        self.udp_output.add()
        self.udp.add_sent(
            waypoint_count=waypoint_count,
            control_flag=control_flag,
            seq=seq,
        )
        if task_to_send_ms is not None:
            self.task_to_send_ms.add(task_to_send_ms)

    def note_udp_failure(self) -> None:
        self.udp.add_failure()

    def snapshot(self) -> dict[str, Any]:
        """The metrics half of the published pool view."""
        return {
            "raw_inference": self.raw_inference.snapshot(),
            "accepted_result": self.accepted_result.snapshot(),
            "udp_output": self.udp_output.snapshot(),
            "udp_packets": self.udp.snapshot(),
            "drops": self.drops.snapshot(),
            "faults": self.faults.snapshot(),
            "latency_ms": {
                "queue_wait": self.queue_wait_ms.snapshot(),
                "infer": self.infer_ms.snapshot(),
                "validate": self.validate_ms.snapshot(),
                "end_to_end": self.end_to_end_ms.snapshot(),
                "task_to_send": self.task_to_send_ms.snapshot(),
            },
        }


__all__ = [
    "CounterSet",
    "Frequency",
    "InferenceMetrics",
    "LATENCY_SAMPLES",
    "LONG_WINDOW_S",
    "LatencyStats",
    "SHORT_WINDOW_S",
    "UdpCounters",
]
