"""The three frequencies, and why one number would not have done.

The point of these tests is that the three rates are *separable*.  A pool whose
model is running at 4 Hz while accepted results trickle out at 1 Hz is either
saturated or discarding, and the difference is exactly which counter moved.  So
each test here drives one of the three in isolation and asserts the other two
did not move.

Time is injected, so nothing sleeps.
"""

from __future__ import annotations

import unittest

from inference.metrics import (
    LATENCY_SAMPLES,
    LONG_WINDOW_S,
    SHORT_WINDOW_S,
    CounterSet,
    Frequency,
    InferenceMetrics,
    LatencyStats,
    UdpCounters,
)


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class FrequencyTests(unittest.TestCase):
    def test_the_windows_are_the_documented_ones(self) -> None:
        self.assertEqual(SHORT_WINDOW_S, 10.0)
        self.assertEqual(LONG_WINDOW_S, 60.0)

    def test_counts_and_rates_over_both_windows(self) -> None:
        clock = Clock()
        frequency = Frequency(clock=clock)
        for _ in range(10):
            frequency.add()
        clock.advance(5.0)
        for _ in range(10):
            frequency.add()
        self.assertEqual(frequency.total, 20)
        # 20 events in the 5 s the counter has existed: 4 Hz, in both windows.
        self.assertAlmostEqual(frequency.rate(SHORT_WINDOW_S), 4.0)
        self.assertAlmostEqual(frequency.rate(LONG_WINDOW_S), 4.0)
        self.assertAlmostEqual(frequency.average_hz, 4.0)

    def test_the_long_window_forgets_what_the_short_one_already_dropped(self) -> None:
        clock = Clock()
        frequency = Frequency(clock=clock)
        for _ in range(10):
            frequency.add()
        clock.advance(100.0)
        for _ in range(10):
            frequency.add()
        self.assertEqual(frequency.total, 20)
        self.assertAlmostEqual(frequency.rate(SHORT_WINDOW_S), 1.0)
        self.assertAlmostEqual(frequency.rate(LONG_WINDOW_S), 10 / 60.0)
        self.assertAlmostEqual(frequency.average_hz, 0.2)

    def test_a_young_counter_reports_its_real_rate_not_a_fraction_of_it(self) -> None:
        """A 2 s old counter divides by 2, not by the window.

        Otherwise the first minute of every process looks like a failure, which
        is exactly the minute an operator is watching it.
        """
        clock = Clock()
        frequency = Frequency(clock=clock)
        for _ in range(5):
            frequency.add()
        clock.advance(2.0)
        self.assertAlmostEqual(frequency.rate(LONG_WINDOW_S), 2.5)

    def test_a_counter_that_has_never_fired_reports_zero(self) -> None:
        frequency = Frequency(clock=Clock())
        self.assertEqual(frequency.rate(SHORT_WINDOW_S), 0.0)
        self.assertEqual(frequency.average_hz, 0.0)
        self.assertEqual(frequency.total, 0)

    def test_the_retained_timestamps_are_bounded_by_the_long_window(self) -> None:
        clock = Clock()
        frequency = Frequency(clock=clock)
        for _ in range(100):
            frequency.add()
            clock.advance(1.0)
        # 100 s of history, but only the last 60 s is ever asked about.
        self.assertLessEqual(len(frequency._times), 62)

    def test_a_batch_count_is_recorded_as_individual_events(self) -> None:
        clock = Clock()
        frequency = Frequency(clock=clock)
        frequency.add(count=10)
        clock.advance(5.0)
        self.assertEqual(frequency.total, 10)
        self.assertAlmostEqual(frequency.rate(SHORT_WINDOW_S), 2.0)

    def test_the_snapshot_carries_all_three_numbers(self) -> None:
        clock = Clock()
        frequency = Frequency(clock=clock)
        frequency.add()
        clock.advance(1.0)
        snapshot = frequency.snapshot()
        self.assertEqual(snapshot["total"], 1)
        self.assertAlmostEqual(snapshot["hz_10s"], 1.0)
        self.assertAlmostEqual(snapshot["hz_60s"], 1.0)
        self.assertAlmostEqual(snapshot["hz_avg"], 1.0)


class LatencyStatsTests(unittest.TestCase):
    def test_an_empty_stats_reports_nothing_rather_than_zero(self) -> None:
        snapshot = LatencyStats().snapshot()
        self.assertEqual(snapshot["count"], 0)
        self.assertIsNone(snapshot["avg_ms"])
        self.assertIsNone(snapshot["p95_ms"])

    def test_the_percentiles_interpolate(self) -> None:
        stats = LatencyStats()
        for value in range(1, 101):
            stats.add(value)
        self.assertEqual(stats.percentile_ms(0.50), 50.5)
        # 0.95 * 99 = 94.05, so 95% of the way from the 95th value to the 96th.
        self.assertAlmostEqual(stats.percentile_ms(0.95), 95.05)

    def test_a_single_sample_is_its_own_percentile(self) -> None:
        stats = LatencyStats()
        stats.add(7.0)
        self.assertEqual(stats.percentile_ms(0.50), 7.0)
        self.assertEqual(stats.percentile_ms(0.95), 7.0)

    def test_the_summary_tracks_last_min_max_and_average(self) -> None:
        stats = LatencyStats()
        for value in (10.0, 30.0, 20.0):
            stats.add(value)
        snapshot = stats.snapshot()
        self.assertEqual(snapshot["count"], 3)
        self.assertEqual(snapshot["last_ms"], 20.0)
        self.assertEqual(snapshot["min_ms"], 10.0)
        self.assertEqual(snapshot["max_ms"], 30.0)
        self.assertEqual(snapshot["avg_ms"], 20.0)

    def test_the_sample_is_bounded(self) -> None:
        """Hours of running must not accumulate an unbounded list of floats."""
        stats = LatencyStats()
        for value in range(LATENCY_SAMPLES * 3):
            stats.add(float(value))
        self.assertEqual(len(stats._values), LATENCY_SAMPLES)
        # The totals still cover every sample, not just the retained ones.
        self.assertEqual(stats.count, LATENCY_SAMPLES * 3)


class UdpCounterTests(unittest.TestCase):
    def test_a_release_frame_is_a_send_but_not_a_trajectory(self) -> None:
        """The number this module exists to stop people quoting."""
        counters = UdpCounters()
        for _ in range(3):
            counters.add_sent(waypoint_count=0, control_flag=0, seq=1)
        for _ in range(7):
            counters.add_sent(waypoint_count=16, control_flag=1, seq=2)
        snapshot = counters.snapshot()
        self.assertEqual(snapshot["sent_total"], 10)
        self.assertEqual(snapshot["sent_with_waypoints"], 7)
        self.assertEqual(snapshot["sent_release_frames"], 3)

    def test_the_last_send_is_described(self) -> None:
        counters = UdpCounters()
        counters.add_sent(waypoint_count=4, control_flag=1, seq=9)
        snapshot = counters.snapshot()
        self.assertEqual(snapshot["last_waypoint_count"], 4)
        self.assertEqual(snapshot["last_control_flag"], 1)
        self.assertEqual(snapshot["last_seq"], 9)

    def test_a_failed_send_is_not_a_send(self) -> None:
        counters = UdpCounters()
        counters.add_failure()
        snapshot = counters.snapshot()
        self.assertEqual(snapshot["sent_total"], 0)
        self.assertEqual(snapshot["send_failures"], 1)

    def test_a_zero_waypoint_frame_is_counted_as_a_release(self) -> None:
        counters = UdpCounters()
        counters.add_sent(waypoint_count=0, control_flag=1, seq=1)
        self.assertEqual(counters.snapshot()["sent_release_frames"], 1)


class CounterSetTests(unittest.TestCase):
    def test_it_counts_by_name_and_defaults_to_zero(self) -> None:
        counters = CounterSet()
        counters.bump("expired")
        counters.bump("expired")
        counters.bump("other", 3)
        self.assertEqual(counters.get("expired"), 2)
        self.assertEqual(counters["other"], 3)
        self.assertEqual(counters.get("never"), 0)

    def test_a_non_positive_bump_is_ignored(self) -> None:
        counters = CounterSet()
        counters.bump("x", 0)
        counters.bump("x", -5)
        self.assertEqual(counters.get("x"), 0)

    def test_the_snapshot_is_a_copy(self) -> None:
        counters = CounterSet()
        counters.bump("x")
        snapshot = counters.snapshot()
        counters.bump("x")
        self.assertEqual(snapshot["x"], 1)


class ThreeFrequencyTests(unittest.TestCase):
    """The three rates are counted separately, and none of them is a proxy."""

    def setUp(self) -> None:
        self.clock = Clock()
        self.metrics = InferenceMetrics(clock=self.clock)

    def test_a_raw_inference_is_counted_without_an_accepted_result(self) -> None:
        """A result later dropped still cost real GPU time."""
        self.metrics.note_raw_inference(infer_ms=1500.0)
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["raw_inference"]["total"], 1)
        self.assertEqual(snapshot["accepted_result"]["total"], 0)
        self.assertEqual(snapshot["udp_output"]["total"], 0)

    def test_an_accepted_result_is_counted_without_udp_output(self) -> None:
        self.metrics.note_accepted(end_to_end_ms=1200.0)
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["accepted_result"]["total"], 1)
        self.assertEqual(snapshot["raw_inference"]["total"], 0)
        self.assertEqual(snapshot["udp_output"]["total"], 0)

    def test_udp_output_is_counted_without_an_accepted_result(self) -> None:
        """Release frames go out on cycles that produced no trajectory."""
        self.metrics.note_udp_sent(waypoint_count=0, control_flag=0, seq=1)
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["udp_output"]["total"], 1)
        self.assertEqual(snapshot["accepted_result"]["total"], 0)

    def test_the_three_rates_can_disagree(self) -> None:
        """The whole reason there are three: 4 Hz in, 1 Hz out, is diagnosable."""
        for _ in range(4):
            self.metrics.note_raw_inference(infer_ms=1200.0)
        self.metrics.note_accepted(end_to_end_ms=1300.0)
        self.metrics.note_udp_sent(waypoint_count=8, control_flag=1, seq=1)
        self.clock.advance(1.0)
        snapshot = self.metrics.snapshot()
        self.assertAlmostEqual(snapshot["raw_inference"]["hz_10s"], 4.0)
        self.assertAlmostEqual(snapshot["accepted_result"]["hz_10s"], 1.0)
        self.assertAlmostEqual(snapshot["udp_output"]["hz_10s"], 1.0)

    def test_a_udp_send_records_the_packet_breakdown(self) -> None:
        self.metrics.note_udp_sent(waypoint_count=16, control_flag=1, seq=4)
        self.metrics.note_udp_sent(waypoint_count=0, control_flag=0, seq=5)
        packets = self.metrics.snapshot()["udp_packets"]
        self.assertEqual(packets["sent_total"], 2)
        self.assertEqual(packets["sent_with_waypoints"], 1)
        self.assertEqual(packets["sent_release_frames"], 1)

    def test_a_failed_send_is_counted_apart_from_the_rate(self) -> None:
        self.metrics.note_udp_failure()
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["udp_output"]["total"], 0)
        self.assertEqual(snapshot["udp_packets"]["send_failures"], 1)

    def test_the_latencies_are_kept_separate(self) -> None:
        self.metrics.queue_wait_ms.add(5.0)
        self.metrics.infer_ms.add(1400.0)
        self.metrics.validate_ms.add(0.4)
        self.metrics.end_to_end_ms.add(1500.0)
        self.metrics.task_to_send_ms.add(1520.0)
        latency = self.metrics.snapshot()["latency_ms"]
        self.assertEqual(latency["queue_wait"]["last_ms"], 5.0)
        self.assertEqual(latency["infer"]["last_ms"], 1400.0)
        self.assertEqual(latency["validate"]["last_ms"], 0.4)
        self.assertEqual(latency["end_to_end"]["last_ms"], 1500.0)
        self.assertEqual(latency["task_to_send"]["last_ms"], 1520.0)

    def test_the_configured_window_is_honoured(self) -> None:
        metrics = InferenceMetrics(clock=self.clock, long_s=60.0)
        self.assertEqual(metrics.window_s, 60.0)
        self.assertEqual(metrics.raw_inference.long_s, 60.0)


if __name__ == "__main__":
    unittest.main()
