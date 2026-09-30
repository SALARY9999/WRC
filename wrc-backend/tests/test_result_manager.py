"""The result manager: every check, in the order it runs, counted once.

This is the last thing between a GPU's answer and the vehicle's steering, so
the tests here are mostly about *refusal*: for each way a result can be wrong,
one test that it is dropped and one that the drop is attributed to the right
counter.  "We are dropping results" is not actionable; "every result is 4 s
old" is, and that only works if each drop is counted exactly once.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from inference.result_manager import (
    CAPTURE_JUMP,
    CAPTURE_ROLLBACK,
    DROP_CAPTURE_ANOMALY,
    DROP_CAPTURE_LAG,
    DROP_DUPLICATE,
    DROP_EXPIRED,
    DROP_INVALID_POINTS,
    DROP_INVALID_SHAPE,
    DROP_NON_FINITE,
    DROP_OUT_OF_ORDER,
    DROP_TASK_SEQ_MISMATCH,
    DROP_UNKNOWN_TASK,
    DROP_WORKER_ERROR,
    ResultManager,
)
from inference.tasks import ERROR_CUDA_OOM, ERROR_WORKER_EXCEPTION, InferenceResult

from _fake_pool import make_task

#: Every drop reason the manager can emit, so a test can assert that exactly one
#: of them moved.
ALL_REASONS = (
    DROP_WORKER_ERROR,
    DROP_UNKNOWN_TASK,
    DROP_TASK_SEQ_MISMATCH,
    DROP_OUT_OF_ORDER,
    DROP_DUPLICATE,
    DROP_EXPIRED,
    DROP_CAPTURE_LAG,
    DROP_CAPTURE_ANOMALY,
    DROP_INVALID_SHAPE,
    DROP_INVALID_POINTS,
    DROP_NON_FINITE,
    ERROR_CUDA_OOM,
    ERROR_WORKER_EXCEPTION,
)


class _Trajectory:
    """The one attribute the manager reads off a model result."""

    def __init__(self, positions) -> None:
        self.positions = positions


#: Distinguishes "the test did not pass positions" from "the result carried no
#: positions at all" - the second is a case the manager has to reject.
_OMIT = object()


def good_positions(points: int = 16) -> np.ndarray:
    steps = np.arange(points, dtype=np.float64)
    return np.stack([steps, np.zeros(points), np.zeros(points)], axis=1)


#: Tasks default to one per worker, because that is the only arrangement the
#: scheduler can produce: a worker holds exactly one task, and no two workers
#: hold the same one.  Tests that need a mismatch say so explicitly.
def _worker_for(seq: int, worker_id: int | None) -> int:
    return int(seq) if worker_id is None else int(worker_id)


def submit(manager: ResultManager, seq: int, worker_id: int | None = None, **kwargs):
    """Note a task and put it in flight, as the scheduler does."""
    task = make_task(seq, **kwargs)
    manager.note_task(task)
    manager.note_dispatch(task, _worker_for(seq, worker_id))
    return task


def report(
    seq: int,
    *,
    worker_id: int | None = None,
    positions=_OMIT,
    ok: bool = True,
    error: str | None = None,
    error_kind: str = "",
) -> InferenceResult:
    return InferenceResult(
        task_seq=seq,
        worker_id=_worker_for(seq, worker_id),
        device=1,
        ok=ok,
        model_result=(
            None
            if not ok
            else _Trajectory(good_positions() if positions is _OMIT else positions)
        ),
        error=error,
        error_kind=error_kind,
    )


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ResultManager()

    def test_a_good_result_is_accepted_and_moves_the_watermark(self) -> None:
        submit(self.manager, 1)
        verdict = self.manager.validate(report(1))
        self.assertTrue(verdict.accepted)
        self.assertEqual(verdict.reason, "")
        self.assertEqual(self.manager.last_accepted_task_seq, 1)

    def test_the_watermark_starts_before_any_task(self) -> None:
        self.assertEqual(self.manager.last_accepted_task_seq, -1)

    def test_a_dropped_result_does_not_move_the_watermark(self) -> None:
        submit(self.manager, 1)
        self.manager.validate(report(1, ok=False, error="boom"))
        self.assertEqual(self.manager.last_accepted_task_seq, -1)

    def test_an_accepted_task_is_no_longer_in_flight(self) -> None:
        submit(self.manager, 1)
        self.manager.validate(report(1))
        self.assertEqual(self.manager.inflight_task_seqs, ())


class OrderingTests(unittest.TestCase):
    """The mandated rule: ``task_seq <= last_accepted`` is a drop, full stop."""

    def setUp(self) -> None:
        self.manager = ResultManager()

    def test_a_lower_sequence_is_dropped_however_new_its_capture_is(self) -> None:
        submit(self.manager, 1, capture_us=1_000_000)
        submit(self.manager, 2, capture_us=2_000_000)
        self.assertTrue(self.manager.validate(report(2)).accepted)
        # Task 1 carries an *older* capture stamp and a lower sequence: dropped.
        verdict = self.manager.validate(report(1))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_OUT_OF_ORDER)
        self.assertEqual(self.manager.last_accepted_task_seq, 2)

    def test_a_higher_sequence_is_accepted_however_old_its_capture_is(self) -> None:
        """Sequence decides, never the camera clock."""
        manager = ResultManager(max_capture_lag_ms=60_000.0)
        submit(manager, 1, capture_us=9_000_000)
        submit(manager, 2, capture_us=1_000_000)  # far *behind*, still newer
        self.assertTrue(manager.validate(report(1)).accepted)
        self.assertTrue(manager.validate(report(2)).accepted)
        self.assertEqual(manager.last_accepted_task_seq, 2)

    def test_a_result_equal_to_the_watermark_is_a_duplicate(self) -> None:
        submit(self.manager, 1)
        self.assertTrue(self.manager.validate(report(1)).accepted)
        verdict = self.manager.validate(report(1))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_DUPLICATE)

    def test_skipping_a_sequence_is_fine(self) -> None:
        """Sequences are allocated only for accepted inputs, so gaps are normal."""
        submit(self.manager, 7)
        self.assertTrue(self.manager.validate(report(7)).accepted)
        self.assertEqual(self.manager.last_accepted_task_seq, 7)


class IdentificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ResultManager()

    def test_a_result_from_a_worker_with_nothing_in_flight_is_unknown(self) -> None:
        verdict = self.manager.validate(report(3, worker_id=2))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_UNKNOWN_TASK)

    def test_a_result_naming_a_task_the_worker_does_not_hold_is_a_mismatch(self) -> None:
        submit(self.manager, 4, worker_id=1)
        verdict = self.manager.validate(report(5, worker_id=1))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_TASK_SEQ_MISMATCH)

    def test_the_task_a_mismatch_reported_is_left_in_flight(self) -> None:
        """Nothing is released on a mismatch: we do not know which side is wrong."""
        submit(self.manager, 4, worker_id=1)
        self.manager.validate(report(5, worker_id=1))
        self.assertEqual(self.manager.inflight_task_seqs, (4,))
        self.assertTrue(self.manager.validate(report(4, worker_id=1)).accepted)

    def test_a_duplicate_does_not_release_another_workers_in_flight_task(self) -> None:
        submit(self.manager, 1, worker_id=0)
        self.assertTrue(self.manager.validate(report(1, worker_id=0)).accepted)
        submit(self.manager, 2, worker_id=0)
        self.manager.validate(report(1, worker_id=0))  # the late copy
        self.assertEqual(self.manager.inflight_task_seqs, (2,))


class FreshnessTests(unittest.TestCase):
    def test_a_task_older_than_the_budget_is_expired(self) -> None:
        manager = ResultManager(result_max_age_ms=100.0)
        submit(manager, 1, created_ns=time.monotonic_ns() - 400_000_000)
        verdict = manager.validate(report(1))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_EXPIRED)

    def test_a_task_inside_the_budget_is_accepted(self) -> None:
        manager = ResultManager(result_max_age_ms=10_000.0)
        submit(manager, 1)
        self.assertTrue(manager.validate(report(1)).accepted)


class CaptureClockTests(unittest.TestCase):
    """The camera clock is checked at submit, and on a *relative* delta.

    Nothing here compares a capture timestamp against a wall clock: camera
    timestamps arrive in whatever domain the config names, so the only portable
    test is how far one capture sits from the previous one.
    """

    def setUp(self) -> None:
        self.manager = ResultManager(max_capture_lag_ms=500.0)

    def test_a_backwards_step_is_a_rollback(self) -> None:
        submit(self.manager, 1, capture_us=10_000_000)
        submit(self.manager, 2, capture_us=10_400_000)
        submit(self.manager, 3, capture_us=9_000_000)
        self.assertEqual(self.manager.count(CAPTURE_ROLLBACK), 1)
        self.assertEqual(self.manager.count(CAPTURE_JUMP), 0)

    def test_a_large_forward_step_is_not_a_jump(self) -> None:
        submit(self.manager, 1, capture_us=10_000_000)
        submit(self.manager, 2, capture_us=10_400_000)
        submit(self.manager, 3, capture_us=11_000_000)  # 600 ms on, past 500 ms
        self.assertEqual(self.manager.count(CAPTURE_JUMP), 0)
        self.assertEqual(self.manager.count(CAPTURE_ROLLBACK), 0)
        self.assertEqual(self.manager._newest_capture_us, 11_000_000)
        self.assertNotIn(3, self.manager._anomalous)

    def test_a_slow_frame_is_not_a_clock_fault(self) -> None:
        submit(self.manager, 1, capture_us=10_000_000)
        submit(self.manager, 2, capture_us=10_400_000)  # inside the bound
        self.assertEqual(self.manager.count(CAPTURE_ROLLBACK), 0)
        self.assertEqual(self.manager.count(CAPTURE_JUMP), 0)

    def test_a_large_forward_step_becomes_the_new_reference(self) -> None:
        """A wide gap must move the reference, or every later window looks anomalous."""
        submit(self.manager, 1, capture_us=10_000_000)
        submit(self.manager, 2, capture_us=40_000_000)  # 30 s on: adopted
        submit(self.manager, 3, capture_us=40_400_000)  # sane against the new reference
        self.assertEqual(self.manager.count(CAPTURE_JUMP), 0)
        self.assertEqual(self.manager.count(CAPTURE_ROLLBACK), 0)
        self.assertEqual(self.manager._newest_capture_us, 40_400_000)

    def test_a_flagged_task_is_dropped_even_with_a_perfect_trajectory(self) -> None:
        submit(self.manager, 1, capture_us=10_000_000)
        submit(self.manager, 2, capture_us=9_000_000)  # 1 s backwards
        verdict = self.manager.validate(report(1))
        self.assertTrue(verdict.accepted)
        verdict = self.manager.validate(report(2))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_CAPTURE_ANOMALY)

    def test_a_task_behind_the_newest_capture_window_is_dropped(self) -> None:
        submit(self.manager, 1, capture_us=10_000_000)
        submit(self.manager, 2, capture_us=10_400_000)
        submit(self.manager, 3, capture_us=10_800_000)
        # Task 1 is now 800 ms behind the newest window seen (limit 500).
        verdict = self.manager.validate(report(1))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_CAPTURE_LAG)

    def test_a_discarded_task_does_not_leave_its_flag_behind(self) -> None:
        submit(self.manager, 1, capture_us=10_000_000)
        task = submit(self.manager, 2, capture_us=9_000_000)
        self.manager.discard_task(task)
        self.assertEqual(self.manager._anomalous, set())


class TrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ResultManager()

    def _verdict_for(self, positions):
        submit(self.manager, 1)
        return self.manager.validate(report(1, positions=positions))

    def test_a_missing_trajectory_is_invalid(self) -> None:
        verdict = self._verdict_for(None)
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_INVALID_SHAPE)

    def test_a_one_dimensional_trajectory_is_invalid(self) -> None:
        verdict = self._verdict_for(np.zeros(16))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_INVALID_SHAPE)

    def test_a_trajectory_with_one_column_is_invalid(self) -> None:
        """The crop and the coordinate transform need x *and* y."""
        verdict = self._verdict_for(np.zeros((16, 1)))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_INVALID_SHAPE)

    def test_an_empty_trajectory_is_invalid(self) -> None:
        verdict = self._verdict_for(np.zeros((0, 3)))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_INVALID_POINTS)

    def test_a_nan_coordinate_is_invalid(self) -> None:
        positions = good_positions()
        positions[3, 1] = np.nan
        verdict = self._verdict_for(positions)
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_NON_FINITE)

    def test_an_infinite_coordinate_is_invalid(self) -> None:
        positions = good_positions()
        positions[0, 0] = np.inf
        verdict = self._verdict_for(positions)
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_NON_FINITE)

    def test_a_single_point_trajectory_is_usable(self) -> None:
        verdict = self._verdict_for(np.zeros((1, 3)))
        self.assertTrue(verdict.accepted)

    def test_extra_columns_are_allowed(self) -> None:
        """The model emits x, y, z; a future variant may emit more."""
        verdict = self._verdict_for(np.zeros((16, 5)))
        self.assertTrue(verdict.accepted)


class WorkerFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ResultManager()

    def test_a_failed_result_counts_as_a_worker_error(self) -> None:
        submit(self.manager, 1)
        verdict = self.manager.validate(
            report(1, ok=False, error="boom", error_kind=ERROR_WORKER_EXCEPTION)
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_WORKER_ERROR)

    def test_a_cuda_oom_counts_once_under_each_name_that_fits(self) -> None:
        submit(self.manager, 1)
        self.manager.validate(
            report(1, ok=False, error="out of memory", error_kind=ERROR_CUDA_OOM)
        )
        stats = self.manager.stats()
        self.assertEqual(stats[DROP_WORKER_ERROR], 1)
        self.assertEqual(stats[ERROR_CUDA_OOM], 1)
        self.assertEqual(stats[ERROR_WORKER_EXCEPTION], 0)

    def test_a_worker_failure_releases_its_slot(self) -> None:
        submit(self.manager, 1, worker_id=0)
        self.manager.validate(report(1, worker_id=0, ok=False, error="boom"))
        self.assertEqual(self.manager.inflight_task_seqs, ())
        # And the worker's own record too, so the next task it takes is not
        # mistaken for the one that just failed.
        submit(self.manager, 2, worker_id=0)
        self.assertTrue(self.manager.validate(report(2, worker_id=0)).accepted)


class CountingTests(unittest.TestCase):
    """Each drop bumps exactly one counter, once.

    The failure this guards against is real and was in the code: ``drop()``
    bumped the reason and a branch bumped it again first, so every expired
    result counted twice and a 2 Hz expiry read as 4 Hz.
    """

    def test_an_expired_result_is_counted_once(self) -> None:
        manager = ResultManager(result_max_age_ms=1.0)
        submit(manager, 1, created_ns=time.monotonic_ns() - 500_000_000)
        self._assert_single_drop(manager, manager.validate(report(1)), DROP_EXPIRED)

    def test_a_worker_error_is_counted_once(self) -> None:
        manager = ResultManager()
        submit(manager, 1)
        self._assert_single_drop(
            manager,
            manager.validate(report(1, ok=False, error="boom")),
            DROP_WORKER_ERROR,
        )

    def test_an_invalid_trajectory_is_counted_once(self) -> None:
        manager = ResultManager()
        submit(manager, 1)
        self._assert_single_drop(
            manager, manager.validate(report(1, positions=None)), DROP_INVALID_SHAPE
        )

    def test_a_non_finite_trajectory_is_counted_once(self) -> None:
        manager = ResultManager()
        submit(manager, 1)
        positions = good_positions()
        positions[0, 0] = np.nan
        self._assert_single_drop(
            manager,
            manager.validate(report(1, positions=positions)),
            DROP_NON_FINITE,
        )

    def test_a_capture_anomaly_is_counted_once(self) -> None:
        manager = ResultManager(max_capture_lag_ms=500.0)
        submit(manager, 1, capture_us=10_000_000)
        submit(manager, 2, capture_us=9_000_000)
        manager.validate(report(1))
        self._assert_single_drop(
            manager, manager.validate(report(2)), DROP_CAPTURE_ANOMALY
        )

    def test_a_capture_lag_is_counted_once(self) -> None:
        manager = ResultManager(max_capture_lag_ms=500.0)
        submit(manager, 1, capture_us=10_000_000)
        submit(manager, 2, capture_us=10_400_000)
        submit(manager, 3, capture_us=10_900_000)
        # Task 1 is now 900 ms behind the newest window seen; the limit is 500.
        self._assert_single_drop(
            manager, manager.validate(report(1)), DROP_CAPTURE_LAG
        )

    def _assert_single_drop(self, manager, verdict, expected: str) -> None:
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, expected)
        self.assertEqual(
            {name: manager.count(name) for name in ALL_REASONS if manager.count(name)},
            {expected: 1},
            f"the {expected} drop should be the only counter that moved",
        )


class BookkeepingTests(unittest.TestCase):
    def test_discarding_a_worker_forgets_its_task(self) -> None:
        manager = ResultManager()
        submit(manager, 1, worker_id=3)
        self.assertEqual(manager.discard_worker(3).task_seq, 1)
        self.assertEqual(manager.inflight_task_seqs, ())
        # Its late result now finds nothing.
        self.assertEqual(manager.validate(report(1, worker_id=3)).reason, DROP_UNKNOWN_TASK)

    def test_discarding_a_worker_that_holds_nothing_is_harmless(self) -> None:
        manager = ResultManager()
        self.assertIsNone(manager.discard_worker(9))

    def test_the_stats_dict_has_every_reason_a_reader_expects(self) -> None:
        stats = ResultManager().stats()
        for key in (
            "out_of_order",
            "duplicate",
            "expired",
            "capture_lag",
            "capture_anomaly",
            "capture_rollback",
            "capture_jump",
            "invalid_trajectory_shape",
            "invalid_trajectory_points",
            "non_finite_trajectory",
            "worker_error",
            "cuda_oom",
            "worker_exception",
            "unknown_task",
            "task_seq_mismatch",
        ):
            self.assertIn(key, stats)
            self.assertEqual(stats[key], 0)


if __name__ == "__main__":
    unittest.main()
