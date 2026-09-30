"""The GPU pool, exercised deterministically with fake workers.

No CUDA, no real processes, no sleeps used as synchronisation.  Every property
here is about *order* - which task gets which worker, which result is allowed
through, which one is discarded - and order a test chooses is worth more than
order it observes.

The process model itself (``spawn``, one CUDA context per child, real
throughput) is not testable here and is covered by
``tools/multi_gpu_offline_test.py`` on a GPU box.
"""

from __future__ import annotations

import ast
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

from config.loader import InferencePoolConfig
from inference.fake_worker import fake_worker_main
from inference.gpu_pool import (
    DROP_PENDING_REPLACED,
    GpuInfo,
    GpuInventory,
    GpuScheduler,
    SUBMIT_DISPATCHED,
    SUBMIT_DROPPED,
    SUBMIT_QUEUED,
    SUBMIT_REPLACED,
    validate_pool_startup,
)
from inference.metrics import InferenceMetrics
from inference.result_manager import (
    DROP_OUT_OF_ORDER,
    DROP_TASK_SEQ_MISMATCH,
    DROP_UNKNOWN_TASK,
    ResultManager,
)
from inference.tasks import (
    ERROR_CUDA_OOM,
    WORKER_FAILED,
    WORKER_READY,
    InferenceResult,
)

from _fake_pool import FakeContext, make_task


class PoolHarness:
    """A running pool over fake workers, with exact control of completion order.

    Gates are keyed by ``task_seq``: a fake worker blocks on its task's gate
    until the test releases it, so "task 3 finishes before task 2" is something
    the test states, not a race it hopes to win.
    """

    def __init__(
        self,
        *,
        devices: tuple[int, ...] = (1, 3, 4, 5),
        latest_only: bool = True,
        pending_capacity: int = 1,
        result_max_age_ms: float = 2500.0,
        max_capture_lag_ms: float = 2500.0,
        **worker_options: object,
    ) -> None:
        self.gates: dict[int, threading.Event] = {}
        self.context = FakeContext()
        self.config = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=devices,
            latest_only=latest_only,
            pending_capacity=pending_capacity,
            result_max_age_ms=result_max_age_ms,
            max_capture_lag_ms=max_capture_lag_ms,
        )
        self.settings = SimpleNamespace(name="fake-settings")
        self.metrics = InferenceMetrics()
        self.result_manager = ResultManager(
            result_max_age_ms=result_max_age_ms,
            max_capture_lag_ms=max_capture_lag_ms,
        )
        #: ``(task_seq, worker_id)`` for every result the pool let through.
        self.accepted: list[tuple[int, int]] = []
        self.scheduler = GpuScheduler(
            settings=self.settings,
            pool_config=self.config,
            metrics=self.metrics,
            result_manager=self.result_manager,
            on_accepted=self._on_accepted,
            context=self.context,
            worker_entry=fake_worker_main,
            # Passed through to every fake worker.  Set here rather than after
            # ``start()``: a worker copies its options as it enters, so a change
            # made once it is running is a change it will never see.
            worker_options={"gates": self.gates, **worker_options},
        )

    def start(self) -> None:
        self.scheduler.start()
        self.wait_for(
            lambda: self.scheduler.ready_worker_count
            == len(self.config.gpu_devices)
        )

    def _on_accepted(self, result, task) -> None:
        self.accepted.append((int(result.task_seq), int(result.worker_id)))

    # -- driving the scheduler --------------------------------------------- #

    def gate(self, task_seq: int) -> threading.Event:
        return self.gates.setdefault(int(task_seq), threading.Event())

    def release(self, *task_seqs: int) -> None:
        for seq in task_seqs:
            self.gate(seq).set()

    def submit(self, task_seq: int, **kwargs) -> str:
        self.gate(task_seq)
        return self.scheduler.submit(make_task(task_seq, **kwargs))

    def pump(self, predicate=None, timeout: float = 5.0) -> bool:
        """Run the scheduler until ``predicate`` holds (or the budget runs out)."""
        deadline = time.monotonic() + timeout
        while True:
            self.scheduler.tick()
            if predicate is not None and predicate():
                return True
            if time.monotonic() >= deadline:
                return predicate is None
            time.sleep(0.002)

    def wait_for(self, predicate, timeout: float = 5.0) -> None:
        if not self.pump(predicate, timeout=timeout):
            raise AssertionError("condition never became true")

    def worker_task_seqs(self) -> list[int | None]:
        return [h.task.task_seq if h.task else None for h in self.scheduler._handles]

    def stop(self, *, release_gates: bool = True, **kwargs) -> dict:
        if release_gates:
            for gate in self.gates.values():
                gate.set()
        return self.scheduler.stop(**kwargs)


class ImmediateDispatchTests(unittest.TestCase):
    """Spec 1, 5: a free worker takes the task now; task_seq is the only key."""

    def test_an_idle_worker_takes_the_task_immediately(self) -> None:
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        self.assertEqual(pool.submit(1), SUBMIT_DISPATCHED)
        self.assertEqual(pool.scheduler.busy_worker_count, 1)
        self.assertEqual(pool.scheduler.pending_count, 0)
        pool.stop()

    def test_each_idle_worker_gets_one_task_until_none_are_left(self) -> None:
        pool = PoolHarness(devices=(1, 3, 4))
        pool.start()
        for seq in (1, 2, 3):
            self.assertEqual(pool.submit(seq), SUBMIT_DISPATCHED)
        self.assertEqual(sorted(pool.worker_task_seqs()), [1, 2, 3])
        pool.stop()

    def test_task_seq_is_strictly_increasing_and_unique(self) -> None:
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        allocated = [pool.scheduler.next_task_seq() for _ in range(50)]
        self.assertEqual(allocated, list(range(1, 51)))
        pool.stop()


class LatestOnlyPendingTests(unittest.TestCase):
    """Spec 2, 3, 4: one pending slot, newest wins, no FIFO backlog."""

    def test_only_one_task_is_kept_while_every_worker_is_busy(self) -> None:
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        pool.submit(1)
        pool.submit(2)
        self.assertEqual(pool.submit(3), SUBMIT_QUEUED)
        self.assertEqual(pool.scheduler.pending_count, 1)
        self.assertEqual(pool.scheduler.pending_task_seq, 3)
        pool.stop()

    def test_a_new_task_replaces_the_pending_one(self) -> None:
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        pool.submit(1)
        pool.submit(2)
        pool.submit(3)
        self.assertEqual(pool.submit(4), SUBMIT_REPLACED)
        self.assertEqual(pool.scheduler.pending_task_seq, 4)
        self.assertEqual(pool.scheduler.pending_count, 1)
        self.assertEqual(pool.metrics.drops.get(DROP_PENDING_REPLACED), 1)
        pool.stop()

    def test_no_fifo_backlog_builds_up(self) -> None:
        """Twenty tasks into two busy workers: the slot never exceeds one."""
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        pool.submit(1)
        pool.submit(2)
        for seq in range(3, 21):
            pool.submit(seq)
            self.assertLessEqual(pool.scheduler.pending_count, 1)
        self.assertEqual(pool.scheduler.pending_task_seq, 20)
        # 3..19 were each displaced by the next one; 20 is still waiting.
        self.assertEqual(pool.metrics.drops.get(DROP_PENDING_REPLACED), 17)
        self.assertEqual(pool.worker_task_seqs(), [1, 2])
        pool.stop()

    def test_a_freed_worker_takes_the_newest_pending_task(self) -> None:
        """Spec 4: whoever frees up takes the newest task, not the oldest."""
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        pool.submit(1)
        pool.submit(2)
        for seq in range(3, 11):
            pool.submit(seq)
        pool.release(1)
        pool.wait_for(lambda: pool.worker_task_seqs()[0] == 10)
        # Worker 0 finished task 1 and picked up task 10 in the same tick - not
        # task 3, which is what a FIFO would have given it.
        self.assertNotIn(3, pool.worker_task_seqs())
        self.assertEqual(pool.scheduler.pending_count, 0)
        pool.stop()

    def test_a_short_fifo_is_used_when_latest_only_is_off(self) -> None:
        """The other policy is reachable, and trades the other way."""
        pool = PoolHarness(devices=(1, 3), latest_only=False, pending_capacity=2)
        pool.start()
        pool.submit(1)
        pool.submit(2)
        self.assertEqual(pool.submit(3), SUBMIT_QUEUED)
        self.assertEqual(pool.submit(4), SUBMIT_QUEUED)
        self.assertEqual(pool.scheduler.pending_count, 2)
        # The slot is full: the newcomer is refused rather than displacing.
        self.assertEqual(pool.submit(5), SUBMIT_DROPPED)
        self.assertEqual(pool.scheduler.pending_task_seq, 3)
        pool.stop()


class OutOfOrderTests(unittest.TestCase):
    """Spec 6, 7: only a strictly increasing task_seq reaches the vehicle."""

    def test_results_arriving_1_3_2_5_4_leave_2_and_4_dropped(self) -> None:
        pool = PoolHarness(devices=(1, 3, 4, 5))
        pool.start()
        for seq in range(1, 6):
            pool.submit(seq)
        self.assertEqual(pool.scheduler.pending_task_seq, 5)

        pool.release(1)
        pool.wait_for(lambda: bool(pool.accepted))
        pool.release(3)
        pool.wait_for(lambda: pool.accepted[-1][0] == 3)
        # Task 2 finishes *after* 3: it must be discarded, never sent.
        pool.release(2)
        pool.wait_for(lambda: pool.result_manager.count(DROP_OUT_OF_ORDER) == 1)
        pool.release(5)
        pool.wait_for(lambda: pool.accepted[-1][0] == 5)
        pool.release(4)
        pool.wait_for(lambda: pool.result_manager.count(DROP_OUT_OF_ORDER) == 2)

        self.assertEqual([seq for seq, _ in pool.accepted], [1, 3, 5])
        self.assertEqual(pool.result_manager.last_accepted_task_seq, 5)
        self.assertEqual(pool.stop()["stopped"], [0, 1, 2, 3])

    def test_an_out_of_order_result_never_reaches_the_callback(self) -> None:
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        pool.submit(1)
        pool.submit(2)
        pool.release(2)
        pool.wait_for(lambda: bool(pool.accepted))
        pool.release(1)
        pool.wait_for(lambda: pool.result_manager.count(DROP_OUT_OF_ORDER) == 1)
        self.assertEqual([seq for seq, _ in pool.accepted], [2])
        pool.stop()

    def test_a_repeated_result_is_dropped(self) -> None:
        pool = PoolHarness(devices=(1,))
        pool.start()
        pool.submit(1)
        pool.release(1)
        pool.wait_for(lambda: bool(pool.accepted))
        # A second copy of an already-accepted result, as a duplicated queue
        # message would be.
        verdict = pool.result_manager.validate(
            InferenceResult(task_seq=1, worker_id=0, device=1, ok=True)
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, "duplicate")
        self.assertEqual([seq for seq, _ in pool.accepted], [1])
        pool.stop()

    def test_a_duplicate_does_not_release_the_slot_of_the_next_task(self) -> None:
        """A late copy of an old result must not orphan the newer task."""
        pool = PoolHarness(devices=(1,))
        pool.start()
        pool.submit(1)
        pool.release(1)
        pool.wait_for(lambda: bool(pool.accepted))
        pool.submit(2)  # dispatched to the now-idle worker
        # The duplicate arrives while task 2 is in flight.
        pool.result_manager.validate(
            InferenceResult(task_seq=1, worker_id=0, device=1, ok=True)
        )
        self.assertEqual(pool.result_manager.inflight_task_seqs, (2,))
        pool.release(2)
        pool.wait_for(lambda: len(pool.accepted) == 2)
        self.assertEqual([seq for seq, _ in pool.accepted], [1, 2])
        pool.stop()

    def test_a_result_the_worker_was_never_given_is_a_mismatch(self) -> None:
        pool = PoolHarness(devices=(1,))
        pool.start()
        pool.submit(1)
        verdict = pool.result_manager.validate(
            InferenceResult(task_seq=9, worker_id=0, device=1, ok=True)
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_TASK_SEQ_MISMATCH)
        # The real task 1 is still waiting for its own result.
        self.assertEqual(pool.result_manager.inflight_task_seqs, (1,))
        pool.release(1)
        pool.wait_for(lambda: bool(pool.accepted))
        pool.stop()

    def test_a_result_for_no_known_task_is_unknown(self) -> None:
        pool = PoolHarness(devices=(1,))
        pool.start()
        verdict = pool.result_manager.validate(
            InferenceResult(task_seq=42, worker_id=7, device=1, ok=True)
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, DROP_UNKNOWN_TASK)
        pool.stop()


class PerTaskDelayTests(unittest.TestCase):
    """Which task is slow has to be choosable per task, not per inference count.

    ``delays_ms`` is cycled by how many inferences *that worker* has run, and
    with four workers each taking its own path through the list there is no way
    to say "task 4 is the slow one" - every worker's first task gets the same
    delay.  ``delays_by_task_seq`` is the override the ordering scenario needs,
    and it is also what makes that scenario picklable: a dict of numbers crosses
    a ``spawn`` boundary, a ``threading.Event`` does not.
    """

    def test_a_listed_task_uses_its_own_delay(self) -> None:
        pool = PoolHarness(devices=(1, 3), delays_ms=[0.0], delays_by_task_seq={1: 300.0})
        pool.start()
        pool.submit(1)
        pool.submit(2)
        pool.release(1, 2)
        pool.wait_for(lambda: pool.scheduler.busy_worker_count == 0)
        pool.pump(lambda: pool.metrics.infer_ms.count >= 2)

        by_device = {
            view.device: view.avg_infer_ms for view in pool.scheduler.worker_views()
        }
        # Task 1 goes to the first idle worker, which is the one on device 1.
        self.assertGreaterEqual(by_device[1] or 0.0, 250.0)
        self.assertLessEqual(by_device[3] or 0.0, 100.0)
        pool.stop()

    def test_a_task_with_no_entry_still_uses_the_cycled_list(self) -> None:
        """The dict is an override, not a replacement: a task missing from it
        must fall back to ``delays_ms``, not silently become instantaneous."""
        pool = PoolHarness(devices=(1, 3), delays_ms=[200.0], delays_by_task_seq={1: 0.0})
        pool.start()
        pool.submit(1)
        pool.submit(2)
        pool.release(1, 2)
        pool.wait_for(lambda: pool.scheduler.busy_worker_count == 0)
        pool.pump(lambda: pool.metrics.infer_ms.count >= 2)

        by_device = {
            view.device: view.avg_infer_ms for view in pool.scheduler.worker_views()
        }
        self.assertLessEqual(by_device[1] or 0.0, 100.0)
        self.assertGreaterEqual(by_device[3] or 0.0, 150.0)
        pool.stop()


class StaleAndAnomalousTests(unittest.TestCase):
    """Spec 8, 9: age and camera-clock faults are drops, not reorderings."""

    def test_an_expired_result_is_dropped(self) -> None:
        pool = PoolHarness(devices=(1,), result_max_age_ms=50.0)
        pool.start()
        pool.submit(1, created_ns=time.monotonic_ns() - 5_000_000_000)
        pool.release(1)
        pool.wait_for(lambda: pool.result_manager.count("expired") == 1)
        self.assertEqual(pool.accepted, [])
        pool.stop()

    def test_the_expired_count_is_not_doubled(self) -> None:
        """Each drop is counted once: one task, one expired drop."""
        pool = PoolHarness(devices=(1,), result_max_age_ms=50.0)
        pool.start()
        pool.submit(1, created_ns=time.monotonic_ns() - 5_000_000_000)
        pool.release(1)
        pool.wait_for(lambda: pool.result_manager.count("expired") == 1)
        pool.pump(timeout=0.05)
        self.assertEqual(pool.result_manager.count("expired"), 1)
        self.assertEqual(pool.result_manager.stats()["expired"], 1)
        pool.stop()

    def test_a_camera_clock_that_steps_backwards_is_flagged(self) -> None:
        pool = PoolHarness(devices=(1,), max_capture_lag_ms=500.0)
        pool.start()
        pool.submit(1, capture_us=10_000_000)
        pool.submit(2, capture_us=10_100_000)
        # A two-second step backwards: a clock fault, not a slow frame.
        pool.submit(3, capture_us=8_000_000)
        self.assertEqual(pool.result_manager.count("rollback"), 1)
        self.assertEqual(pool.result_manager.count("jump"), 0)
        pool.stop()

    def test_a_camera_clock_that_jumps_forward_is_not_flagged(self) -> None:
        pool = PoolHarness(devices=(1,), max_capture_lag_ms=500.0)
        pool.start()
        pool.submit(1, capture_us=10_000_000)
        pool.submit(2, capture_us=10_100_000)
        pool.submit(3, capture_us=20_000_000)  # 9.9 s forward: not an anomaly
        self.assertEqual(pool.result_manager.count("jump"), 0)
        self.assertEqual(pool.result_manager.count("rollback"), 0)
        self.assertEqual(pool.result_manager._newest_capture_us, 20_000_000)
        pool.stop()

    def test_a_rollback_is_dropped_even_though_its_sequence_is_newer(self) -> None:
        """A later sequence does not excuse a clock that stepped backwards."""
        pool = PoolHarness(
            devices=(1,), max_capture_lag_ms=500.0, result_max_age_ms=10_000.0
        )
        pool.start()
        pool.submit(1, capture_us=10_000_000)
        pool.release(1)
        pool.wait_for(lambda: bool(pool.accepted))
        pool.submit(2, capture_us=8_000_000)  # 2 s backwards
        pool.release(2)
        pool.wait_for(
            lambda: pool.result_manager.count("capture_timestamp_anomaly") == 1
        )
        self.assertEqual([seq for seq, _ in pool.accepted], [1])
        pool.stop()

    def test_a_task_left_behind_by_a_newer_capture_window_is_dropped(self) -> None:
        pool = PoolHarness(
            devices=(1,), max_capture_lag_ms=1000.0, result_max_age_ms=10_000.0
        )
        pool.start()
        pool.submit(1, capture_us=10_000_000)  # in flight on the only worker
        pool.submit(2, capture_us=10_500_000)  # 500 ms on: still sane
        pool.submit(3, capture_us=11_200_000)  # 700 ms on: still sane
        # Task 1's window is now 1.2 s behind the newest one seen.  It is the
        # oldest task but still admissible by sequence, so only the capture-lag
        # check can stop it.
        pool.release(1)
        pool.wait_for(lambda: pool.result_manager.count("capture_lag") == 1)
        self.assertEqual(pool.accepted, [])
        pool.stop()


class WorkerFailureTests(unittest.TestCase):
    """Spec 10, 11: a broken worker is contained, and reported."""

    def test_a_worker_exception_is_reported_and_the_pool_keeps_running(self) -> None:
        pool = PoolHarness(devices=(1, 3), fail_task_seqs=[1])
        pool.start()
        pool.submit(1)
        pool.release(1)
        pool.wait_for(lambda: pool.result_manager.count("worker_error") == 1)
        self.assertEqual(pool.accepted, [])
        self.assertEqual(pool.scheduler.ready_worker_count, 2)
        self.assertEqual(pool.result_manager.count("worker_exception"), 1)
        pool.stop()

    def test_a_cuda_oom_is_counted_separately_from_a_generic_failure(self) -> None:
        pool = PoolHarness(devices=(1,), oom_task_seqs=[1])
        pool.start()
        pool.submit(1)
        pool.release(1)
        pool.wait_for(lambda: pool.result_manager.count(ERROR_CUDA_OOM) == 1)
        stats = pool.result_manager.stats()
        # worker_error counts every failed result; cuda_oom and worker_exception
        # are its disjoint parts.
        self.assertEqual(stats["worker_error"], 1)
        self.assertEqual(stats["cuda_oom"], 1)
        self.assertEqual(stats["worker_exception"], 0)
        pool.stop()

    def test_a_failed_task_does_not_stop_later_tasks(self) -> None:
        pool = PoolHarness(devices=(1,), fail_task_seqs=[1])
        pool.start()
        pool.submit(1)
        pool.release(1)
        pool.wait_for(lambda: pool.result_manager.count("worker_error") == 1)
        pool.submit(2)
        pool.release(2)
        pool.wait_for(lambda: bool(pool.accepted))
        self.assertEqual([seq for seq, _ in pool.accepted], [2])
        pool.stop()

    def test_one_failed_worker_does_not_stop_the_others(self) -> None:
        pool = PoolHarness(devices=(1, 3))
        pool.start()
        pool.submit(1)
        pool.submit(2)
        # Worker 0 dies outright: no result, no status, just a process that is
        # gone - which is exactly what the liveness poll exists to catch.
        pool.scheduler._handles[0].process._alive = False
        pool.scheduler._handles[0].process.exitcode = 137
        pool.release(2)
        pool.wait_for(lambda: bool(pool.accepted))
        pool.pump(lambda: pool.scheduler.failed_worker_count == 1, timeout=2.0)
        self.assertEqual([seq for seq, _ in pool.accepted], [2])
        self.assertEqual(pool.scheduler.failed_worker_count, 1)
        self.assertEqual(pool.scheduler._handles[1].state, WORKER_READY)
        pool.stop()

    def test_a_worker_that_fails_to_load_is_reported(self) -> None:
        pool = PoolHarness(devices=(1,), load_failure="checkpoint missing")
        pool.scheduler.start()  # it can never become ready; do not wait for it
        pool.pump(lambda: pool.scheduler.failed_worker_count == 1)
        view = pool.scheduler.worker_views()[0]
        self.assertEqual(view.state, WORKER_FAILED)
        self.assertIn("checkpoint missing", view.error or "")
        self.assertEqual(pool.scheduler.ready_worker_count, 0)
        pool.stop()


class ShutdownTests(unittest.TestCase):
    """Spec 12: shutdown is bounded, and it says what it had to force."""

    def test_a_cooperative_pool_stops_within_its_budget(self) -> None:
        pool = PoolHarness(devices=(1, 3, 4))
        pool.start()
        started = time.monotonic()
        report = pool.stop(drain_s=1.0, terminate_s=1.0)
        elapsed = time.monotonic() - started
        self.assertEqual(report["stopped"], [0, 1, 2])
        self.assertEqual(report["terminated"], [])
        self.assertEqual(report["killed"], [])
        self.assertLess(elapsed, 3.0)

    def test_a_worker_stuck_in_an_inference_does_not_stall_shutdown(self) -> None:
        pool = PoolHarness(devices=(1,), gate_timeout_s=5.0)
        pool.start()
        pool.submit(1)  # blocked on a gate, holding the only worker
        started = time.monotonic()
        report = pool.stop(drain_s=1.0, terminate_s=1.0)
        elapsed = time.monotonic() - started
        self.assertEqual(report["workers"], 1)
        self.assertLess(elapsed, 3.0)

    def test_a_worker_that_ignores_the_sentinel_is_terminated(self) -> None:
        """The budget is a deadline, not a formality."""
        pool = PoolHarness(devices=(1,), gate_timeout_s=5.0)
        pool.start()
        pool.submit(1)
        started = time.monotonic()
        # No gate release: the worker cannot reach its sentinel.
        report = pool.stop(release_gates=False, drain_s=0.3, terminate_s=0.3)
        elapsed = time.monotonic() - started
        self.assertEqual(report["stopped"], [])
        self.assertEqual(report["terminated"], [0])
        self.assertLess(elapsed, 2.5)
        # Let the fake worker's own gate timeout elapse rather than leaking a
        # blocked thread into the rest of the suite.
        pool.release(1)

    def test_stopping_an_unstarted_pool_is_a_no_op(self) -> None:
        pool = PoolHarness(devices=(1,))
        report = pool.scheduler.stop()
        self.assertEqual(report["workers"], 0)
        self.assertIsNone(pool.scheduler.tick())

    def test_shutdown_clears_a_pending_task(self) -> None:
        pool = PoolHarness(devices=(1,))
        pool.start()
        pool.submit(1)
        pool.submit(2)  # pending
        self.assertEqual(pool.scheduler.pending_count, 1)
        pool.stop()
        self.assertEqual(pool.scheduler.pending_count, 0)


class StartupValidationTests(unittest.TestCase):
    """An invalid config fails at startup, with a message that names the fault."""

    def test_a_device_that_does_not_exist_is_refused(self) -> None:
        config = InferencePoolConfig(mode="multi_process", gpu_devices=(0, 7))
        inventory = GpuInventory(
            available=True,
            source="pynvml",
            devices=(GpuInfo(0, "RTX 4090"), GpuInfo(1, "RTX 4090")),
        )
        problem = validate_pool_startup(config, inventory)
        self.assertIsNotNone(problem)
        self.assertIn("7", problem or "")

    def test_an_empty_inventory_does_not_reject_a_valid_config(self) -> None:
        """No visible device is not the same finding as a bad device number."""
        config = InferencePoolConfig(mode="multi_process", gpu_devices=(0, 1))
        inventory = GpuInventory(
            available=True, source="torch", devices=(), error="no CUDA device"
        )
        self.assertIsNone(validate_pool_startup(config, inventory))

    def test_an_unreadable_inventory_does_not_reject_a_valid_config(self) -> None:
        config = InferencePoolConfig(mode="multi_process", gpu_devices=(0, 1))
        inventory = GpuInventory(available=False, source="none", error="no nvml")
        self.assertIsNone(validate_pool_startup(config, inventory))

    def test_duplicate_devices_are_refused(self) -> None:
        problem = InferencePoolConfig(
            mode="multi_process", gpu_devices=(1, 1)
        ).validate()
        self.assertIsNotNone(problem)
        self.assertIn("twice", problem or "")

    def test_latest_only_with_a_larger_capacity_is_refused(self) -> None:
        problem = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=(1,),
            latest_only=True,
            pending_capacity=4,
        ).validate()
        self.assertIsNotNone(problem)
        self.assertIn("pending_capacity", problem or "")

    def test_minimum_ready_workers_above_the_worker_count_is_refused(self) -> None:
        problem = InferencePoolConfig(
            mode="multi_process",
            gpu_devices=(1, 3),
            minimum_ready_workers=4,
        ).validate()
        self.assertIsNotNone(problem)
        self.assertIn("minimum_ready_workers", problem or "")

    def test_a_negative_device_index_is_refused(self) -> None:
        # ``_read_device_list`` keeps an unparseable entry as -1 so the message
        # can show the offending value rather than silently dropping a worker.
        problem = InferencePoolConfig(
            mode="multi_process", gpu_devices=(-1,)
        ).validate()
        self.assertIsNotNone(problem)
        self.assertIn("negative", problem or "")


class InventoryIsReadOnlyTests(unittest.TestCase):
    """``probe_gpu_inventory`` must not be what allocates the memory it reports.

    It runs at startup, before the pool commits anything, on a machine that is
    shared.  Measured on the eight-GPU deployment box: asking torch for each
    card's free memory creates a CUDA context on *that* card, ~390MiB each,
    across the whole machine - including a card another user had a job on.  That
    is a side effect on someone else's work caused by a function whose entire
    contract is to look and not touch.

    Asserted against the source, because the property is "this call is not made
    here": a behavioural test would need a torch that has a CUDA device, which
    is the one thing a test runner does not have.
    """

    def setUp(self) -> None:
        self.source = (ROOT / "inference" / "gpu_pool.py").read_text(encoding="utf-8")
        # Parsed rather than grepped: the module explains at length *why* this
        # call is absent, and a text search cannot tell an explanation from a
        # use.  Docstrings and comments are not ``ast.Attribute`` nodes.
        self.tree = ast.parse(self.source)

    def _attributes(self) -> list[str]:
        return [
            node.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Attribute)
        ]

    def test_the_module_never_calls_mem_get_info(self) -> None:
        self.assertNotIn(
            "mem_get_info",
            self._attributes(),
            "torch.cuda.mem_get_info needs an active context, so calling it for "
            "each card creates a context on each card - machine-wide, other "
            "people's cards included.  Use NVML or nvidia-smi for free memory.",
        )

    def test_the_torch_fallback_still_reports_total_memory(self) -> None:
        """Dropping the context was allowed to cost free memory, not the whole
        reading: the device list and totals are what the pool's startup check
        actually uses."""
        self.assertIn("get_device_properties", self._attributes())


if __name__ == "__main__":
    unittest.main()
