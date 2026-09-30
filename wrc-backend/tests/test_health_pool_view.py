"""The pool's readiness verdict, and why it is not the inline one.

The change that needs guarding is small and easy to get backwards: in
multi-process mode the parent process never loads the checkpoint - the workers
own the devices - so ``snapshot.model_loaded`` is False forever on a perfectly
healthy pool.  A readiness check that read it directly would report 503 for the
lifetime of a working process, which is worse than no check at all, because the
deploy script gates on it.

So there are two things to pin here: that a pool is judged by its workers, and
that inline mode - where there is no pool - is judged exactly as it was before
any of this existed.
"""

from __future__ import annotations

import unittest

import health
from health import (
    InferencePoolView,
    InferenceWorkerView,
    evaluate_readiness,
)

NOW_NS = 1_000_000_000_000_000
#: Long enough that the startup grace period is over, so every failure below is
#: the failure under test and not "still starting".
UPTIME_S = 600.0


def pool(**overrides) -> InferencePoolView:
    values = {
        "mode": "multi_process",
        "state": "loading",
        "devices": (1, 3, 4, 5),
        "worker_count": 4,
        "ready_workers": 0,
        "busy_workers": 0,
        "loading_workers": 4,
        "failed_workers": 0,
        "minimum_ready_workers": 4,
        "latest_only": True,
        "pending_capacity": 1,
        "pending_task_seq": None,
        "pending_count": 0,
    }
    values.update(overrides)
    return InferencePoolView(**values)


def snapshot(**overrides):
    values = {
        "phase": "running",
        "started_at_monotonic": NOW_NS / 1e9 - UPTIME_S,
        "last_window_ok_at": NOW_NS / 1e9 - 1.0,
        "last_pose_arrival_us": NOW_NS // 1000 - 10_000,
        "pose_stale_ms": 500.0,
        "pose_mode": "udp",
        "camera_stale_ms": 500.0,
        "model_loaded": False,
    }
    values.update(overrides)
    return health.advance(health.EMPTY_SNAPSHOT, **values)


def verdict(**overrides):
    return evaluate_readiness(
        snapshot(**overrides), now_monotonic_ns=NOW_NS, pipeline_alive=True
    )


def check(result, name):
    return next(c for c in result.checks if c.name == name)


class InlineIsUnchangedTests(unittest.TestCase):
    """No pool means every pool rule is vacuously satisfied."""

    def test_an_inline_process_is_judged_by_its_own_runner(self) -> None:
        result = verdict(model_loaded=True)
        self.assertTrue(result.ready, result.failing)
        self.assertIn("runner=inline", check(result, "model_loaded").detail)

    def test_an_inline_process_with_an_unloaded_model_still_fails(self) -> None:
        """The old verdict, on the old evidence - that is the rollback guarantee."""
        result = verdict(model_loaded=False)
        self.assertFalse(result.ready)
        self.assertIn("model_loaded", result.failing)
        self.assertEqual(check(result, "model_loaded").reason, "model_not_loaded")

    def test_the_check_list_is_exactly_the_eight_it_always_was(self) -> None:
        """``/ready`` consumers read these by name; a pool must not add one."""
        names = [c.name for c in verdict(model_loaded=True).checks]
        self.assertEqual(
            names,
            [
                "not_stopping",
                "pipeline_alive",
                "pipeline_started",
                "startup_complete",
                "camera_window_fresh",
                "pose_fresh",
                "model_loaded",
                "time_sync_configured",
            ],
        )

    def test_the_check_list_does_not_grow_when_a_pool_exists(self) -> None:
        names = [c.name for c in verdict(inference_pool=pool(ready_workers=4)).checks]
        self.assertEqual(len(names), 8)
        self.assertNotIn("inference_workers_ready", names)


class PoolReadinessTests(unittest.TestCase):
    def test_a_loading_pool_is_not_ready(self) -> None:
        result = verdict(inference_pool=pool())
        self.assertFalse(result.ready)
        self.assertIn("model_loaded", result.failing)
        self.assertEqual(
            check(result, "model_loaded").reason, "inference_workers_not_ready"
        )

    def test_the_parents_unloaded_runner_does_not_fail_a_ready_pool(self) -> None:
        """The bug this file exists for: workers are up, the parent has no model."""
        result = verdict(model_loaded=False, inference_pool=pool(ready_workers=4, loading_workers=0))
        self.assertTrue(result.ready, result.failing)
        self.assertTrue(check(result, "model_loaded").ok)

    def test_the_configured_minimum_is_what_gates_it(self) -> None:
        """The same pool, three of four ready: serving at a minimum of three,
        refused at a minimum of four.  Nothing else may differ."""
        self.assertTrue(
            verdict(
                inference_pool=pool(ready_workers=3, loading_workers=1,
                                    minimum_ready_workers=3)
            ).ready
        )
        self.assertFalse(
            verdict(
                inference_pool=pool(ready_workers=3, loading_workers=1,
                                    minimum_ready_workers=4)
            ).ready
        )

    def test_the_detail_names_the_shortfall(self) -> None:
        detail = check(
            verdict(inference_pool=pool(ready_workers=2, failed_workers=1)), "model_loaded"
        ).detail
        self.assertIn("ready=2/4", detail)
        self.assertIn("minimum=4", detail)
        self.assertIn("failed=1", detail)

    def test_a_lost_worker_is_a_warning_while_the_pool_still_serves(self) -> None:
        """Degraded is not dead.  A 503 here would take a driving process out of
        rotation for a fault it was configured to survive."""
        result = verdict(
            inference_pool=pool(ready_workers=3, failed_workers=1,
                                minimum_ready_workers=1)
        )
        self.assertTrue(result.ready, result.failing)
        self.assertIn("inference_workers_failed:1", result.warnings)

    def test_a_pool_error_is_a_warning(self) -> None:
        result = verdict(inference_pool=pool(ready_workers=4, error="boom"))
        self.assertIn("inference_pool_error:boom", result.warnings)

    def test_a_stopped_pool_still_fails_on_its_phase_first(self) -> None:
        """Severity order: shutting down beats any worker count."""
        result = verdict(phase="stopping", inference_pool=pool(ready_workers=4))
        self.assertEqual(result.reason, "shutting_down")


class ViewShapeTests(unittest.TestCase):
    def test_a_worker_view_reports_every_field_the_endpoints_promise(self) -> None:
        worker = InferenceWorkerView(
            worker_id=2,
            device=4,
            state="busy",
            task_seq=17,
            infer_count=9,
            last_infer_ms=1512.0,
            avg_infer_ms=1500.0,
            p50_infer_ms=1495.0,
            p95_infer_ms=1600.0,
            infer_samples=9,
            allocated_bytes=1 << 30,
            reserved_bytes=2 << 30,
            peak_allocated_bytes=3 << 30,
        )
        self.assertEqual(
            sorted(worker.as_dict()),
            sorted(
                [
                    "worker_id", "device", "state", "task_seq", "infer_count",
                    "last_infer_ms", "avg_infer_ms", "p50_infer_ms", "p95_infer_ms",
                    "infer_samples", "allocated_bytes", "reserved_bytes",
                    "peak_allocated_bytes", "error", "endpoint",
                ]
            ),
        )

    def test_the_pool_view_carries_the_worker_roster(self) -> None:
        view = pool(workers=(InferenceWorkerView(worker_id=0, device=1, state="ready"),))
        body = view.as_dict()
        self.assertEqual(body["worker_count"], 4)
        self.assertEqual(body["devices"], [1, 3, 4, 5])
        self.assertEqual(body["workers"][0]["device"], 1)
        self.assertEqual(body["stop_report"], {})

    def test_the_stop_report_is_copied_not_referenced(self) -> None:
        """The one mutable field in a frozen view; a reader must not be able to
        scribble on the scheduler's report through it."""
        report = {"workers": 4, "killed": []}
        view = pool(stop_report=report)
        body = view.as_dict()
        body["stop_report"]["workers"] = 999
        self.assertEqual(view.stop_report["workers"], 4)


if __name__ == "__main__":
    unittest.main()
