"""Exercise the real takeover decision layer without loading GPU/camera runtimes.

The layer lives in ``trajectory/takeover.py`` (the decision PRIMITIVES live in
``trajectory.takeover`` - see that module's docstring).  These tests construct
it for real, with a pinned clock, rather than AST-extracting a method out of
``app.py`` and executing it against a fake ``self``.  That makes them strict:
they exercise the shipped deviation path.  Camera health, a stale pose and a
missing route leave the control flag and the confirmation counters alone.
"""
import math
import pathlib
import time
import types
import unittest
import numpy as np
from trajectory.takeover import *
# Real import instead of AST-extracting these two from app.py: they now live
# in trajectory.coordinates, and importing the actual function objects is both
# simpler and stricter (it exercises the shipped transform, not a copy).
from trajectory.coordinates import enu_from_ll, enu_to_ego  # noqa: F401
from trajectory.takeover import TakeoverDecisionLayer


class BackendRouteControlTests(unittest.TestCase):
    def setUp(self):
        self.cache = PlannedRouteCache()
        self.cache.update(PlannedRoute(tuple(PlannedRoutePoint((113., 28. + y/110540.))
                         for y in (0, 10, 30)), "geographic_lon_lat_degrees"))
        # Pinned clock: the layer advances the pose clock by its monotonic
        # arrival age, so a real clock would make these assertions flaky.
        self.now_ns = time.monotonic_ns()
        self.pose = types.SimpleNamespace(latitude=28., longitude=113., altitude=0.,
                     timestamp_us=2_000_000, arrival_us=self.now_ns // 1000, speed_mps=2.)
        self.camera_stats = [dict(health="HEALTHY", frame_age_ms=0)]
        self.layer = TakeoverDecisionLayer(
            takeover_engine=TakeoverDecisionEngine(.8, .4),
            trajectory_compensator=TrajectoryCompensator(.1, 10),
            deviation_calculator=TrajectoryDeviationCalculator(),
            reference_provider=ReferenceTrajectorySelector(
                PlannedRouteReferenceProvider(self.cache)),
            planned_route_cache=self.cache,
            expected_camera_count=1,
            camera_stale_ms=1000,
            pose_stale_ms=800,
            now_us_fn=lambda: self.now_ns,
        )
        self.result = types.SimpleNamespace(positions=np.column_stack((np.arange(64)*.2, np.zeros(64))),
            t0_us=2_000_000, ref_pose=types.SimpleNamespace(latitude=28., longitude=113., altitude=0.,
            heading_deg=0., timestamp_us=2_000_000))

    def decide(self, result=None):
        result = self.result if result is None else result
        return self.layer.decide(
            positions=result.positions,
            t0_us=result.t0_us,
            ref_pose=result.ref_pose,
            camera_stats_fn=lambda: self.camera_stats,
            latest_pose_fn=lambda: self.pose,
        )

    def test_route_detour_confirmation_and_release(self):
        self.assertEqual(self.decide(self.result).control_flag, 0)
        self.result.positions[:, 1] = 1.2
        self.assertEqual(self.decide(self.result).control_flag, 0)
        self.assertEqual(self.decide(self.result).control_flag, 1)
        self.result.positions[:, 1] = 0
        self.assertEqual(self.decide(self.result).control_flag, 1)
        self.assertEqual(self.decide(self.result).control_flag, 0)

    def test_gates_do_not_break_confirmation(self):
        route, _version = self.cache.snapshot()
        self.result.positions[:, 1] = 1.2
        self.assertEqual(self.decide(self.result).takeover_count, 1)
        self.assertEqual(self.layer.takeover_engine.control_flag, 0)

        self.pose.arrival_us -= 1_000_000
        decision = self.decide(self.result)
        self.assertEqual(decision.decision_reason, "pose_stale")
        self.assertEqual(decision.control_flag, 0)
        self.assertEqual(decision.takeover_count, 1)

        self.pose.arrival_us = self.now_ns // 1000
        self.cache.clear()
        decision = self.decide(self.result)
        self.assertEqual(decision.control_flag, 0)
        self.assertEqual(decision.takeover_count, 1)
        self.assertEqual(decision.reference_fallback_reason, "planned_route_not_received")

        # A stale camera is not a control input. The pending confirmation
        # survives, and this second over-threshold sample takes over.
        self.cache.update(route)
        self.camera_stats[0]["health"] = "STALE"
        decision = self.decide(self.result)
        self.assertEqual(decision.control_flag, 1)
        self.assertEqual(decision.decision_reason, "takeover_confirmed")

    def test_model_stop_takes_over_on_the_first_cycle(self):
        self.result.positions = np.zeros((64, 2))
        decision = self.decide(self.result)
        self.assertEqual(decision.control_flag, 1)
        self.assertEqual(decision.decision_reason, "model_stop_takeover")
        self.assertEqual(decision.takeover_count, 0)
        self.assertEqual(decision.release_count, 0)
        again = self.decide(self.result)
        self.assertEqual(again.control_flag, 1)
        self.assertEqual(again.decision_reason, "model_stop_takeover")
        self.assertEqual(again.release_count, 0)

    def test_missing_pose_keeps_flag_and_counts(self):
        engine = self.layer.takeover_engine
        engine.takeover_count = 1
        engine.release_count = 1
        decision = self.layer.decide(
            positions=self.result.positions,
            t0_us=self.result.t0_us,
            ref_pose=self.result.ref_pose,
            camera_stats_fn=lambda: [{"health": "STALE", "frame_age_ms": 10_000}],
            latest_pose_fn=lambda: None,
        )
        self.assertNotEqual(decision.decision_reason, "camera_status_unavailable")
        self.assertEqual(decision.control_flag, 0)
        self.assertEqual(decision.takeover_count, 1)
        self.assertEqual(decision.release_count, 1)
        self.assertEqual(engine.takeover_count, 1)
        self.assertEqual(engine.release_count, 1)

    def test_expired_future_and_nan_reject(self):
        self.result.t0_us -= 6_000_000
        self.assertEqual(self.decide(self.result).control_flag, 0)
        self.result.t0_us = 3_000_000
        self.assertEqual(self.decide(self.result).decision_reason, "invalid_model_input_timestamp")
        self.result.t0_us = 2_000_000
        self.result.positions[0, 0] = float("nan")
        self.assertEqual(self.decide(self.result).decision_reason, "non_finite_ai_trajectory")

    def test_actual_send_uses_decision_not_model_or_manual_flag(self):
        # Stage 2 moved the loop into pipeline.py.  Both halves are asserted:
        # the new home owns the assignment, and the old one is clean, so a
        # second send path cannot hide in app.py.
        root = pathlib.Path(__file__).parents[1]
        pipeline = (root / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("control = decision.control_flag", pipeline)
        self.assertNotIn("control = resolve_control_flag(", pipeline)
        entry = (root / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("control = decision.control_flag", entry)
        self.assertNotIn("control = resolve_control_flag(", entry)


class TakeoverSafetyMatrixTests(unittest.TestCase):
    """Pins the behaviour of invalid input, which is a SAFETY POLICY, not a bug.

    The current behaviour is **fail-hold**: :meth:`TakeoverDecisionEngine.invalid`
    returns ``self.control_flag`` unchanged.  Once a takeover has been
    confirmed, no invalid input - stale camera, stale pose, unreachable
    planned route, non-finite trajectory - will release it.  Both confirmation
    counters are reset, so an interrupted release must start over.

    This is deliberate and is NOT changed by the refactor: whether a failed
    planned route should hold the takeover or release it is a vehicle-level
    decision, so the refactor only records it.  See also
    TakeoverAndSendPathInteractionTests, which shows that a held takeover can
    be accompanied by no waypoint datagram at all.
    """

    def setUp(self) -> None:
        self.engine = TakeoverDecisionEngine(0.8, 0.4, 2, 2)

    @staticmethod
    def _metrics(mean: float) -> DeviationMetrics:
        return DeviationMetrics(mean_lateral_error=mean, max_lateral_error=mean)

    def _take_over(self) -> None:
        self.engine.update(self._metrics(1.2))
        decision = self.engine.update(self._metrics(1.2))
        self.assertEqual(decision.control_flag, 1)
        self.assertEqual(decision.decision_reason, "takeover_confirmed")

    def test_invalid_on_clear_state_stays_clear(self):
        decision = self.engine.invalid("camera_stale")
        self.assertEqual(decision.control_flag, 0)
        self.assertEqual(decision.decision_reason, "camera_stale")

    def test_invalid_cannot_start_a_takeover(self):
        # However many invalid frames arrive, they never accumulate.
        for _ in range(10):
            self.assertEqual(self.engine.invalid("pose_stale").control_flag, 0)

    def test_invalid_holds_an_active_takeover(self):
        self._take_over()
        for reason in (
            "camera_status_unavailable", "camera_stale", "pose_unavailable",
            "pose_arrival_timestamp_unavailable", "pose_arrival_timestamp_in_future",
            "pose_stale", "reference_coordinate_conversion_failed",
            "reference_trajectory_unavailable", "planned_route_not_received",
            "trajectory_deviation_calculation_failed",
            "planned_route_changed_during_decision",
        ):
            with self.subTest(reason=reason):
                decision = self.engine.invalid(reason)
                self.assertEqual(decision.control_flag, 1,
                                 "invalid input must never release a takeover")
                self.assertIsNone(decision.mean_lateral_error)
                self.assertEqual(decision.remaining_points, 0)

    def test_invalid_discards_an_in_progress_takeover_confirmation(self):
        self.engine.update(self._metrics(1.2))
        self.assertEqual(self.engine.takeover_count, 1)
        self.assertEqual(self.engine.invalid("camera_stale").takeover_count, 0)
        # One more over-threshold frame is therefore not enough to confirm.
        self.assertEqual(self.engine.update(self._metrics(1.2)).control_flag, 0)

    def test_invalid_discards_an_in_progress_release(self):
        self._take_over()
        self.engine.update(self._metrics(0.1))
        self.assertEqual(self.engine.release_count, 1)
        self.engine.invalid("pose_stale")
        self.assertEqual(self.engine.release_count, 0)
        # The release needed two consecutive samples, so it must start over.
        self.assertEqual(self.engine.update(self._metrics(0.1)).control_flag, 1)
        self.assertEqual(self.engine.update(self._metrics(0.1)).control_flag, 0)

    def test_hysteresis_band_holds_a_takeover_indefinitely(self):
        # Between release_threshold_m and takeover_threshold_m nothing changes.
        self._take_over()
        for _ in range(20):
            decision = self.engine.update(self._metrics(0.6))
            self.assertEqual(decision.control_flag, 1)
            self.assertEqual(decision.decision_reason, "takeover_held")

    def test_invalid_thresholds_are_rejected(self):
        for takeover, release in ((0.4, 0.8), (0.8, 0.8), (0.8, -0.1),
                                  (float("nan"), 0.4), (float("inf"), 0.4)):
            with self.subTest(takeover=takeover, release=release):
                with self.assertRaises(ValueError):
                    TakeoverDecisionEngine(takeover, release)

    def test_confirmation_counts_are_clamped_to_at_least_one(self):
        # 0 or negative would otherwise make a single frame confirm anything.
        engine = TakeoverDecisionEngine(0.8, 0.4, takeover_confirmations=0,
                                        release_confirmations=-5)
        self.assertEqual(engine.takeover_confirmations, 1)
        self.assertEqual(engine.release_confirmations, 1)
        self.assertEqual(engine.update(self._metrics(1.2)).control_flag, 1)


if __name__ == "__main__":
    unittest.main()
