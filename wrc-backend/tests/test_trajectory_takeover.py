import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trajectory.takeover import (  # noqa: E402
    DeviationMetrics,
    HistoryPoseReferenceProvider,
    ModelFrameOrigin,
    PlannedRoute,
    PlannedRouteCache,
    PlannedRoutePoint,
    PlannedRouteReferenceProvider,
    ReferenceTrajectoryRequest,
    ReferenceTrajectorySelector,
    TakeoverDecisionEngine,
    TrajectoryCompensator,
    TrajectoryDeviationCalculator,
)


class ReferenceProviderTests(unittest.TestCase):
    @staticmethod
    def request():
        return ReferenceTrajectoryRequest(
            history_xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            history_timestamps_us=np.array([0, 1_000_000, 2_000_000]),
            target_timestamps_us=np.array([3_000_000, 4_000_000]),
        )

    def test_history_provider_extrapolates_straight_motion(self):
        provider = HistoryPoseReferenceProvider()
        result = provider.generate(self.request())
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.xy, [[3.0, 0.0], [4.0, 0.0]])

    def test_planned_route_provider_is_reserved_and_non_blocking(self):
        cache = PlannedRouteCache()
        provider = PlannedRouteReferenceProvider(cache)
        self.assertIsNone(provider.generate(self.request()))
        self.assertEqual(provider.last_unavailable_reason, "planned_route_not_received")

    def test_route_cache_is_provider_input_and_selector_never_falls_back(self):
        cache = PlannedRouteCache()
        route = PlannedRoute(
            points=(PlannedRoutePoint((114.0, 30.0)),),
            coordinate_frame="adapter_defined",
            route_id="test-route",
        )
        version = cache.update(route)
        cached, cached_version = cache.snapshot()
        self.assertEqual(version, cached_version)
        self.assertEqual(cached.route_id, "test-route")

        planned = PlannedRouteReferenceProvider(cache)
        selector = ReferenceTrajectorySelector(planned)
        selection = selector.generate(self.request())
        self.assertIsNone(selection.trajectory)
        self.assertEqual(selection.fallback_reason, "planned_route_unsupported_coordinates")

    def test_real_route_conversion_and_matching(self):
        cache = PlannedRouteCache()
        cache.update(PlannedRoute(tuple(PlannedRoutePoint((113, 28 + y / 110540))
            for y in (0, 10, 30)), "geographic_lon_lat_degrees"))
        request = ReferenceTrajectoryRequest(np.empty((0, 2)), np.empty(0),
            np.array([2_000_000, 3_000_000, 4_000_000]),
            ModelFrameOrigin(28, 113, 0, 0, 0), np.array([5., 0.]), 2_000_000, 2.)
        provider = PlannedRouteReferenceProvider(cache)
        result = provider.generate(request)
        np.testing.assert_allclose(result.xy, [[5, 0], [7, 0], [9, 0]], atol=1e-8)
        request_east = ReferenceTrajectoryRequest(np.empty((0, 2)), np.empty(0),
            request.target_timestamps_us, ModelFrameOrigin(28, 113, 0, 90, 0),
            np.array([0., 5.]), 2_000_000, 2.)
        np.testing.assert_allclose(provider.generate(request_east).xy,
            [[0, 5], [0, 7], [0, 9]], atol=1e-8)

    def test_missing_completed_stale_short_and_nonfinite_route_reject(self):
        from dataclasses import replace
        cache = PlannedRouteCache()
        provider = PlannedRouteReferenceProvider(cache)
        request = ReferenceTrajectoryRequest(np.empty((0, 2)), np.empty(0),
            np.array([0, 1_000_000, 2_000_000]), ModelFrameOrigin(28, 113, 0, 0, 0),
            np.array([0., 0.]), 0, 2.)
        for points in (((113, 28), (113, 28)), ((113, 28), (113, 28 + 1/110540)),
                       ((float('nan'), 28), (113, 29))):
            cache.update(PlannedRoute(tuple(PlannedRoutePoint(p) for p in points),
                                      "geographic_lon_lat_degrees"))
            self.assertIsNone(provider.generate(request))
        cache.clear()
        self.assertIsNone(provider.generate(request))
        cache.update(PlannedRoute((PlannedRoutePoint((113, 28)), PlannedRoutePoint((113, 29))),
                                  "geographic_lon_lat_degrees"))
        cache.updated_monotonic -= 31
        self.assertIsNotNone(provider.generate(request))
        cache.update(cache.snapshot()[0])
        self.assertIsNone(provider.generate(replace(request, speed_mps=0)))
        self.assertIsNone(provider.generate(replace(request, current_xy=np.array([0., 100.]))))


class CompensationTests(unittest.TestCase):
    def test_delay_crops_expired_points(self):
        compensator = TrajectoryCompensator(0.1, min_remaining_points=10)
        trajectory = np.column_stack((np.arange(64), np.zeros(64), np.zeros(64)))
        result, reason = compensator.compensate(trajectory, 1_000_000, 3_300_000)
        self.assertEqual(reason, "ok")
        self.assertEqual(result.start_index, 23)
        self.assertEqual(result.remaining_points if hasattr(result, "remaining_points") else len(result.xy), 41)
        self.assertEqual(result.xy[0, 0], 23)

    def test_too_few_remaining_points_is_invalid(self):
        compensator = TrajectoryCompensator(0.1, min_remaining_points=10)
        result, reason = compensator.compensate(np.zeros((64, 3)), 0, 5_600_000)
        self.assertIsNone(result)
        self.assertEqual(reason, "insufficient_remaining_trajectory")


class DeviationAndDecisionTests(unittest.TestCase):
    def setUp(self):
        self.engine = TakeoverDecisionEngine(0.8, 0.4, 2, 2)

    def test_straight_matching_trajectory_does_not_take_over(self):
        ref = np.column_stack((np.arange(10), np.zeros(10)))
        metrics = TrajectoryDeviationCalculator().calculate(ref.copy(), ref)
        decision = self.engine.update(metrics)
        self.assertEqual(decision.control_flag, 0)
        self.assertAlmostEqual(decision.mean_lateral_error, 0.0)

    def test_obvious_ai_detour_has_lateral_error(self):
        ref = np.column_stack((np.arange(10), np.zeros(10)))
        ai = ref.copy()
        ai[:, 1] = 1.0
        metrics = TrajectoryDeviationCalculator().calculate(ai, ref)
        self.assertAlmostEqual(metrics.mean_lateral_error, 1.0)
        self.assertAlmostEqual(metrics.max_lateral_error, 1.0)

    def test_single_abnormal_prediction_does_not_take_over(self):
        high = DeviationMetrics(1.0, 1.2)
        self.assertEqual(self.engine.update(high).control_flag, 0)
        self.assertEqual(self.engine.takeover_count, 1)
        low = DeviationMetrics(0.1, 0.2)
        self.assertEqual(self.engine.update(low).control_flag, 0)
        self.assertEqual(self.engine.takeover_count, 0)

    def test_two_deviations_take_over_and_two_recoveries_release(self):
        high = DeviationMetrics(1.0, 1.2)
        low = DeviationMetrics(0.1, 0.2)
        self.assertEqual(self.engine.update(high).control_flag, 0)
        self.assertEqual(self.engine.update(high).control_flag, 1)
        self.assertEqual(self.engine.update(low).control_flag, 1)
        self.assertEqual(self.engine.update(low).control_flag, 0)

    def test_invalid_or_stale_input_cannot_start_takeover(self):
        self.engine.update(DeviationMetrics(1.0, 1.2))
        decision = self.engine.invalid("camera_stale")
        self.assertEqual(decision.control_flag, 0)
        self.assertEqual(decision.takeover_count, 0)
        self.assertEqual(decision.decision_reason, "camera_stale")


if __name__ == "__main__":
    unittest.main()
