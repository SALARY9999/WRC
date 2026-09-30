"""Waypoint expiry trimming and UDP count/length regression tests.

Nothing tested here needs a camera.  The trim, the decision engine and the UDP
wire format are pure logic, so this file imports the modules that actually own
that logic - ``trajectory.coordinates``, ``trajectory.postprocess``,
``vehicle.models``, ``config.loader``, ``waypoint.sender``,
``trajectory.takeover`` - and parses ``pipeline.py`` as *text* instead of
reaching all of it through ``app``.

That is what keeps the file honest.  ``app`` imports ``camera.sync``, which
imports ``cv2``, which is absent from lightweight developer machines; reaching
the trim through ``app`` therefore required installing
``sys.modules["cv2"] = MagicMock()`` at *collection* time, and that stub then
leaked into every other module in the same interpreter - turning a later
``import camera.sync``'s truthful "cv2 is not installed" into a baffling
``AttributeError: module 'cv2' has no attribute 'resize'``.

The one thing here that genuinely is ``app``'s own surface, ``app.status()``,
is a FastAPI route handler and cannot be reached without importing ``app``.  It
is checked in a child interpreter instead (see ``_status_keys``), so a stub
used to get there dies with that child and cannot reach this process.
"""

import ast
import json
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from _source_tree import pipeline_result_tree
from config.loader import AppConfig
from trajectory.coordinates import local_xy_to_latlon
from trajectory.postprocess import (
    hold_waypoint_at_latest_pose,
    trim_waypoints_to_latest_pose,
)
from trajectory.takeover import DeviationMetrics, TakeoverDecisionEngine
from vehicle.models import GlobalPose, PoseSample, vehicle_azimuth_for_compass
from waypoint.sender import WaypointSender

# ``Settings`` is a plain alias for ``AppConfig``, and these tests only ever
# construct one - so the alias is kept, minus the detour through ``app``.
Settings = AppConfig

BACKEND_ROOT = Path(__file__).resolve().parents[1]


_STATUS_CHILD = """
import json, sys, types
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    # Only this child interpreter ever sees the stub; it exits with it.
    sys.modules["cv2"] = types.ModuleType("cv2")
import app
print(json.dumps(sorted(app.status())))
"""


def _status_keys() -> set[str]:
    """The keys ``/status`` answers with when no runner exists.

    Run in a child interpreter because ``app.status()`` is defined next to
    ``app``'s FastAPI routes and this file has no other reason to import
    ``app``.  A failure here is reported with the child's stderr rather than
    being allowed to masquerade as an assertion failure.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _STATUS_CHILD],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"importing app to call status() failed in a child interpreter "
            f"(exit {completed.returncode}):\n{completed.stderr}"
        )
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


def pose_at_model_xy(
    x_m: float,
    y_m: float,
    timestamp_us: int = 2_000_000,
    *,
    heading_deg: float = 0.0,
    v_north: float = 0.0,
    v_east: float = 0.0,
    arrival_us: int | None = None,
):
    """A pose whose model-frame position is ``(x_m, y_m)``.

    The default heading is north, model +x for the reference pose these
    tests use.  The trim drops a leading point behind this heading, so the
    old zero azimuth (compass west) would aim the nose off the straight
    fixture and delete every point on it.
    """
    lat, lon = local_xy_to_latlon(
        np.asarray([x_m]), np.asarray([y_m]), 28.0, 113.0, 0.0
    )
    return GlobalPose(
        timestamp_us, float(lat[0]), float(lon[0]), 0.0,
        v_north, v_east, 0.0, 0.0, 0.0,
        vehicle_azimuth_for_compass(heading_deg),
        arrival_us=arrival_us,
    )


class CapturingSocket:
    def __init__(self):
        self.sent = []

    def sendto(self, packet, address):
        self.sent.append((packet, address))


class WaypointTrimmingTests(unittest.TestCase):
    def setUp(self):
        self.ref = PoseSample(1_000_000, 28.0, 113.0, 0.0, 0.0)
        self.trajectory = np.column_stack(
            (np.arange(64, dtype=float), np.zeros(64), np.zeros(64))
        )

    def test_vehicle_before_p0_keeps_all_waypoints(self):
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(-0.2, 0.1), self.ref
        )
        self.assertEqual(trimmed.removed_waypoint_count, 0)
        self.assertEqual(len(trimmed.positions), 64)

    def test_projection_on_segment_10_to_11_removes_through_p10(self):
        # No receive wait is supplied, so nothing is extrapolated.  The
        # vehicle at 10.4 m stays on the 10→11 segment and the first sent
        # point is still 11 m.
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(10.4, 0.3), self.ref
        )
        self.assertEqual(trimmed.matched_segment_index, 10)
        self.assertEqual(trimmed.removed_waypoint_count, 11)
        np.testing.assert_array_equal(trimmed.positions, self.trajectory[11:])

    def test_straight_line_at_10_4_is_unchanged_when_extrapolation_does_not_apply(self):
        arrival_us = 2_000_000
        pose = pose_at_model_xy(
            10.4, 0.3, v_north=4.0, v_east=1.0, arrival_us=arrival_us,
        )
        held = trim_waypoints_to_latest_pose(
            self.trajectory, pose, self.ref, now_us=arrival_us,
        )
        self.assertEqual(held.matched_segment_index, 10)
        self.assertEqual(held.removed_waypoint_count, 11)
        np.testing.assert_array_equal(held.positions, self.trajectory[11:])

        pose.arrival_us = None
        missing_arrival = trim_waypoints_to_latest_pose(
            self.trajectory, pose, self.ref, now_us=arrival_us + 100_000,
        )
        self.assertEqual(missing_arrival.matched_segment_index, 10)
        np.testing.assert_array_equal(
            missing_arrival.positions, self.trajectory[11:],
        )

        # Past the cap the old velocity is not applied, even in part.
        # 0.5 s at 4 m/s would be 2 m and would leave 11 m behind.
        pose.arrival_us = arrival_us
        stale = trim_waypoints_to_latest_pose(
            self.trajectory, pose, self.ref, now_us=arrival_us + 500_000,
        )
        self.assertEqual(stale.matched_segment_index, 10)
        np.testing.assert_array_equal(stale.positions, self.trajectory[11:])

    def test_missing_pose_preserves_original_trajectory(self):
        trimmed = trim_waypoints_to_latest_pose(self.trajectory, None, self.ref)
        self.assertEqual(trimmed.fallback_reason, "latest_pose_unavailable")
        self.assertFalse(trimmed.applied)
        self.assertFalse(trimmed.expired)
        np.testing.assert_array_equal(trimmed.positions, self.trajectory)

    def test_coordinate_conversion_failure_safely_falls_back(self):
        # The crop moved to trajectory/postprocess.py in Stage 2, so the patch
        # target moved with it: the module resolves poses_to_model_frame_xy
        # from its OWN globals, and patching app.poses_to_model_frame_xy no
        # longer reaches it.  This is the one test-visible consequence of the
        # move - production code never patches this name.
        with patch(
            "trajectory.postprocess.poses_to_model_frame_xy",
            side_effect=ValueError("conversion failed"),
        ):
            trimmed = trim_waypoints_to_latest_pose(
                self.trajectory, pose_at_model_xy(3.0, 0.0), self.ref
            )
        self.assertEqual(trimmed.fallback_reason, "conversion failed")
        np.testing.assert_array_equal(trimmed.positions, self.trajectory)

    def test_projection_beyond_last_point_has_no_remaining_waypoints(self):
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(70.0, 0.0), self.ref
        )
        self.assertEqual(trimmed.removed_waypoint_count, 64)
        self.assertEqual(len(trimmed.positions), 0)

    def test_udp_count_and_packet_length_follow_trimmed_points(self):
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(10.4, 0.0), self.ref
        )
        settings = Settings(
            magic=0x12345678,
            vehicle_id=7,
            allow_unconfigured_udp=True,
        )
        sender = WaypointSender(settings)
        sender.sock.close()
        sender.sock = CapturingSocket()
        lat = np.arange(len(trimmed.positions), dtype=float)
        lon = lat + 100.0
        sent_count = sender.send(1, lat, lon)
        packet = sender.sock.sent[0][0]
        header = struct.unpack(settings.byte_order + "IBBIHH", packet[:14])
        self.assertEqual(sent_count, len(trimmed.positions))
        self.assertEqual(header[-1], len(trimmed.positions))
        self.assertEqual(len(packet), 14 + 16 * len(trimmed.positions))


class LeadingPointAheadOfNoseTests(unittest.TestCase):
    """The crop must not hand the controller a first point behind the nose.

    The polyline crop still deletes the matched segment's start.  Two things
    are added around it: the pose is pushed by its packet velocity over the
    receive wait, and a leading point that is still behind the body heading
    is then dropped.  A model stop does not come through this path.
    """

    def setUp(self):
        self.ref = PoseSample(1_000_000, 28.0, 113.0, 0.0, 0.0)

    def test_velocity_over_the_receive_wait_advances_the_projection(self):
        # 0.5 m spacing.  At 10.4 m the matched segment is 10.0→10.5, so the
        # first sent point would be 10.5 m.  0.1 s at 3 m/s north is +0.3 m
        # in this frame, about 10.7 m, on 10.5→11.0.
        xs = np.arange(0.0, 24.0, 0.5)
        trajectory = np.column_stack((xs, np.zeros_like(xs), np.zeros_like(xs)))
        arrival_us = 5_000_000
        pose = pose_at_model_xy(
            10.4, 0.0, v_north=3.0, v_east=0.0, arrival_us=arrival_us,
        )
        trimmed = trim_waypoints_to_latest_pose(
            trajectory, pose, self.ref, now_us=arrival_us + 100_000,
        )
        self.assertGreater(len(trimmed.positions), 0)
        self.assertAlmostEqual(float(trimmed.positions[0, 0]), 11.0, places=4)
        self.assertNotAlmostEqual(float(trimmed.positions[0, 0]), 10.5, places=3)
        self.assertTrue(trimmed.applied)
        self.assertIsNone(trimmed.fallback_reason)

    def test_a_wait_past_the_cap_does_not_push_with_the_old_speed(self):
        xs = np.arange(0.0, 24.0, 0.5)
        trajectory = np.column_stack((xs, np.zeros_like(xs), np.zeros_like(xs)))
        arrival_us = 5_000_000
        pose = pose_at_model_xy(
            10.4, 0.0, v_north=3.0, arrival_us=arrival_us,
        )
        # 0.5 s * 3 m/s would land near 11.9 m and drop 10.5.  Clamping the
        # wait to 0.2 s would land near 11.0 m and drop it too.  Neither
        # happens: the first sent point stays 10.5 m.
        trimmed = trim_waypoints_to_latest_pose(
            trajectory, pose, self.ref, now_us=arrival_us + 500_000,
        )
        self.assertAlmostEqual(float(trimmed.positions[0, 0]), 10.5, places=4)

    def test_corner_behind_the_nose_is_not_the_first_sent_point(self):
        # Incoming leg along +x, corner at (10, 0), then a left turn to +y.
        # The vehicle is still closest to the incoming leg, so the polyline
        # crop leaves the corner as the first point.
        trajectory = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 8.0, 0.0],
            [10.0, 16.0, 0.0],
        ])
        still_on_the_leg = pose_at_model_xy(8.0, 0.4, heading_deg=0.0)
        along_the_leg = trim_waypoints_to_latest_pose(
            trajectory, still_on_the_leg, self.ref,
        )
        np.testing.assert_allclose(
            along_the_leg.positions[0, :2], [10.0, 0.0], atol=1e-3,
        )

        # The body has already yawed onto the exit.  The corner is behind
        # that nose, so it is not what gets sent first.
        turned = pose_at_model_xy(8.0, 0.4, heading_deg=270.0)
        trimmed = trim_waypoints_to_latest_pose(trajectory, turned, self.ref)
        self.assertGreater(len(trimmed.positions), 0)
        self.assertFalse(
            np.allclose(trimmed.positions[0, :2], [10.0, 0.0], atol=1e-3)
        )
        np.testing.assert_allclose(
            trimmed.positions[0, :2], [10.0, 8.0], atol=1e-3,
        )

    def test_a_point_a_few_centimetres_behind_the_nose_is_kept(self):
        # Just past the corner, outside the exit leg, so both segments clamp
        # to the corner and the polyline crop keeps it.  It is 3 cm behind
        # the nose and stays; it is on the vehicle, not a reverse target.
        trajectory = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 20.0, 0.0],
        ])
        pose = pose_at_model_xy(10.03, -0.02, heading_deg=0.0)
        trimmed = trim_waypoints_to_latest_pose(trajectory, pose, self.ref)
        np.testing.assert_allclose(
            trimmed.positions[0, :2], [10.0, 0.0], atol=1e-3,
        )

    def test_a_curve_that_folds_back_keeps_the_returning_points(self):
        # The vehicle is on the outgoing leg.  Later points come back behind
        # the nose, well off the vehicle, and must still be sent.
        trajectory = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 1.0, 0.0],
            [8.0, 5.0, 0.0],
            [5.0, 9.0, 0.0],
            [1.0, 9.0, 0.0],
            [0.0, 6.0, 0.0],
        ])
        pose = pose_at_model_xy(2.0, 0.1, heading_deg=0.0)
        trimmed = trim_waypoints_to_latest_pose(trajectory, pose, self.ref)
        self.assertGreater(float(trimmed.positions[0, 0]), 2.0)
        folded = trimmed.positions[:, :2]
        self.assertTrue(
            np.any(np.isclose(folded[:, 0], 0.0) & np.isclose(folded[:, 1], 6.0))
        )
        self.assertFalse(
            np.allclose(trimmed.positions[0, :2], [0.0, 6.0], atol=1e-3)
        )

    def test_a_prefix_entirely_behind_the_nose_expires_the_trajectory(self):
        # Polyline crop keeps the points ahead along +x.  The body is facing
        # back down the leg, so every one of them is behind the nose.
        trajectory = np.column_stack(
            (np.arange(8, dtype=float), np.zeros(8), np.zeros(8))
        )
        pose = pose_at_model_xy(3.4, 0.0, heading_deg=180.0)
        trimmed = trim_waypoints_to_latest_pose(trajectory, pose, self.ref)
        self.assertEqual(len(trimmed.positions), 0)
        self.assertTrue(trimmed.applied)
        self.assertTrue(trimmed.expired)
        self.assertIsNone(trimmed.fallback_reason)
        self.assertEqual(trimmed.removed_waypoint_count, len(trajectory))

    def test_pipeline_passes_the_monotonic_clock_into_the_trim(self):
        calls = [
            node for node in ast.walk(pipeline_result_tree())
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "trim_waypoints_to_latest_pose"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn("monotonic_ns", ast.unparse(calls[0]))


class EmptyTrimReleaseFrameTests(unittest.TestCase):
    """``waypoint_count == 0`` implies ``control_flag == 0``, and the frame IS sent.

    Confirmed with the vehicle owner on 2026-09-17.  An expired trajectory
    leaves nothing to execute, so there is nothing to take over *for*: the
    release frame carries ``control=0``.  Sending it is the point - "no packet"
    and "not taking over" are indistinguishable on the wire, so the previous
    behaviour (log, then ``continue`` without sending) left the vehicle holding
    the last ``control=1`` it had received.

    These assertions are structural because ``PipelineRunner`` needs a camera
    window and a model; the wire form of the frame itself is pinned separately,
    by driving the real sender.  They read ``_handle_result``, which is where
    the post-model code lives for both the inline and the pooled path - see
    ``_source_tree.pipeline_result_tree``.
    """

    def setUp(self) -> None:
        self.ref = PoseSample(1_000_000, 28.0, 113.0, 0.0, 0.0)
        self.trajectory = np.column_stack(
            (np.arange(64, dtype=float), np.zeros(64), np.zeros(64))
        )
        self.tree = pipeline_result_tree()
        self.guard = self._find_empty_trim_guard()

    def _find_empty_trim_guard(self) -> ast.If:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.If):
                continue
            call = node.test.left if isinstance(node.test, ast.Compare) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "len"
                and len(call.args) == 1
                and ast.unparse(call.args[0]) == "trim.positions"
            ):
                return node
        self.fail("the empty-trim guard is gone from PipelineRunner._handle_result")

    def _enclosing_body(self) -> list:
        for node in ast.walk(self.tree):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(node, field, None)
                if isinstance(body, list) and self.guard in body:
                    return body
        self.fail("the empty-trim guard has no enclosing block")

    def test_the_guard_forces_control_to_zero(self):
        assigned = {
            node.targets[0].id: node.value
            for node in self.guard.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertIn(
            "control", assigned,
            "an empty trim must override the control flag to 0",
        )
        self.assertIsInstance(assigned["control"], ast.Constant)
        self.assertEqual(assigned["control"].value, 0)

    def test_the_guard_no_longer_skips_the_send(self):
        skipped = [n for n in ast.walk(self.guard) if isinstance(n, ast.Continue)]
        self.assertEqual(
            skipped, [],
            "an empty trim must fall through to the UDP send, not skip it: "
            "staying silent leaves the vehicle holding its last control=1",
        )

    def test_the_udp_send_follows_the_guard_in_the_same_block(self):
        after = self._enclosing_body()[self._enclosing_body().index(self.guard) + 1:]
        sends = [
            n for n in after
            if isinstance(n, ast.Assign) and "sender.send(" in ast.unparse(n)
        ]
        self.assertTrue(sends, "the empty-trim branch must still reach the sender")

    def test_the_trim_itself_really_can_be_empty(self):
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(500.0, 0.0), self.ref
        )
        self.assertEqual(len(trimmed.positions), 0)
        self.assertEqual(trimmed.removed_waypoint_count, len(self.trajectory))
        self.assertTrue(trimmed.applied)
        self.assertTrue(trimmed.expired)
        self.assertIsNone(trimmed.fallback_reason)

    def test_release_frame_is_count_zero_control_zero_on_the_wire(self):
        settings = Settings(
            magic=0x514D3031, vehicle_id=1, allow_unconfigured_udp=True
        )
        sender = WaypointSender(settings)
        sender.sock.close()
        sender.sock = CapturingSocket()
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(500.0, 0.0), self.ref
        )
        lat, lon = local_xy_to_latlon(
            trimmed.positions[:, 0], trimmed.positions[:, 1], 28.0, 113.0, 0.0
        )
        sent = sender.send(0, lat, lon)
        packet = sender.sock.sent[0][0]
        magic, version, control, seq, vehicle_id, count = struct.unpack(
            settings.byte_order + "IBBIHH", packet[:14]
        )
        self.assertEqual(sent, 0)
        self.assertEqual(count, 0)
        self.assertEqual(control, 0)
        self.assertEqual(magic, 0x514D3031)
        self.assertEqual(version, 1)
        self.assertEqual(vehicle_id, 1)
        self.assertEqual(len(packet), 14, "a release frame is header-only")

    def test_the_send_log_never_indexes_an_empty_endpoint_array(self):
        # `lat[0]` on the zero-length array a release frame produces is an
        # IndexError, which would abort the cycle *after* the packet went out.
        # The log call must therefore take pre-computed endpoints, not
        # subscripts, and the guard that computes them must test `n`.
        log_call = None
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first = ast.unparse(node.args[0])
            if "inference sent: control_flag=" in first:
                log_call = node
        self.assertIsNotNone(log_call, "the send log line is gone")
        indexed = [
            ast.unparse(arg)
            for arg in log_call.args
            if any(
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in ("lat", "lon")
                for sub in ast.walk(arg)
            )
        ]
        self.assertEqual(indexed, [], "the send log must not index lat/lon")

        guarded = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.IfExp)
            and isinstance(node.test, ast.Name)
            and node.test.id == "n"
            and any(
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in ("lat", "lon")
                for sub in ast.walk(node)
            )
        ]
        self.assertTrue(
            guarded,
            "lat/lon endpoints must be computed behind an `if n` guard",
        )

    def test_sent_flag_is_recorded_separately_from_the_decision_flag(self):
        # The release frame is the one case where the two differ, so both have
        # to be visible: /status for the operator, the package for replay.
        tree = pipeline_result_tree()
        assigned = {
            ast.unparse(node.targets[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
        }
        self.assertIn("self.last_sent_control_flag", assigned)
        self.assertIn("self.last_control_flag", assigned)

    def test_status_reports_both_flags(self):
        # Without a runner both are null, but the keys must exist: /status is
        # what the operator reads to tell "not taking over" from "sent a
        # release frame".
        keys = _status_keys()
        self.assertIn("last_control_flag", keys)
        self.assertIn("last_sent_control_flag", keys)


class TakeoverAndSendPathInteractionTests(unittest.TestCase):
    """The two trajectory layers are deliberately independent, not in conflict.

    ``TakeoverDecision`` and ``trim_waypoints_to_latest_pose`` answer different
    questions and are not required to consume the same trajectory:

    * the decision sums lateral deviation over the **time-cropped**
      ``compensated.xy`` to answer "is the model's prediction wrong?";
    * the send path geometrically trims the **raw** ``result.positions``
      against the latest pose to answer "does the trajectory we hand the
      vehicle start where the vehicle is?".

    They can therefore reach different conclusions in the same cycle, and that
    is intended (confirmed with the vehicle owner 2026-09-17).  If the trim
    leaves nothing to send and this is not a model stop, the frame carries
    ``control=0`` regardless of the decision flag.  A model stop is the
    exception: the wire carries ``control=1`` and one waypoint at the current
    pose, because the stop cluster is already behind the vehicle.
    """

    def setUp(self) -> None:
        self.ref = PoseSample(1_000_000, 28.0, 113.0, 0.0, 0.0)
        self.trajectory = np.column_stack(
            (np.arange(64, dtype=float), np.zeros(64), np.zeros(64))
        )

    def test_fully_expired_trajectory_leaves_nothing_to_send(self):
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(500.0, 0.0), self.ref
        )
        self.assertEqual(len(trimmed.positions), 0)
        self.assertEqual(trimmed.removed_waypoint_count, len(self.trajectory))
        self.assertTrue(trimmed.applied)
        self.assertTrue(trimmed.expired)
        self.assertIsNone(trimmed.fallback_reason)

    def test_the_decision_flag_survives_an_expired_trajectory_unchanged(self):
        # The engine is never told the send path found nothing, so a confirmed
        # takeover stays confirmed across the expiry - fail-hold is the policy.
        engine = TakeoverDecisionEngine(0.8, 0.4, 2, 2)
        engine.update(DeviationMetrics(1.2, 1.2))
        self.assertEqual(engine.update(DeviationMetrics(1.2, 1.2)).control_flag, 1)
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(500.0, 0.0), self.ref
        )
        self.assertEqual(len(trimmed.positions), 0)
        # ... and a fresh invalid frame still reports takeover held.
        self.assertEqual(engine.invalid("trajectory_expired").control_flag, 1)

    def test_the_decision_flag_does_not_reach_the_wire_on_an_expired_trim(self):
        # Held takeover + empty trim: the decision says 1, the wire must say 0.
        engine = TakeoverDecisionEngine(0.8, 0.4, 2, 2)
        engine.update(DeviationMetrics(1.2, 1.2))
        decision_flag = engine.update(DeviationMetrics(1.2, 1.2)).control_flag
        self.assertEqual(decision_flag, 1)
        trimmed = trim_waypoints_to_latest_pose(
            self.trajectory, pose_at_model_xy(500.0, 0.0), self.ref
        )
        settings = Settings(magic=0x12345678, vehicle_id=7,
                                allow_unconfigured_udp=True)
        sender = WaypointSender(settings)
        sender.sock.close()
        sender.sock = CapturingSocket()
        control = 0 if len(trimmed.positions) == 0 else decision_flag
        lat, lon = local_xy_to_latlon(
            trimmed.positions[:, 0], trimmed.positions[:, 1], 28.0, 113.0, 0.0
        )
        sender.send(control, lat, lon)
        packet = sender.sock.sent[0][0]
        self.assertEqual(len(packet), 14)
        self.assertEqual(
            struct.unpack(settings.byte_order + "IBBIHH", packet[:14]),
            (0x12345678, 1, 0, 0, 7, 0),
        )


class ModelStopHoldTests(unittest.TestCase):
    """A model stop sends the vehicle's current position, not the old cluster."""

    def setUp(self) -> None:
        self.ref = PoseSample(1_000_000, 28.0, 113.0, 0.0, 0.0)
        self.tree = pipeline_result_tree()

    def test_hold_point_is_the_current_pose_not_the_inference_origin(self):
        latest = pose_at_model_xy(8.0, -1.5)
        held = hold_waypoint_at_latest_pose(latest, self.ref)
        self.assertIsNotNone(held)
        self.assertEqual(held.shape, (1, 3))
        np.testing.assert_allclose(held[0, :2], [8.0, -1.5], atol=1e-4)
        lat, lon = local_xy_to_latlon(
            held[:, 0], held[:, 1], self.ref.latitude, self.ref.longitude,
            self.ref.heading_deg,
        )
        self.assertAlmostEqual(float(lat[0]), latest.latitude, places=6)
        self.assertAlmostEqual(float(lon[0]), latest.longitude, places=6)

    def test_a_stop_cluster_behind_the_vehicle_would_otherwise_expire(self):
        latest = pose_at_model_xy(8.0, 0.0)
        stop = np.zeros((64, 3))
        stop[:, 0] = np.linspace(0.0, 0.3, 64)
        trimmed = trim_waypoints_to_latest_pose(stop, latest, self.ref)
        self.assertEqual(len(trimmed.positions), 0)
        self.assertTrue(trimmed.expired)

    def test_missing_or_non_finite_pose_returns_no_hold_point(self):
        self.assertIsNone(hold_waypoint_at_latest_pose(None, self.ref))
        latest = pose_at_model_xy(8.0, 0.0)
        latest.latitude = float("nan")
        self.assertIsNone(hold_waypoint_at_latest_pose(latest, self.ref))

    def test_hold_replaces_the_sent_trajectory_before_the_empty_release(self):
        hold_line = None
        guard_line = None
        stop_if = None
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "hold_waypoint_at_latest_pose"
            ):
                hold_line = node.lineno
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and ast.unparse(test.left) == "len(trim.positions)"
            ):
                guard_line = node.lineno
            if (
                isinstance(test, ast.Compare)
                and ast.unparse(test.left) == "decision.decision_reason"
                and any(
                    isinstance(item, ast.Constant)
                    and item.value == "model_stop_takeover"
                    for item in test.comparators
                )
            ):
                stop_if = node
        self.assertIsNotNone(hold_line)
        self.assertIsNotNone(guard_line)
        self.assertLess(hold_line, guard_line)
        self.assertIsNotNone(stop_if)
        returns = [
            node for node in stop_if.body
            if isinstance(node, ast.If)
        ]
        silent = [
            node for node in ast.walk(ast.Module(body=returns, type_ignores=[]))
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        ]
        self.assertTrue(
            silent,
            "a model stop with no current pose must return before the send",
        )
        self.assertEqual(
            [
                node for node in ast.walk(stop_if)
                if isinstance(node, ast.Call) and "sender.send" in ast.unparse(node.func)
            ],
            [],
        )

    def test_hold_point_is_sent_with_control_one(self):
        latest = pose_at_model_xy(8.0, 0.0)
        held = hold_waypoint_at_latest_pose(latest, self.ref)
        settings = Settings(
            magic=0x514D3031, vehicle_id=1, allow_unconfigured_udp=True
        )
        sender = WaypointSender(settings)
        sender.sock.close()
        sender.sock = CapturingSocket()
        lat, lon = local_xy_to_latlon(
            held[:, 0], held[:, 1], self.ref.latitude, self.ref.longitude,
            self.ref.heading_deg,
        )
        sent = sender.send(1, lat, lon)
        packet = sender.sock.sent[0][0]
        magic, version, control, _seq, vehicle_id, count = struct.unpack(
            settings.byte_order + "IBBIHH", packet[:14]
        )
        self.assertEqual(sent, 1)
        self.assertEqual((magic, version, control, vehicle_id, count),
                         (0x514D3031, 1, 1, 1, 1))
        point = struct.unpack("<dd", packet[14:])
        self.assertAlmostEqual(point[0], latest.latitude, places=6)
        self.assertAlmostEqual(point[1], latest.longitude, places=6)


if __name__ == "__main__":
    unittest.main()
