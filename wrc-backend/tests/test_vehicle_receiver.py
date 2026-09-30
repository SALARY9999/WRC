"""Behavioural tests for the vehicle payload decoders and mock pose source.

These exercise the code that used to sit in ``app.py`` and now lives in
``vehicle.receiver``.  They are written against the real functions rather than
a copy, so a move that loses a constant or a default fails here.

The mock-source test is a genuine regression guard: ``MockPoseProvider.run``
referenced module-level metre-per-degree constants that were moved to
``trajectory.coordinates``.  No test executed that loop, so the resulting
``NameError`` was invisible until this file existed.
"""

from __future__ import annotations

import json
import math
import time
import unittest
from unittest.mock import patch

from time_sync import TimestampManager
from vehicle.history import PoseHistory
from vehicle.models import ChassisInfo, GlobalPose
from vehicle.receiver import (
    MockPoseProvider,
    decode_chassis_json,
    decode_global_pose_json,
    decode_vehicle_timestamp_json,
)


_ABSENT = object()
"""Pass as a value to delete a key, e.g. _packet(pose={"azimuth": _ABSENT})."""

_DEFAULT_POSE = {
    "latitude": 28.0,
    "longitude": 113.0,
    "height": 12.5,
    "azimuth": 90.0,
    "vNorth": 0.0,
    "vEast": 2.0,
    "vUp": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "chassis": {"cur_spd": 2.0, "yaw_rate": 0.0,
                "lon_acc": 0.1, "lat_acc": 0.0, "cur_gear": 1},
}


def _packet(**overrides) -> bytes:
    """The real vehicle layout: pose and chassis nested under one timestamp.

    ``pose={...}`` is merged into the default pose rather than replacing it,
    so a test only has to state the field it is varying.  Use ``_ABSENT`` to
    drop a field entirely.
    """
    pose = dict(_DEFAULT_POSE)
    for key, value in overrides.pop("pose", {}).items():
        if value is _ABSENT:
            pose.pop(key, None)
        else:
            pose[key] = value
    body = {
        "type": "vehicle_state",
        "timestamp": 1_700_000_000.25,
        "pose": pose,
    }
    body.update(overrides)
    return json.dumps(body).encode()


class GlobalPoseDecodeTests(unittest.TestCase):
    def test_real_vehicle_layout_decodes_with_documented_units(self):
        pose = decode_global_pose_json(_packet())
        self.assertAlmostEqual(pose.latitude, 28.0)
        self.assertAlmostEqual(pose.longitude, 113.0)
        self.assertAlmostEqual(pose.altitude, 12.5)
        self.assertAlmostEqual(pose.azimuth_deg, 90.0)
        self.assertAlmostEqual(pose.speed_mps, 2.0)
        # seconds on the wire, microseconds in the model
        self.assertEqual(pose.timestamp_us, 1_700_000_000_250_000)
        self.assertEqual(pose.payload_timestamp_us, pose.timestamp_us)

    def test_altitude_accepts_either_spelling(self):
        for body in ({"height": 7.0}, {"height": _ABSENT, "altitude": 7.0}):
            with self.subTest(keys=sorted(body)):
                pose = decode_global_pose_json(_packet(pose=body))
                self.assertAlmostEqual(pose.altitude, 7.0)

    def test_height_takes_precedence_over_altitude(self):
        # Both spellings are accepted, but the vehicle's "height" wins when
        # both are present - this ordering is relied on in the field.
        pose = decode_global_pose_json(_packet(pose={"height": 7.0, "altitude": 99.0}))
        self.assertAlmostEqual(pose.altitude, 7.0)

    def test_altitude_defaults_to_zero_when_absent(self):
        pose = decode_global_pose_json(
            _packet(pose={"height": _ABSENT, "altitude": _ABSENT})
        )
        self.assertAlmostEqual(pose.altitude, 0.0)

    def test_heading_converts_vehicle_azimuth_and_ignores_speed(self):
        # Default packet azimuth 90 is north on the vehicle, compass heading 0.
        pose = decode_global_pose_json(_packet())
        self.assertAlmostEqual(pose.azimuth_deg, 90.0)
        self.assertAlmostEqual(pose.heading_deg(), 0.0)

    def test_vehicle_azimuth_cardinals_map_to_compass_heading(self):
        expected = {0.0: 270.0, 90.0: 0.0, 180.0: 90.0, 270.0: 180.0}
        for azimuth, heading in expected.items():
            with self.subTest(azimuth=azimuth):
                pose = decode_global_pose_json(_packet(pose={"azimuth": azimuth}))
                self.assertAlmostEqual(pose.heading_deg(), heading)

    def test_azimuth_offset_is_applied_after_the_protocol_zero(self):
        pose = decode_global_pose_json(_packet())
        pose.azimuth_offset_deg = 180.0
        self.assertAlmostEqual(pose.heading_deg(), 180.0)

    def test_heading_uses_azimuth_even_when_velocity_points_elsewhere(self):
        pose = decode_global_pose_json(
            _packet(pose={"latitude": 1.0, "longitude": 2.0, "azimuth": 10.0,
                          "vNorth": 0.0, "vEast": 5.0})
        )
        self.assertAlmostEqual(pose.heading_deg(), 280.0)

    def test_missing_or_non_finite_azimuth_is_rejected(self):
        for body in (
            {"azimuth": _ABSENT},
            {"azimuth": None},
            {"azimuth": True},
            {"azimuth": float("nan")},
            {"azimuth": "north"},
        ):
            with self.subTest(azimuth=body["azimuth"]):
                with self.assertRaises(ValueError):
                    decode_global_pose_json(_packet(pose=body))

    def test_missing_timestamp_and_bad_layout_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_global_pose_json(json.dumps({"pose": {"latitude": 1.0}}).encode())
        with self.assertRaises(ValueError):
            decode_global_pose_json(b'{"timestamp":1,"pose":{"longitude":2}}')
        with self.assertRaises(ValueError):
            decode_global_pose_json(b'{"timestamp":1}')


class TimestampAndChassisDecodeTests(unittest.TestCase):
    def test_raw_timestamp_is_returned_unscaled(self):
        # The listener converts seconds to microseconds. This stays unscaled.
        self.assertEqual(decode_vehicle_timestamp_json(_packet()), 1_700_000_000.25)

    def test_non_numeric_timestamp_is_rejected(self):
        for value in ("now", None, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    decode_vehicle_timestamp_json(
                        json.dumps({"timestamp": value}).encode()
                    )

    def test_chassis_decodes_from_the_nested_vehicle_layout(self):
        chassis = decode_chassis_json(_packet())
        self.assertIsInstance(chassis, ChassisInfo)
        self.assertAlmostEqual(chassis.speed_mps, 2.0)
        self.assertEqual(chassis.gear, 1)
        self.assertEqual(chassis.timestamp_us, 1_700_000_000_250_000)

    def test_chassis_decodes_from_the_documented_top_level_layout(self):
        payload = json.dumps({
            "timestamp": 10.0,
            "chassis": {"cur_spd": 3.5, "yaw_rate": 90.0, "cur_gear": 2},
        }).encode()
        chassis = decode_chassis_json(payload)
        self.assertAlmostEqual(chassis.speed_mps, 3.5)
        # yaw_rate arrives in degrees/second and is stored in radians/second.
        self.assertAlmostEqual(chassis.yaw_rate_rad_s, math.pi / 2)
        self.assertEqual(chassis.gear, 2)

    def test_chassis_rejects_payloads_without_cur_spd(self):
        for body in ({"chassis": {"yaw_rate": 1.0}}, {"pose": {}}, {}):
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    decode_chassis_json(
                        json.dumps({"timestamp": 1.0, **body}).encode()
                    )


class MockPoseProviderTests(unittest.TestCase):
    """Runs the real generator loop; this is what caught the NameError."""

    def _run_for(self, seconds: float) -> list[GlobalPose]:
        history = PoseHistory()
        manager = TimestampManager()
        provider = MockPoseProvider(history, manager)
        provider.start()
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and len(history.snapshot()) < 3:
                time.sleep(0.02)
        finally:
            provider.stop()
            provider.join(timeout=2.0)
        self.assertFalse(provider.is_alive(), "mock provider did not stop")
        return history.snapshot()

    def test_generates_monotonic_poses_heading_east(self):
        poses = self._run_for(1.0)
        self.assertGreaterEqual(len(poses), 3, "mock provider produced no poses")
        timestamps = [p.timestamp_us for p in poses]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(set(timestamps)), len(timestamps))

        first, last = poses[0], poses[-1]
        # Straight east at 2 m/s: longitude increases, latitude is unchanged.
        self.assertGreater(last.longitude, first.longitude)
        self.assertAlmostEqual(last.latitude, first.latitude, places=9)
        # Latitude is untouched by an eastward track, so it still matches the
        # generator's origin exactly.  Longitude has already advanced by one
        # 0.1 s step before the first sample is published, hence >=.
        self.assertAlmostEqual(first.latitude, 31.1190589, places=9)
        self.assertGreaterEqual(first.longitude, 114.4639576)
        self.assertLessEqual(first.longitude, 114.4639576 + 1e-5)
        self.assertAlmostEqual(first.speed_mps, 2.0, places=6)

    def test_mock_step_matches_the_metre_per_degree_constants(self):
        # The generator advances a fixed speed*dt_s = 0.2 m per iteration and
        # converts it with the same constants the rest of the pipeline uses,
        # so consecutive samples must be exactly one step apart.  A mismatch
        # here means the mock pose drifts relative to the waypoints.
        #
        # Note the mock advances per *iteration*, while its timestamp is wall
        # clock, so its effective speed is only 2 m/s if each loop really takes
        # 0.1 s.  That is why this test asserts the step, not distance/second.
        poses = self._run_for(1.0)
        for before, after in zip(poses, poses[1:]):
            east_m = (after.longitude - before.longitude) * (
                111_320.0 * math.cos(math.radians(before.latitude))
            )
            self.assertAlmostEqual(east_m, 0.2, delta=1e-6)
            self.assertAlmostEqual(after.latitude, before.latitude, places=12)


if __name__ == "__main__":
    unittest.main()
