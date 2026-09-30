"""Independent compass/ego-frame checks; never start inference or send UDP."""
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

try:
    import app
except ModuleNotFoundError as exc:  # Local lightweight environment.
    raise unittest.SkipTest(f"camera runtime dependency unavailable: {exc}")


class HeadingCoordinatesTests(unittest.TestCase):
    def test_relative_orientation_matches_physical_body_axes(self):
        for hi, h0 in [(90, 0), (0, 90), (359, 1), (1, 359), (72.411, 72.422)]:
            with self.subTest(hi=hi, h0=h0):
                samples = [app.PoseSample(0, 28, 113, 50, hi),
                           app.PoseSample(100000, 28, 113, 50, h0)]
                xyz, rot, _ = app.build_ego_arrays(samples)
                # Physical historical forward axis projected into t0 frame.
                expected = app.enu_to_ego(math.sin(math.radians(hi)),
                                          math.cos(math.radians(hi)), 0,
                                          math.radians(h0))
                np.testing.assert_allclose(rot[0, 0, 0].numpy() @ [1, 0, 0], expected, atol=1e-6)
                np.testing.assert_allclose(rot[0, 0, -1].numpy(), np.eye(3), atol=1e-6)
                np.testing.assert_allclose(xyz[0, 0, -1].numpy(), [0, 0, 0], atol=1e-6)

    def test_waypoint_direction_and_roundtrip(self):
        for h, forward, left in [(0, (0, 10), (-3, 0)), (90, (10, 0), (0, 3)),
                                  (180, (0, -10), (3, 0)), (270, (-10, 0), (0, -3))]:
            lat, lon = app.local_xy_to_latlon(np.array([10., 0.]), np.array([0., 3.]), 28, 113, h)
            for i, expected in enumerate([forward, left]):
                e, n, u = app.enu_from_ll(lat[i], lon[i], 50, (28, 113, 50))
                np.testing.assert_allclose([e, n], expected, atol=1e-7)
                np.testing.assert_allclose(app.enu_to_ego(e, n, u, math.radians(h)),
                                           [10, 0, 0] if i == 0 else [0, 3, 0], atol=1e-7)

    def test_heading_protocol_and_invalid_values(self):
        # Vehicle azimuth is clockwise from west. Compass heading is clockwise
        # from north, so heading = (azimuth - 90) mod 360.
        for value, heading in [(0, 270), (90, 0), (180, 90), (270, 180), (72.411, 342.411)]:
            packet = {'timestamp': 1, 'pose': {'latitude': 28, 'longitude': 113, 'azimuth': value}}
            self.assertAlmostEqual(
                app.decode_global_pose_json(json.dumps(packet).encode()).heading_deg(),
                heading,
            )
        for value in [None, True, '90', float('nan'), float('inf'), -float('inf')]:
            packet = {'timestamp': 1, 'pose': {'latitude': 28, 'longitude': 113, 'azimuth': value}}
            with self.assertRaises(ValueError):
                app.decode_global_pose_json(json.dumps(packet).encode())
        with self.assertRaises(ValueError):
            app.decode_global_pose_json(b'{"timestamp":1,"pose":{"latitude":28,"longitude":113}}')


if __name__ == '__main__':
    unittest.main()
