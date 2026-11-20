import math
import unittest

from bot import _angle_delta, _lead_point, _normalize


class GeometryTests(unittest.TestCase):
    def test_angle_delta_wraps_at_pi(self):
        self.assertAlmostEqual(_angle_delta(-math.pi + 0.1, math.pi - 0.1), 0.2)

    def test_normalize_handles_zero_and_direction(self):
        self.assertEqual(_normalize(0.0, 0.0), [0.0, 0.0])
        x, y = _normalize(3.0, 4.0)
        self.assertAlmostEqual(math.hypot(x, y), 1.0)
        self.assertAlmostEqual(x, 0.6)

    def test_intercept_leads_a_moving_target(self):
        me = {"pos": [0.0, 0.0, 0.0], "stats": {"muzzle_height": 7.0}}
        target = {"pos": [100.0, 0.0, 0.0], "vel": [0.0, 10.0, 0.0]}
        yaw, pitch = _lead_point(me, target, 100.0, 4.5)
        self.assertGreater(yaw, 0.0)
        self.assertLess(yaw, 0.2)
        self.assertLess(pitch, 0.0)


if __name__ == "__main__":
    unittest.main()
