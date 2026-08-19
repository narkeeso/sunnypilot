import unittest

from openpilot.sunnypilot.selfdrive.controls.lib.tuning.cruise_tuning import decel_min

MPH_TO_MS = 0.44704


class TestCruiseTuning(unittest.TestCase):
  def test_gentle_small_reductions(self):
    v_cruise = 15.0  # ~54 km/h
    for mph_over in (0, 1, 2, 3):
      v_ego = v_cruise + mph_over * MPH_TO_MS
      self.assertAlmostEqual(decel_min(v_ego, v_cruise), -0.3, places=3)

  def test_ramps_toward_cap(self):
    v_cruise = 15.0
    # ~5 mph over: just past the gentle region, mild
    v_ego = v_cruise + 5.0 * MPH_TO_MS
    d = decel_min(v_ego, v_cruise)
    self.assertAlmostEqual(d, -0.34, places=2)
    # ~7 mph over: ramping hard
    d = decel_min(15.0 + 7.0 * MPH_TO_MS, v_cruise)
    self.assertAlmostEqual(d, -0.97, places=2)

  def test_full_authority_by_10_mph(self):
    v_cruise = 15.0
    for mph_over in (10, 14, 25):
      d = decel_min(15.0 + mph_over * MPH_TO_MS, v_cruise)
      self.assertEqual(d, -1.2)

  def test_no_underspeed_no_decel(self):
    # at or below the set speed the floor is the gentle minimum, never positive
    self.assertAlmostEqual(decel_min(15.0, 15.0), -0.3, places=3)
    self.assertGreaterEqual(decel_min(14.0, 15.0), -0.31)


if __name__ == "__main__":
  unittest.main()
