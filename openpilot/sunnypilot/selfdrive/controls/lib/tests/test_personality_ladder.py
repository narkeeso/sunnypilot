"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import unittest

from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.controls.lib.personality_ladder import (
  CONFIRM_FRAMES,
  HYSTERESIS_MPH,
  MPH_TO_MS,
  PersonalityLadder,
)

AGGRESSIVE = log.LongitudinalPersonality.aggressive
STANDARD = log.LongitudinalPersonality.standard
RELAXED = log.LongitudinalPersonality.relaxed


def mph(v):
  return v * MPH_TO_MS


class TestPersonalityLadder(unittest.TestCase):
  def test_thresholds_up(self):
    ladder = PersonalityLadder()
    ladder.seed(0.0)
    assert ladder.tier == AGGRESSIVE
    ladder.seed(mph(40.0))
    assert ladder.tier == STANDARD
    ladder.seed(mph(60.0))
    assert ladder.tier == RELAXED

  def test_no_up_shift_below_threshold(self):
    ladder = PersonalityLadder()
    ladder.seed(mph(39.9))
    assert ladder.tier == AGGRESSIVE
    ladder.seed(mph(59.9))
    assert ladder.tier == STANDARD

  def test_hysteresis_holds_down_shift(self):
    # already standard: still standard 3 mph under 40, aggressive below that
    ladder = PersonalityLadder()
    ladder.seed(mph(45.0))
    for _ in range(10):
      ladder.update(mph(40.0 - HYSTERESIS_MPH + 0.1))
    assert ladder.tier == STANDARD

    for _ in range(10):
      ladder.update(mph(40.0 - HYSTERESIS_MPH - 0.1))
    assert ladder.tier == AGGRESSIVE

  def test_hysteresis_holds_relaxed(self):
    ladder = PersonalityLadder()
    ladder.seed(mph(65.0))
    assert ladder.tier == RELAXED
    for _ in range(10):
      ladder.update(mph(60.0 - HYSTERESIS_MPH + 0.1))
    assert ladder.tier == RELAXED
    for _ in range(10):
      ladder.update(mph(60.0 - HYSTERESIS_MPH - 0.1))
    assert ladder.tier == STANDARD

  def test_confirm_frames(self):
    ladder = PersonalityLadder()
    ladder.seed(0.0)
    for i in range(CONFIRM_FRAMES - 1):
      assert ladder.update(mph(45.0)) is None, i
      assert ladder.tier == AGGRESSIVE
    assert ladder.update(mph(45.0)) == STANDARD
    assert ladder.tier == STANDARD

  def test_candidate_reset_needs_consecutive_frames(self):
    ladder = PersonalityLadder()
    ladder.seed(0.0)
    ladder.update(mph(45.0))
    ladder.update(mph(45.0))
    assert ladder.tier == STANDARD
    ladder.update(mph(30.0))
    ladder.update(mph(45.0))
    ladder.update(mph(30.0))
    assert ladder.tier == STANDARD

  def test_no_thrash_on_boundary(self):
    ladder = PersonalityLadder()
    ladder.seed(mph(45.0))
    for v in (39.5, 40.5) * 100:
      ladder.update(mph(v))
    assert ladder.tier == STANDARD

  def test_stoplight_returns_to_aggressive(self):
    ladder = PersonalityLadder()
    ladder.seed(mph(50.0))
    assert ladder.tier == STANDARD
    for _ in range(10):
      ladder.update(0.0)
    assert ladder.tier == AGGRESSIVE

  def test_seed_does_not_walk_ladder(self):
    ladder = PersonalityLadder()
    ladder.seed(mph(70.0))
    assert ladder.tier == RELAXED
    ladder.seed(0.0)
    assert ladder.tier == AGGRESSIVE

  def test_reversing(self):
    ladder = PersonalityLadder()
    ladder.seed(-1.0)
    assert ladder.tier == AGGRESSIVE

  def test_custom_thresholds(self):
    ladder = PersonalityLadder(standard_mph=35, relaxed_mph=65)
    ladder.seed(mph(50.0))
    assert ladder.tier == STANDARD
    for _ in range(10):
      ladder.update(mph(64.0))
    assert ladder.tier == STANDARD
    ladder.update(mph(65.0))
    ladder.update(mph(65.0))
    assert ladder.tier == RELAXED

  def test_minimum_gap(self):
    ladder = PersonalityLadder(standard_mph=50, relaxed_mph=40)
    assert (ladder.standard_mph, ladder.relaxed_mph) == (50, 60)


if __name__ == "__main__":
  unittest.main()
