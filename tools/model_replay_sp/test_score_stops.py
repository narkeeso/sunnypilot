"""Checks for the stop scorer. Run: python tools/model_replay_sp/test_score_stops.py (no device or openpilot needed)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_stops import held_since, stops  # noqa: E402

S = int(1e9)


def series(a_req, dt=0.05):
  return [(int(i * dt * S), {"aReq": a}) for i, a in enumerate(a_req)]


class TestScoreStops(unittest.TestCase):
  def test_onset_survives_standstill_relax(self):
    # braking from t=2 s, relaxed in the final 0.5 s before a stop at t=10 s (standstill hand-off)
    a = [0.0] * 40 + [-1.0] * 150 + [0.0] * 10
    onset = held_since(series(a), 10 * S, lambda r: r["aReq"] < -0.3)
    self.assertAlmostEqual(onset, 8.0, delta=0.06)

  def test_onset_resets_after_a_gap(self):
    # brief release 3 s before the stop: onset counts only the final continuous stretch
    a = [-1.0] * 100 + [0.0] * 20 + [-1.0] * 80
    onset = held_since(series(a), 10 * S, lambda r: r["aReq"] < -0.3)
    self.assertAlmostEqual(onset, 4.0, delta=0.06)

  def test_stop_needs_prior_speed(self):
    fast = [(int(i * 0.1 * S), max(0.0, 8.0 - i * 0.2), False) for i in range(60)]
    creep = [(int(i * 0.1 * S), max(0.0, 2.0 - i * 0.1), False) for i in range(60)]
    self.assertEqual(len(stops(fast)), 1)
    self.assertEqual(stops(creep), [])  # never above 4 m/s: not a stop worth scoring


if __name__ == "__main__":
  unittest.main()
