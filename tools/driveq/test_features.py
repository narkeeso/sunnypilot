"""Synthetic-row checks for stop_profile and phantom_episodes. Run: python -m driveq.test_features"""
from driveq.features import phantom_episodes, stop_profile

NAN = float("nan")


def row(t, v, a_ego, d=NAN, vr=NAN, la=True, bp=False):
  # (t, vEgo, aTarget, src, dRel, vRel, modelProb, radarMatched, brakePressed, longActive, aEgoMin)
  return [t, v, a_ego, 4, d, vr, NAN, False, bp, la, a_ego]


def rr_of(seg, rows):
  return [(seg, seg * 60.0 + r[0], r) for r in rows]


def test_stops():
  # segment starts already stopped: not a stop
  assert stop_profile(rr_of(0, [row(0.5, 0.0, 0.0), row(1.0, 0.0, 0.0)])) == ([], 0)
  # engaged approach: one stop, bite from aEgo
  rows = [row(0.5 * i, 3.0 - 0.5 * i, -1.0 - 0.1 * i) for i in range(7)]  # v hits 0.0 at t=3.0
  stops, skipped = stop_profile(rr_of(1, rows))
  assert skipped == 0 and len(stops) == 1 and stops[0][0] == 1 and abs(stops[0][2] + 1.6) < 1e-9
  # driver braking during the approach: skipped
  rows[3][8] = True
  assert stop_profile(rr_of(1, rows)) == ([], 1)


def test_phantom():
  cruise = [row(1.0 + 0.5 * i, 15.0, 0.0) for i in range(10)]
  brake = [row(6.0 + 0.5 * i, 15.0 - 0.5 * i, -1.0) for i in range(4)]
  after = [row(8.0 + 0.5 * i, 13.0, 0.0) for i in range(40)]
  assert len(phantom_episodes(rr_of(0, cruise + brake + after))) == 1
  # braking for a closing lead is not phantom
  closing = [row(r[0], r[1], r[10], d=30.0, vr=-3.0) for r in brake]
  assert phantom_episodes(rr_of(0, cruise + closing + after)) == []
  # driver not in control is not phantom
  driver = [row(r[0], r[1], r[10], la=False) for r in brake]
  assert phantom_episodes(rr_of(0, cruise + driver + after)) == []
  # ends in a stop within 15 s (red light): dropped
  stop = [row(8.0 + 0.5 * i, 0.0, 0.0) for i in range(4)]
  assert phantom_episodes(rr_of(0, cruise + brake + stop)) == []


if __name__ == "__main__":
  test_stops()
  test_phantom()
  print("ok")
