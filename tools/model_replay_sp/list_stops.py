"""List stops with exact segment/second (runs on device). usage: list_stops.py [--jobs] ROUTE [...]

--jobs prints run_batch.sh job lines instead: one 35 s window (30 s before to 5 s after) per stop behind a car.

Stop = vEgo below 0.3 m/s after >= 4 m/s within the prior 20 s. Uses qlog carState/carControl.
Prints segment, seconds into that segment, peak measured decel in the last 3 s, lead state, and
whether the driver braked (or long control was inactive) in the last 3 s.
"""
import os
import sys

from openpilot.tools.lib.logreader import LogReader

BASE = os.environ.get("REPLAY_LOG_DIR", "/data/media/0/realdata")


def iter_msgs(route, segs, which):
  """Yield (seg, t, msg) from each segment's qlog; t is seconds since that segment's first wanted message."""
  for seg in segs:
    d = f"{BASE}/{route}--{seg}"
    ql = [f for f in os.listdir(d) if f.startswith("qlog")]
    if not ql:
      continue
    t0 = None
    for m in LogReader(f"{d}/{ql[0]}"):
      if m.which() in which:
        t0 = m.logMonoTime if t0 is None else t0
        yield seg, (m.logMonoTime - t0) / 1e9, m


JOBS = "--jobs" in sys.argv
for route in [a for a in sys.argv[1:] if a != "--jobs"]:
  segs = sorted(int(d.split("--")[2]) for d in os.listdir(BASE) if d.startswith(route + "--"))
  cs, ctl, lead = [], [], []
  for seg, t, m in iter_msgs(route, segs, ("carState", "carControl", "radarState")):
    w = m.which()
    if w == "carState":
      cs.append((seg, t, m.carState.vEgo, m.carState.aEgo, m.carState.brakePressed))
    elif w == "carControl":
      ctl.append((seg, t, m.carControl.longActive))
    else:
      lo = m.radarState.leadOne
      lead.append((seg, t, lo.present, lo.dRel))
  for i in range(1, len(cs)):
    seg, t, v, _, _ = cs[i]
    if not (v < 0.3 <= cs[i - 1][2]):
      continue
    prior = [c for c in cs[:i] if c[0] == seg and t - 20 <= c[1] or c[0] == seg - 1 and c[1] >= 40 + t]
    if not any(c[2] >= 4.0 for c in prior):
      continue
    last3 = [c for c in cs[:i + 1] if c[0] == seg and t - 3 <= c[1] <= t]
    peak = min(c[3] for c in last3)
    braked = any(c[4] for c in last3) or not all(c[2] for c in ctl if c[0] == seg and t - 3 <= c[1] <= t)
    ld = [x for x in lead if x[0] == seg and t - 3 <= x[1] <= t and x[2]]
    if JOBS:
      if ld:  # stops behind a car only; a window needs 30 s before the stop, possibly from the previous segment
        segs_s, start = (str(seg), t - 30) if t >= 30 or seg == 0 else (f"{seg - 1},{seg}", t + 30)
        print(f"{route[-8:]}_{seg}_{int(t)} {route} {segs_s} {max(start, 0):.1f}")
      continue
    print(f"{route} seg {seg:>2} t {t:5.1f}s peak {peak:+.2f} lead {'yes %.0fm' % ld[-1][3] if ld else 'no':>8} "
          f"{'DRIVER' if braked else 'system'}")
