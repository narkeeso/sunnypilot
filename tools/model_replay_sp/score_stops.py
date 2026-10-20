"""Score replayed models on stops. usage: score_stops.py RESULT_JSON [...]

Replayed modelV2 rows align by order with the logged modelV2 frames (which carry logMonoTime).
A stop = vEgo below 0.3 m/s after >= 4 m/s within the prior 20 s (from logged carState).
Per model per stop, times are seconds BEFORE the stop (bigger = earlier = more anticipation):
  brake_onset   earliest time from which aReq stays below -0.3 until 1 s before the stop
  sees_stopped  (shown as warn_s) earliest time from which the model's lead (prob>0.1, within 60 m) stays below 1.0 m/s until the stop
                (lead prob runs low on some models even when the stopped car is clearly tracked)
  peak          deepest aReq in the final 3 s
  pre_peak      deepest aReq 3-8 s before the stop (early braking; less negative = later)
"""
import json
import sys
from collections import defaultdict


def stops(car):
  out = []
  for i in range(1, len(car)):
    t, v, _ = car[i]
    if v < 0.3 <= car[i - 1][1]:
      if any(vv >= 4.0 for tt, vv, _ in car[:i] if tt >= t - 20e9):
        out.append(t)
  return out


def held_since(series, t_stop, cond, margin_s=1.0):
  """Seconds before t_stop from which cond(row) holds continuously until margin_s before the stop
  (the last second is standstill hand-off, where models relax the request)."""
  start = None
  for t, row in series:
    if t > t_stop - margin_s * 1e9:
      break
    if cond(row):
      start = t if start is None else start
    else:
      start = None
  return (t_stop - start) / 1e9 if start is not None else 0.0


def score(d):
  monos = [mono for _, _, mono in d["logged"]]
  series = list(zip(monos, d["rows"]))
  res = []
  for t_stop in stops(d["car"]):
    if t_stop - 8e9 < monos[0] + 5e9:  # need warmup + 8 s of approach inside the replay
      continue
    def window(lo, hi):
      return [r["aReq"] for t, r in series if t_stop - hi * 1e9 <= t <= t_stop - lo * 1e9]
    braked = any(b for t, _, b in d["car"] if t_stop - 3e9 <= t <= t_stop)
    res.append({
      "t": (t_stop - monos[0]) / 1e9,
      "brake_onset": held_since(series, t_stop, lambda r: r["aReq"] < -0.3),
      "sees_stopped": held_since(series, t_stop, lambda r: (r["leadProb"] or 0) > 0.1 and (r["leadX"] or 999) < 60 and (r["leadV"] or 99) < 1.0),
      "peak": min(window(0, 3), default=float("nan")),
      "pre_peak": min(window(3, 8), default=float("nan")),
      "driver_braked": braked,
    })
  return res


def main():
  table = defaultdict(list)
  for f in sys.argv[1:]:
    d = json.load(open(f))
    for s in score(d):
      table[d["model"]].append({**s, "where": f"{d['route'][-10:]}/{'-'.join(map(str, d['segs']))}"})
  print(f"{'model':<8}{'stop':>22}{'onset_s':>9}{'warn_s':>8}{'pre_pk':>8}{'peak':>7}  driver")
  for model, rows in table.items():
    for s in rows:
      print(f"{model:<8}{s['where'] + '@' + str(round(s['t'])) + 's':>22}{s['brake_onset']:>9.1f}{s['sees_stopped']:>8.1f}"
            f"{s['pre_peak']:>8.2f}{s['peak']:>7.2f}  {'yes' if s['driver_braked'] else ''}")
  print("\nAVERAGES (earlier onset/sees = better anticipation; less negative peak = gentler finish)")
  for model, rows in table.items():
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    print(f"{model:<8} stops={n}  onset {avg('brake_onset'):.1f}s  warning {avg('sees_stopped'):.1f}s before stop  "
          f"pre_peak {avg('pre_peak'):.2f}  peak {avg('peak'):.2f}")


if __name__ == "__main__":
  main()
