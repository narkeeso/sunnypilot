"""Composite rows for longitudinal analysis.

The aligned (t, vEgo, aTarget, src, dRel, vRel) stream is the shape we kept
building by hand all session. Units: vEgo m/s, aTarget m/s^2, dRel m, vRel m/s.
src: 0=cruise, 1-3=lead0/1/2 (MPC), 4=e2e.
"""
import numpy as np

from driveq.reader import iter_msgs

SRC = {0: "cruise", 1: "lead0", 2: "lead1", 3: "lead2", 4: "e2e"}
LP = 0.3  # lowpass factor for ttc_dot (matches e2e_bias GAP_TTC_LP)


def follow_rows(base, route, segs, fast=True):
  """Last-known carState/radarState values aligned to each longitudinalPlan."""
  rows = []
  for seg in segs:
    v = d = vr = np.nan
    for s, t, msg in iter_msgs(base, route, [seg], ("carState", "radarState", "longitudinalPlan"), fast):
      w = msg.which()
      if w == "carState":
        v = msg.carState.vEgo
      elif w == "radarState":
        lo = msg.radarState.leadOne
        d = lo.dRel if lo.present else np.nan
        vr = lo.vRel if lo.present else np.nan
      elif w == "longitudinalPlan":
        en = msg.longitudinalPlan.longitudinalPlanSource
        src = en.raw if hasattr(en, "raw") else int(en)
        rows.append([t, v, msg.longitudinalPlan.aTarget, src, d, vr])
  return rows


def ttc(row, min_v=0.5):
  v = row[1]
  d = row[4]
  if np.isnan(d) or v < min_v:
    return np.nan
  return d / v


def ttc_series(rows, lp=LP):
  """Per-row (t, vEgo, dRel, ttc, ttc_dot_raw, ttc_dot_lp). Uses real t diffs."""
  out = []
  prev_t = prev_ttc = None
  dot_lp = 0.0
  for r in rows:
    t, v, d = r[0], r[1], r[4]
    tt = ttc(r)
    dot = dot_lp = np.nan
    if not np.isnan(tt) and prev_ttc is not None and prev_t is not None and t > prev_t:
      dt = t - prev_t
      dot = (tt - prev_ttc) / dt
      dot_lp = (1.0 - lp) * dot_lp if not np.isnan(dot_lp) else 0.0
      # recompute lp properly with stored prev
    out.append([t, v, d, tt, dot, None])
    prev_t, prev_ttc = t, tt
  # second pass for the lowpass (needs ordered dots)
  lp_val = 0.0
  for row in out:
    dot = row[4]
    if not np.isnan(dot):
      lp_val = (1.0 - lp) * lp_val + lp * dot
      row[5] = lp_val
    else:
      row[5] = np.nan
  return out


def brake_flips(rows, thresh=-0.12):
  """Count brake-state transitions (the highway-pulse metric)."""
  st = [r[2] <= thresh for r in rows]
  return sum(1 for i in range(1, len(st)) if st[i] != st[i - 1])


def brake_episodes(rows, thresh=-0.12):
  """Consecutive brake-on runs: [(duration_s, min_a), ...]."""
  eps = []
  cur = None
  for r in rows:
    if r[2] <= thresh and cur is None:
      cur = [r]
    elif r[2] <= thresh and cur is not None:
      cur.append(r)
    elif r[2] > thresh and cur is not None:
      eps.append((cur[-1][0] - cur[0][0], min(x[2] for x in cur)))
      cur = None
  if cur:
    eps.append((cur[-1][0] - cur[0][0], min(x[2] for x in cur)))
  return eps


def stops(rows, v_thresh=0.4):
  """Stop events: vEgo crossing below v_thresh. Returns list of (t, v_at_event)."""
  out = []
  prev = False
  for r in rows:
    below = r[1] < v_thresh
    if below and not prev:
      out.append((r[0], r[1]))
    prev = below
  return out


def stop_profile(rows, v_thresh=0.4, lookback=1.0):
  """Per stop: deepest aTarget in the last `lookback` s before the stop."""
  evs = stops(rows, v_thresh)
  prof = []
  for t, _ in evs:
    window = [r for r in rows if t - lookback <= r[0] <= t]
    amin = min((r[2] for r in window), default=np.nan)
    prof.append((t, amin))
  return prof


def seg_stats(rows):
  """Compact per-segment metrics: rows, episodes, min depth, flips, brake %, src, lead%."""
  from collections import Counter
  eps = brake_episodes(rows)
  flips = brake_flips(rows)
  brk = sum(1 for r in rows if r[2] <= -0.12)
  brk_pct = 100.0 * brk / len(rows) if rows else 0.0
  srcs = Counter(r[3] for r in rows)
  top_src = srcs.most_common(1)[0][0] if srcs else -1
  lead = sum(1 for r in rows if r[4] == r[4])
  lead_pct = 100.0 * lead / len(rows) if rows else 0.0
  amin = min((r[2] for r in rows), default=0.0)
  return {
    "rows": len(rows), "episodes": len(eps), "min_a": round(amin, 3),
    "flips": flips, "brk_pct": round(brk_pct, 1), "lead_pct": round(lead_pct, 1),
    "top_src": SRC.get(top_src, str(top_src)),
  }