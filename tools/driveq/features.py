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
  """Last-known carState/radarState values aligned to each longitudinalPlan.

  Row = (t, vEgo, aTarget, src, dRel, vRel, modelProb, radarMatched, brakePressed).
  modelProb = vision lead confidence when present (nan if no lead);
  radarMatched = True if the lead is radar-matched (radar field) vs vision-only;
  brakePressed = physical brake pedal switch (the brake-light signal, nan if
  the field is absent on this car)."""
  rows = []
  for seg in segs:
    v = d = vr = prob = np.nan
    isradar = False
    bp = np.nan
    for s, t, msg in iter_msgs(base, route, [seg], ("carState", "radarState", "longitudinalPlan"), fast):
      w = msg.which()
      if w == "carState":
        v = msg.carState.vEgo
        bp = getattr(msg.carState, "brakePressed", np.nan)
      elif w == "radarState":
        lo = msg.radarState.leadOne
        d = lo.dRel if lo.present else np.nan
        vr = lo.vRel if lo.present else np.nan
        prob = lo.modelProb if lo.present else np.nan
        isradar = bool(lo.radar) if lo.present else False
      elif w == "longitudinalPlan":
        en = msg.longitudinalPlan.longitudinalPlanSource
        src = en.raw if hasattr(en, "raw") else int(en)
        rows.append([t, v, msg.longitudinalPlan.aTarget, src, d, vr, prob, isradar, bp])
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


def oscillation_rows(rows, min_run=5.0, d_lo=15.0, d_hi=90.0, v_min=5.0,
                      detrend=15, thresh=0.03):
  """Steady-follow oscillation windows — the "perfect distance" bob metric.

  Finds contiguous runs where a lead is present, vEgo > v_min and dRel in
  [d_lo, d_hi] for >= min_run seconds. For each run reports the 1-3s-period
  oscillation in BOTH the command (aTarget) and the physics (vRel):

  - follow_s      : run duration
  - aCyc / vrCyc  : zero-crossing cycles of the signal after removing the
                    ~7.5s trend (detrend rows of moving average), counted
                    against a +/-thresh band around the residual median —
                    catches 1-3s wobble, ignores slow approach trends
  - aAmp / vrAmp  : mean |residual| (m/s^2 / m/s) — the wobble amplitude
  - vEgoPP        : 5-95 percentile spread of vEgo (feel proxy)
  - dRelPP        : 5-95 percentile spread of dRel
  - bpTrans       : brakePressed transitions inside the run (brake-light
                    flicker signal — 0 = the wobble never trips the pedal)
  - e2e% / lead%  : source mix + lead-present share in the run

  Returns list of dicts (one per window), empty if no qualifying runs.
  """
  out = []
  if not rows:
    return out
  ts = np.array([r[0] for r in rows])
  vs = np.array([r[1] for r in rows])
  at = np.array([r[2] for r in rows])
  srcs = np.array([r[3] for r in rows])
  ds = np.array([r[4] for r in rows])
  vrs = np.array([r[5] for r in rows])
  present = np.isfinite(ds) & np.isfinite(vrs)
  mask = present & (vs > v_min) & (ds >= d_lo) & (ds <= d_hi)

  # contiguous runs
  runs = []
  st = None
  for i in range(len(mask)):
    if mask[i] and st is None:
      st = i
    elif not mask[i] and st is not None:
      if ts[i - 1] - ts[st] >= min_run:
        runs.append((st, i))
      st = None
  if st is not None and ts[-1] - ts[st] >= min_run:
    runs.append((st, len(mask)))

  for (s, e) in runs:
    n = e - s
    if n <= detrend + 4:
      continue
    t = ts[s:e]
    a = at[s:e]
    vr = vrs[s:e]
    vv = vs[s:e]
    dd = ds[s:e]

    def wobble(x):
      ker = np.ones(detrend) / detrend
      trend = np.convolve(x, ker, mode="same")
      res = x - trend
      mid = np.median(res[detrend // 2:-(detrend // 2)])
      prev = None
      cyc = 0
      for v in res:
        b = v > mid + thresh
        if prev is not None and b != prev:
          cyc += 1
        prev = b
      return cyc // 2, float(np.mean(np.abs(res[detrend // 2:-(detrend // 2)])))

    a_cyc, a_amp = wobble(a)
    vr_cyc, vr_amp = wobble(vr)

    # brakePressed transitions: read from the raw row (index 8)
    bp = 0
    prev_bp = None
    for r in rows[s:e + 1]:
      cur = r[8] if len(r) > 8 else None
      if cur is None or (isinstance(cur, float) and np.isnan(cur)):
        prev_bp = None
        continue
      if prev_bp is not None and cur != prev_bp:
        bp += 1
      prev_bp = cur

    out.append({
      "follow_s": round(t[-1] - t[0], 1),
      "a_cycles": a_cyc, "a_amp": round(a_amp, 3),
      "vr_cycles": vr_cyc, "vr_amp": round(vr_amp, 3),
      "vEgo_pp": round(float(np.percentile(vv, 95) - np.percentile(vv, 5)), 2),
      "dRel_pp": round(float(np.percentile(dd, 95) - np.percentile(dd, 5)), 2),
      "bp_trans": bp,
      "e2e_pct": round(float(np.mean(srcs[s:e] == 4) * 100), 0),
      "lead_pct": round(float(np.mean(present[s:e]) * 100), 0),
    })
  return out


def lead_health(rows):
  """Lead-detection metrics for a segment. Rows: (t, vEgo, aTarget, src, dRel, vRel).

  Answers: does the model/radar pipeline actually SEE the lead?
  - present%        : share of rows with dRel cleared (present lead)
  - slow%           : share of rows vEgo < 3 m/s (stopped/crawling behind traffic)
  - present@slow%   : lead present AMONG slow rows — the stopped-lead vision test
  - flips           : present->absent->present transitions (flicker)
  - late_detect%    : of hard-close first-seen events (vRel < -4, v>5), share where
                      the lead first appeared at dRel < 60m (early is 60m+)
  - first_d_med     : median first-seen dRel of hard-close events
  Returns dict of metrics (or None if segment has too few rows).
  """
  if len(rows) < 60:
    return None
  n = len(rows)
  slow = np.array([r[1] < 3.0 for r in rows])
  pres = np.array([r[4] == r[4] for r in rows])  # dRel not NaN
  present_pct = 100.0 * np.mean(pres)
  slow_pct = 100.0 * np.mean(slow)
  present_at_slow = 100.0 * np.mean(pres[slow]) if slow.any() else float("nan")
  flips = int(np.sum(np.array([bool(r[4] == r[4]) for r in rows[1:]]) !=
                     np.array([bool(r[4] == r[4]) for r in rows[:-1]])))
  # hard-close first-seen events
  events = []
  seen = False
  for r in rows:
    pres_now = r[4] == r[4]
    vrel = r[5] if r[5] == r[5] else 0.0
    if pres_now and r[1] > 5.0 and vrel < -4.0:
      if not seen:
        events.append(r[4])
      seen = True
    elif not pres_now:
      seen = False
  late = sum(1 for d in events if d < 60)
  late_pct = 100.0 * late / len(events) if events else float("nan")
  first_med = float(np.median(events)) if events else float("nan")
  # vision-confidence metrics (row index 6 = modelProb, 7 = radarMatched)
  probs = [r[6] for r in rows if r[6] == r[6]]
  prob_med = float(np.median(probs)) if probs else float("nan")
  prob_p25 = float(np.percentile(probs, 25)) if probs else float("nan")
  prec = [r[7] for r in rows if r[6] == r[6]]
  vision_only_frac = (1.0 - float(np.mean(prec))) if prec else float("nan")
  soft_frac = (sum(1 for p in probs if p < 0.8) / len(probs)) if probs else float("nan")
  return {
    "present_pct": round(present_pct, 1),
    "slow_pct": round(slow_pct, 1),
    "present_at_slow": round(present_at_slow, 1),
    "flips": flips,
    "late_detect_pct": round(late_pct, 1),
    "first_d_med": round(first_med, 1),
    "prob_med": round(prob_med, 2),
    "prob_p25": round(prob_p25, 2),
    "vision_only": round(vision_only_frac, 2),
    "soft_frac": round(soft_frac, 3),
  }


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