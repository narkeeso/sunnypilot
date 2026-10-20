"""Composite rows for longitudinal analysis.

The aligned (t, vEgo, aTarget, src, dRel, vRel, ...) stream is the shape we kept
building by hand all session. Units: vEgo m/s, aTarget/aEgo m/s^2, dRel m, vRel m/s.
Leads are vision-only on this car (radarState leadOne comes from the model).
src: 0=cruise, 1-3=lead0/1/2 (MPC), 4=e2e.
"""
import numpy as np

from driveq.reader import iter_msgs

SRC = {0: "cruise", 1: "lead0", 2: "lead1", 3: "lead2", 4: "e2e"}
LP = 0.3  # lowpass factor for ttc_dot (matches e2e_bias GAP_TTC_LP)


def follow_rows(base, route, segs, fast=True):
  """Last-known carState/radarState/carControl values aligned to each longitudinalPlan.

  Row = (t, vEgo, aTarget, src, dRel, vRel, modelProb, radarMatched, brakePressed,
         longActive, aEgoMin).
  modelProb = vision lead confidence when present (nan if no lead);
  radarMatched = radar field (always False on this car: leads are vision-only);
  brakePressed = driver brake pedal pressed at any carState since the previous row
  (NOT the brake lights; openpilot braking leaves it False);
  longActive = openpilot long control active at every carControl since the
  previous row (False until the first carControl);
  aEgoMin = min measured carState.aEgo since the previous row (carState is
  ~10 Hz vs the ~2 Hz plan in qlogs, so this keeps the felt peak)."""
  rows = []
  for seg in segs:
    v = d = vr = prob = ae = np.nan
    isradar = False
    la = None
    # accumulated between plan rows so a short brake tap or disengage isn't missed
    bp_any, la_all, ae_min = False, True, np.inf
    for s, t, msg in iter_msgs(base, route, [seg], ("carState", "carControl", "radarState", "longitudinalPlan"), fast):
      w = msg.which()
      if w == "carState":
        v = msg.carState.vEgo
        ae = msg.carState.aEgo
        ae_min = min(ae_min, ae)
        bp_any = bp_any or bool(msg.carState.brakePressed)
      elif w == "carControl":
        la = bool(msg.carControl.longActive)
        la_all = la_all and la
      elif w == "radarState":
        lo = msg.radarState.leadOne
        d = lo.dRel if lo.present else np.nan
        vr = lo.vRel if lo.present else np.nan
        prob = lo.modelProb if lo.present else np.nan
        isradar = bool(lo.radar) if lo.present else False
      elif w == "longitudinalPlan":
        en = msg.longitudinalPlan.longitudinalPlanSource
        src = en.raw if hasattr(en, "raw") else int(en)
        rows.append([t, v, msg.longitudinalPlan.aTarget, src, d, vr, prob, isradar, bp_any,
                     la_all and la is True, ae_min if ae_min < np.inf else ae])
        bp_any, la_all, ae_min = False, True, np.inf
  return rows


def engaged(row):
  """openpilot was in control: long active and the driver not braking."""
  return row[9] and not row[8]


def route_rows(base, route, segs, fast=True):
  """follow_rows for a whole route as (seg, tg, row), tg = route-relative s.

  shortcut: tg = seg*60 + t assumes 60 s segments (openpilot rotates on that),
  so ~1 s error across a boundary; fine for 3-15 s windows, use logMonoTime if
  sub-second cross-segment timing ever matters."""
  out = []
  for seg in segs:
    for r in follow_rows(base, route, [seg], fast):
      out.append((seg, seg * 60.0 + r[0], r))
  return out


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


def stop_profile(rr, v_thresh=0.4, lookback=3.0):
  """Stops in route rows rr (from route_rows), with the felt bite.

  A stop = vEgo drops below v_thresh from a valid vEgo >= v_thresh on the
  previous row of the SAME segment (a segment that starts already stopped is
  not a stop). Returns (stops, skipped): stops = [(seg, t, aEgoMin, aTargetMin)]
  over the last `lookback` s where openpilot was in control the whole
  approach; skipped = count of stops with the driver braking or long inactive."""
  out = []
  skipped = 0
  for i in range(1, len(rr)):
    seg, tg, r = rr[i]
    pseg, _, pr = rr[i - 1]
    if not (r[1] < v_thresh and pseg == seg and pr[1] >= v_thresh):  # nan vEgo fails >=
      continue
    win = [x[2] for x in rr[max(0, i - 40):i + 1] if tg - lookback <= x[1] <= tg]  # 40 rows > lookback at 2 Hz
    if not all(engaged(x) for x in win):
      skipped += 1
      continue
    out.append((seg, r[0], min(x[10] for x in win), min(x[2] for x in win)))
  return out, skipped


def phantom_episodes(rr, min_decel=-0.6, min_v=2.0, far_lead=60.0, stop_ahead=15.0,
                     gap=2.0, settle=1.0):
  """Phantom braking episodes in route rows rr (from route_rows).

  A row is a hit when openpilot is in control, measured aEgo <= min_decel at
  v > min_v, and there is no lead to brake for: none present, or one that is
  opening (vRel > -0.5) and farther than far_lead. Rows in the first `settle`
  s of a segment are ignored (radarState not yet seen, so dRel reads nan).
  Hits within `gap` s form one episode. Episodes followed by vEgo < 0.5
  within stop_ahead s are dropped as light/stop-sign stops. Returns
  [dict(seg, t, dur, aEgoMin, v, dRel, vRel, src)]."""
  def hit(r):
    if r[0] < settle or not engaged(r) or not (r[10] <= min_decel and r[1] > min_v):
      return False
    d, vr = r[4], r[5]
    return d != d or (vr > -0.5 and d > far_lead)

  eps = []
  cur = None  # [first idx, last idx]
  for i, (seg, tg, r) in enumerate(rr):
    if r[0] < settle:
      continue  # neither hit nor miss: don't split an episode across a boundary
    if hit(r):
      if cur is not None and tg - rr[cur[1]][1] <= gap:
        cur[1] = i
      else:
        if cur is not None:
          eps.append(cur)
        cur = [i, i]
  if cur is not None:
    eps.append(cur)

  out = []
  for a, b in eps:
    t_end = rr[b][1]
    if any(x[2][1] < 0.5 for x in rr[b:b + 100] if x[1] <= t_end + stop_ahead):  # 100 rows > stop_ahead at 2 Hz
      continue
    seg, tg, r = rr[a]
    out.append({"seg": seg, "t": r[0], "dur": t_end - tg,
                "aEgoMin": min(x[2][10] for x in rr[a:b + 1]),
                "v": r[1], "dRel": r[4], "vRel": r[5], "src": r[3]})
  return out


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
  eng = np.array([bool(engaged(r)) for r in rows])
  mask = present & eng & (vs > v_min) & (ds >= d_lo) & (ds <= d_hi)

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

  Answers: does the model actually SEE the lead (vision-only on this car)?
  - present%        : share of rows with dRel cleared (present lead)
  - slow%           : share of rows vEgo < 3 m/s (stopped/crawling behind traffic)
  - present@slow%   : lead present AMONG slow rows — the stopped-lead vision test
  - flips           : present->absent->present transitions (flicker)
  - late_detect%    : of hard-close first-seen events (vRel < -4, v>5), share where
                      the lead first appeared at dRel < 60m (early is 60m+)
  - first_d_med     : median first-seen dRel of hard-close events
  Only rows where openpilot was in control (see engaged) count.
  Returns dict of metrics (or None if segment has too few rows).
  """
  rows = [r for r in rows if engaged(r)]
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
  # shortcut: dropping driver-controlled rows joins the engaged rows on either
  # side, so an episode can span a disengagement; split runs on gaps if that matters
  rows = [r for r in rows if engaged(r)]
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