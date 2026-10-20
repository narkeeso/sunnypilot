"""driveq — query comma driving logs (device or local cache).

Base: list routes (by time range or most recent), dump aligned follow-rows.
Layers: phantom / stats / ttc / stops — segment-level FILTERS that answer
"which segments contain behavior X". Runs against ANY realdata-style base dir.

GOTCHAS (baked in, do not re-learn):
- realdata = flat `route--hash--seg` dirs; route = first two `--` parts
- wall time = segment dir mtimes; scan() takes route start/end from the
  mtimes near the median (an unsynced-clock segment can be months off) and
  never drops segments. Prefer --route to pick drives explicitly.
- t is seg-relative seconds (logMonoTime/1e9, monotonic since boot)
- vEgo m/s, vCruise KPH, aTarget m/s^2
- src: 0=cruise 1-3=lead-mpc 4=e2e
- qlog (default, fast): carState/carControl ~10 Hz, radarState ~4 Hz,
  longitudinalPlan ~2 Hz, NO modelV2. rlog: full rate, slow, small windows only
- leads are vision-only on this car (radarUnavailable): "lead" never means radar
- every metric except dump/ttc skips rows where openpilot was not in control
  (carControl.longActive False or carState.brakePressed True)

Usage:
  driveq list [--recent N | --since ISO --until ISO] [--tz HH] [--base DIR]
    Routes table: route, start/end UTC, segs, minutes. Default recent 3.

  driveq dump --route PREFIX --segs "0-5,8" [--fast|--full] [--window T0:T1]
              [--jsonl] [--base DIR]
    Aligned follow-rows (t, vEgo, aTarget, src, dRel, vRel, ttc) + flip count.
    This is the "profile" layer — use it to trace a window after a filter
    flags a segment.

  driveq phantom [--route PREFIX | --recent N | --since ISO --until ISO]
                 [--min-decel -0.6] [--min-v 2.0] [--far-lead 60.0]
    Phantom braking EPISODES: measured aEgo <= min-decel while engaged and no
    lead to brake for (none, or opening and > far-lead m), excluding episodes
    that end in a stop within 15 s (lights/stop signs). One row per episode:
    segment, t, dur, aMin, v, dRel, vRel, src. dRel=nan = no lead at all.

  driveq stats [--route PREFIX | --recent N | --since ISO --until ISO]
    Per-segment: rows, episodes (consecutive a<=-0.12 runs), minA, flips
    (brake-state transitions — the highway-pulse metric), brk% (share of time
    a<=-0.12), lead% (share with a lead, vision-only on this car), dominant src.

  driveq ttc --route PREFIX --segs "0-5" [--series]
    Default: noise summary — ttc range, raw |ttc_dot| med/p90/p99/max, and
    lowpassed same. --series: per-sample rows (t, v, dRel, ttc, dot_raw, dot_lp).
    Use for sizing controller deadbands; the gap-law deadband is 0.1 s/s.

  driveq stops [--route PREFIX | --recent N | --since ISO --until ISO]
    One row per stop (vEgo crossing <0.4 within a segment) with openpilot in
    control for the last 3 s: min measured aEgo (the felt "bite") and min
    aTarget (requested) over those 3 s. Summary counts stops skipped for
    driver braking / long inactive.

  driveq leadhealth [--route PREFIX | --recent N | --since ISO --until ISO]
    Lead-detection health per segment: present%, slow%, present@slow% (lead
    among stopped/slow frames — the model's stopped-lead vision), flips
    (flicker), late_detect% + first-dRel med on hard closes. The RDFM-vs-PMV2
    signature: RDFM fails stopped-lead detection (present@slow% low, late% high).

  driveq oscillation [--route PREFIX | --recent N | --since ISO --until ISO]
    Steady-follow bob metric: for windows where a lead is present at v>5,
    dRel 15-90m, reports 1-3s oscillation cycles + amplitude in BOTH aTarget
    (command) and vRel (physics), vEgo/dRel spread, brakePressed transitions
    (brake-light flicker), and src mix. bpTr=0 + high cycles = regen-band bob
    (comfort); bpTr>0 at speed = brake light flickering (real annoyance).

TIME ZONES: device logs are UTC. Naive --since/--until strings are read as
UTC. Pass --tz HH (e.g. -7 for Pacific summer / PDT) to interpret naive input
times as that offset (and display the list table in it). Explicit ISO offsets
(2026-08-22T11:00:00-07:00) or Z always win over --tz.

All commands accept --base to point at a downloaded local cache instead of
the device path.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from driveq.discover import filter_routes, scan, to_epoch
from driveq.features import (
  SRC,
  brake_flips,
  follow_rows,
  lead_health,
  oscillation_rows,
  phantom_episodes,
  route_rows,
  seg_stats,
  stop_profile,
  ttc,
  ttc_series,
)


def _routes(args, default_recent=3):
  routes = scan(args.base)
  tz = getattr(args, "tz", None)
  since = to_epoch(args.since, tz=tz) if getattr(args, "since", None) else None
  until = to_epoch(args.until, tz=tz) if getattr(args, "until", None) else None
  recent = getattr(args, "recent", None)
  prefixes = [p.strip() for spec in (getattr(args, "route", None) or []) for p in spec.split(",") if p.strip()]
  if recent is None and since is None and until is None and not prefixes:
    recent = default_recent
  return filter_routes(routes, since=since, until=until, recent=recent, prefixes=prefixes)


def _parse_segs(spec):
  segs = []
  for part in spec.split(","):
    part = part.strip()
    if "-" in part:
      lo, hi = part.split("-")
      segs.extend(range(int(lo), int(hi) + 1))
    else:
      segs.append(int(part))
  return segs


def _fmt(epoch, tz=None):
  dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
  if tz is not None:
    dt = dt.astimezone(timezone(timedelta(hours=tz)))
  return dt.strftime("%m-%d %H:%M")


def cmd_list(args):
  """Print a routes table: route, start/end UTC, segs, minutes.

  Scope: --recent N (default 3), or --since/--until. Naive ISO = UTC, unless
  --tz is given — then naive times are interpreted as that offset (e.g. -7 for
  PDT summer Pacific) and the table is also displayed in it.
  """
  routes = scan(args.base)
  tz = args.tz
  since = to_epoch(args.since, tz=tz) if args.since else None
  until = to_epoch(args.until, tz=tz) if args.until else None
  recent = args.recent
  if recent is None and since is None and until is None:
    recent = 3
  out = filter_routes(routes, since=since, until=until, recent=recent)
  print(f"{'route':<24}{'start':>16}{'end':>16}{'segs':>5}{'min':>6}")
  for r in out:
    mins = (r.end_epoch - r.start_epoch) / 60.0
    print(f"{r.prefix:<24}{_fmt(r.start_epoch, tz):>16}{_fmt(r.end_epoch, tz):>16}{len(r.segs):>5}{mins:>6.0f}")


def cmd_dump(args):
  """Print aligned follow-rows for segments (the "profile" layer).

  Columns: t (seg-relative s), vEgo (m/s), aTarget (m/s^2), src
  (cruise/lead0-2/e2e), dRel (m, nan = no lead), vRel (m/s), ttc (s).
  Ends with rows= and flips= (brake-state transitions, the pulse metric).
  --jsonl emits one JSON object per row instead.
  """
  # Light route resolution: only this route's dirs, no full scan (which reads
  # clocks from every segment of every route — slow).
  segs = []
  prefix = args.route + "--"
  for d in os.listdir(args.base):
    if not d.startswith(prefix):
      continue
    try:
      segs.append(int(d.rsplit("--", 1)[1]))
    except ValueError:
      pass
  if not segs:
    sys.exit(f"route not found: {args.route}")
  segs.sort()
  want = _parse_segs(args.segs)
  segs = [s for s in want if s in segs]
  if not segs:
    sys.exit("no matching segments")
  rows = follow_rows(args.base, args.route, segs, fast=args.fast)
  if args.window:
    t0, t1 = (float(x) for x in args.window.split(":"))
    rows = [row for row in rows if t0 <= row[0] <= t1]
  if args.jsonl:
    for row in rows:
      print(json.dumps({"t": round(row[0], 3), "vEgo": round(row[1], 2),
                        "aTarget": round(row[2], 4), "src": SRC.get(row[3], row[3]),
                        "dRel": round(row[4], 2) if row[4] == row[4] else None,
                        "vRel": round(row[5], 2) if row[5] == row[5] else None}))
  else:
    print(f"{'t':>8}{'vEgo':>7}{'aTarget':>9}{'src':>7}{'dRel':>7}{'vRel':>7}{'ttc':>6}")
    for row in rows:
      d = row[4]
      vr = row[5]
      tt = ttc(row)
      print(f"{row[0]:>8.1f}{row[1]:>7.2f}{row[2]:>9.3f}{SRC.get(row[3], row[3]):>7}"
            f"{('nan' if np.isnan(d) else f'{d:.1f}'):>7}{('nan' if np.isnan(vr) else f'{vr:.2f}'):>7}"
            f"{('nan' if np.isnan(tt) else f'{tt:.2f}'):>6}")
    print(f"\nrows={len(rows)} flips={brake_flips(rows)}")


def cmd_phantom(args):
  """Print phantom braking episodes in scope, one row per episode.

  Phantom = measured aEgo <= --min-decel at v > --min-v while openpilot is in
  control and there is no lead to brake for (none present, or opening with
  vRel > -0.5 and farther than --far-lead). Episodes that end in a stop
  within 15 s are dropped (model stopping for lights/stop signs).
  Columns: segment, t (seg-relative s), dur, aMin (aEgo), v, dRel, vRel, src
  at episode start. dRel=nan = no lead at all.
  """
  out = _routes(args)
  total = 0
  print(f"{'segment':<30}{'t':>6}{'dur':>5}{'aMin':>7}{'v':>6}{'dRel':>7}{'vRel':>6}{'src':>7}")
  for r in out:
    rr = route_rows(args.base, r.prefix, r.segs)
    for e in phantom_episodes(rr, args.min_decel, args.min_v, args.far_lead):
      d, vr = e["dRel"], e["vRel"]
      print(f"{r.prefix}--{e['seg']:<4}{e['t']:>6.1f}{e['dur']:>5.1f}{e['aEgoMin']:>7.2f}{e['v']:>6.1f}"
            f"{('nan' if np.isnan(d) else f'{d:.1f}'):>7}{('nan' if np.isnan(vr) else f'{vr:.1f}'):>6}"
            f"{SRC.get(e['src'], e['src']):>7}")
      total += 1
  print(f"\n{total} phantom episodes (engaged, aEgo<={args.min_decel}, v>{args.min_v}, "
        f"no lead or opening lead >{args.far_lead}m, no stop within 15s)")


def cmd_stats(args):
  """Print per-segment metrics in scope.

  Columns: rows (follow-row count), eps (consecutive a<=-0.12 runs),
  minA (deepest decel), flips (brake-state transitions — pulse metric),
  brk% (share of time braking), lead% (share with a lead), src
  (dominant source). Use flips + brk% to find pulse/yoyo segments.
  Only rows with openpilot in control count; rows=0 = never engaged.
  """
  out = _routes(args)
  print(f"{'segment':<30}{'rows':>6}{'eps':>5}{'minA':>7}{'flips':>6}{'brk%':>6}{'lead%':>6} src")
  for r in out:
    for seg in r.segs:
      rows = follow_rows(args.base, r.prefix, [seg], fast=True)
      if not rows:
        continue
      s = seg_stats(rows)
      print(f"{r.prefix}--{seg:<4}{s['rows']:>6}{s['episodes']:>5}{s['min_a']:>7}"
            f"{s['flips']:>6}{s['brk_pct']:>6}{s['lead_pct']:>6} {s['top_src']}")


def cmd_ttc(args):
  """Print TTC + TTC-rate series or noise summary for segments.

  Default: ttc range and |ttc_dot| med/p90/p99/max (raw and lowpassed).
  Use to size controller deadbands — the gap-law deadband is 0.1 s/s, and
  measured lead (vision-only on this car) dRel noise should sit well under it.
  --series: per-sample rows (t, v, dRel, ttc, dot_raw, dot_lp).
  """
  rows = follow_rows(args.base, args.route, _parse_segs(args.segs), fast=True)
  series = ttc_series(rows)
  if args.series:
    print(f"{'t':>8}{'v':>7}{'dRel':>8}{'ttc':>7}{'dot_raw':>9}{'dot_lp':>9}")
    for row in series:
      t, v, d, tt, dot, lp = row
      print(f"{t:>8.1f}{v:>7.2f}{('nan' if np.isnan(d) else f'{d:.1f}'):>8}"
            f"{('nan' if np.isnan(tt) else f'{tt:.2f}'):>7}"
            f"{('nan' if np.isnan(dot) else f'{dot:.3f}'):>9}"
            f"{('nan' if np.isnan(lp) else f'{lp:.3f}'):>9}")
  else:
    dots = [row[4] for row in series if row[4] == row[4]]
    lps = [row[5] for row in series if row[5] == row[5]]
    tts = [row[3] for row in series if row[3] == row[3]]
    if not dots:
      print("no lead data")
      return
    print(f"n={len(dots)}  ttc: {min(tts):.2f}..{max(tts):.2f}s")
    print(f"raw  |ttc_dot|: med={np.median(np.abs(dots)):.3f} "
          f"p90={np.percentile(np.abs(dots), 90):.3f} p99={np.percentile(np.abs(dots), 99):.3f} "
          f"max={np.abs(dots).max():.3f} s/s")
    if lps:
      print(f"lowp |ttc_dot|: med={np.median(np.abs(lps)):.3f} "
            f"p90={np.percentile(np.abs(lps), 90):.3f} p99={np.percentile(np.abs(lps), 99):.3f} "
            f"max={np.abs(lps).max():.3f} s/s")


def cmd_stops(args):
  """Print one row per stop in scope.

  A stop = vEgo crossing below 0.4 m/s from >= 0.4 on the previous row of the
  same segment, with openpilot in control (longActive, no brakePressed) for
  the last 3 s. Columns: segment, t (seg-relative s), aEgoMin (measured, the
  felt "bite"), aTgtMin (requested). Use dump --window to profile a stop.
  """
  out = _routes(args)
  n = skipped = 0
  print(f"{'segment':<30}{'t':>6}{'aEgoMin':>9}{'aTgtMin':>9}")
  for r in out:
    stops, sk = stop_profile(route_rows(args.base, r.prefix, r.segs))
    skipped += sk
    for seg, t, ae, at in stops:
      print(f"{r.prefix}--{seg:<4}{t:>6.1f}{ae:>9.2f}{at:>9.2f}")
      n += 1
  print(f"\n{n} stops with openpilot in control; {skipped} skipped (driver braking or long inactive)")


def cmd_leadhealth(args):
  """Lead-detection health per segment (does the pipeline SEE the lead?).

  Columns: present% (share with a lead), slow% (share v<3 m/s),
  present@slow% (lead among slow rows — the stopped-lead vision test, see
  RDFM vs PMV2), flips (track flicker), late_detect% + first-dRel med of
  hard-close first-seen events (vRel<-4, v>5). High flips or high
  late_detect% / low present@slow% = the model is NOT reading the lead
  reliably (classic RDFM-on-stopped-traffic signature).
  """
  out = _routes(args)
  print(f"{'segment':<30}{'present%':>9}{'slow%':>7}{'pres@slow%':>11}{'flips':>7}{'late%':>7}{'first_d':>8}{'probMed':>8}{'probP25':>8}{'soft%':>7}")
  for r in out:
    for seg in r.segs:
      rows = follow_rows(args.base, r.prefix, [seg], fast=True)
      if not rows:
        continue
      h = lead_health(rows)
      if not h:
        continue
      print(f"{r.prefix}--{seg:<4}{h['present_pct']:>9}{h['slow_pct']:>7}{h['present_at_slow']:>11}"
            f"{h['flips']:>7}{h['late_detect_pct']:>7}{h['first_d_med']:>8}"
            f"{h['prob_med']:>8}{h['prob_p25']:>8}{100*h['soft_frac']:>6.1f}")


def cmd_oscillation(args):
  """Steady-follow oscillation per segment — the "perfect distance" bob.

  Finds windows where a lead is present, v>5 m/s, dRel 15-90m for >=5s, then
  reports the 1-3s-period wobble in aTarget (command) AND vRel (physics):
  cycles + amplitude per window, vEgo/dRel peak-peak spread, brakePressed
  transitions (0 = brake light never trips during the bob), and src mix.
  High aCyc/aAmp with bpTrans=0 = regen-band bob (comfort nit, no light);
  bpTrans>0 at speed = the brake light IS flickering (the real annoyance).
  """
  out = _routes(args)
  print(f"{'segment':<30}{'follS':>6}{'aCyc':>5}{'aAmp':>6}{'vrCyc':>6}{'vrAmp':>6}{'vPP':>6}{'dPP':>6}{'bpTr':>5}{'e2e%':>5}")
  for r in out:
    for seg in r.segs:
      rows = follow_rows(args.base, r.prefix, [seg], fast=True)
      if not rows:
        continue
      for w in oscillation_rows(rows):
        print(f"{r.prefix}--{seg:<4}{w['follow_s']:>6}{w['a_cycles']:>5}{w['a_amp']:>6}"
              f"{w['vr_cycles']:>6}{w['vr_amp']:>6}{w['vEgo_pp']:>6}{w['dRel_pp']:>6}"
              f"{w['bp_trans']:>5}{w['e2e_pct']:>4.0f}")


def main():
  ap = argparse.ArgumentParser(prog="driveq", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--base", default="/data/media/0/realdata",
                  help="realdata dir (device path, or local cache)")
  sub = ap.add_subparsers(dest="cmd", required=True)

  p_list = sub.add_parser("list", help="list routes (recent or by time range)",
                          description="Print routes: prefix, start/end UTC, segs, minutes. "
                                      "Scope: --recent N, or --since/--until (naive ISO = UTC).",
                          formatter_class=argparse.RawDescriptionHelpFormatter)
  p_list.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_list.add_argument("--since", help="ISO time (naive = UTC), e.g. 2026-08-22T18:00:00")
  p_list.add_argument("--until", help="ISO time (naive = UTC)")
  p_list.add_argument("--tz", type=int, help="interpret naive --since/--until as this offset AND display in it (e.g. -7 = PDT)")
  p_list.set_defaults(fn=cmd_list)

  p_dump = sub.add_parser("dump", help="dump aligned follow-rows (profile a segment)",
                          description="Print (t, vEgo, aTarget, src, dRel, vRel, ttc) per plan "
                                      "sample + flips count. --window filters seconds within "
                                      "each segment. This is the 'profile' layer.",
                          formatter_class=argparse.RawDescriptionHelpFormatter)
  p_dump.add_argument("--route", required=True, help="route prefix, e.g. 000000ab--6ae3c349db")
  p_dump.add_argument("--segs", default="0", help="e.g. '0-5,8'")
  p_dump.add_argument("--fast", action="store_true", default=True, help="use qlog (default, fast)")
  p_dump.add_argument("--full", action="store_false", dest="fast", help="use rlog (slow, precise)")
  p_dump.add_argument("--window", help="t0:t1 seconds within each segment")
  p_dump.add_argument("--jsonl", action="store_true", help="emit JSON objects instead of a table")
  p_dump.set_defaults(fn=cmd_dump)

  p_phantom = sub.add_parser("phantom", help="find segments with phantom braking",
                             description="Phantom episodes: engaged, aEgo <= min-decel, v > min-v, "
                                         "no lead or an opening lead beyond far-lead, and no stop "
                                         "within 15 s after. dRel=nan = no lead at all.",
                             formatter_class=argparse.RawDescriptionHelpFormatter)
  p_phantom.add_argument("--route", action="append", help="route prefix(es), repeatable or comma list; overrides time scope")
  p_phantom.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_phantom.add_argument("--since", help="ISO time (naive = UTC)")
  p_phantom.add_argument("--until", help="ISO time (naive = UTC)")
  p_phantom.add_argument("--tz", type=int, help="input offset hours for naive --since/--until (e.g. -7 = PDT); display stays UTC")
  p_phantom.add_argument("--min-decel", type=float, default=-0.6, help="measured aEgo threshold, m/s^2 (default -0.6)")
  p_phantom.add_argument("--min-v", type=float, default=2.0, help="min speed m/s (default 2.0)")
  p_phantom.add_argument("--far-lead", type=float, default=60.0, help="an opening lead farther than this (m) is not a reason to brake (default 60)")
  p_phantom.set_defaults(fn=cmd_phantom)

  p_stats = sub.add_parser("stats", help="per-segment metrics (episodes, flips, brake_pct, src)",
                           description="Per segment: rows, episodes (a<=-0.12 runs), minA, flips "
                                       "(pulse metric), brk% (time braking), lead%, dominant src. "
                                       "High flips+brk% = pulse/yoyo segments.",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
  p_stats.add_argument("--route", action="append", help="route prefix(es), repeatable or comma list; overrides time scope")
  p_stats.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_stats.add_argument("--since", help="ISO time (naive = UTC)")
  p_stats.add_argument("--until", help="ISO time (naive = UTC)")
  p_stats.add_argument("--tz", type=int, help="input offset hours for naive --since/--until (e.g. -7 = PDT); display stays UTC")
  p_stats.set_defaults(fn=cmd_stats)

  p_ttc = sub.add_parser("ttc", help="ttc + ttc-rate series or noise summary",
                         description="Default: ttc range + |ttc_dot| med/p90/p99/max (raw and "
                                     "lowpassed) for sizing controller deadbands (gap-law "
                                     "deadband = 0.1 s/s). --series prints per-sample rows.",
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  p_ttc.add_argument("--route", required=True, help="route prefix")
  p_ttc.add_argument("--segs", default="0", help="e.g. '0-5,8'")
  p_ttc.add_argument("--series", action="store_true", help="print per-sample rows (t, v, dRel, ttc, dot_raw, dot_lp)")
  p_ttc.set_defaults(fn=cmd_ttc)

  p_stops = sub.add_parser("stops", help="stop events per segment + deepest decel before stop",
                           description="One row per stop (vEgo < 0.4 crossing) with openpilot in "
                                       "control for the last 3 s: min aEgo (felt bite) and min aTarget.",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
  p_stops.add_argument("--route", action="append", help="route prefix(es), repeatable or comma list; overrides time scope")
  p_stops.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_stops.add_argument("--since", help="ISO time (naive = UTC)")
  p_stops.add_argument("--until", help="ISO time (naive = UTC)")
  p_stops.add_argument("--tz", type=int, help="input offset hours for naive --since/--until (e.g. -7 = PDT); display stays UTC")
  p_stops.set_defaults(fn=cmd_stops)

  p_leadhealth = sub.add_parser("leadhealth", help="lead-detection health per segment",
                                description="Per segment: present%, slow%, present@slow% "
                                            "(stopped-lead vision), flips, late_detect%, "
                                            "first-dRel med. Low present@slow% + high "
                                            "late_detect% = failed stopped-lead detection "
                                            "(the RDFM signature).",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
  p_leadhealth.add_argument("--route", action="append", help="route prefix(es), repeatable or comma list; overrides time scope")
  p_leadhealth.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_leadhealth.add_argument("--since", help="ISO time (naive = UTC)")
  p_leadhealth.add_argument("--until", help="ISO time (naive = UTC)")
  p_leadhealth.add_argument("--tz", type=int, help="input offset hours for naive --since/--until (e.g. -7 = PDT); display stays UTC")
  p_leadhealth.set_defaults(fn=cmd_leadhealth)

  p_osc = sub.add_parser("oscillation", help="steady-follow bob (aTarget + vRel cycles)",
                         description="Per segment: steady-follow windows (lead, v>5, "
                                     "dRel 15-90m, >=5s) with 1-3s oscillation cycles/"
                                     "amplitude in aTarget and vRel, vEgo/dRel spread, "
                                     "brakePressed transitions, e2e%. bpTr>0 = brake "
                                     "light flickers; bpTr=0 + high cycles = regen-band "
                                     "bob (comfort only).",
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  p_osc.add_argument("--route", action="append", help="route prefix(es), repeatable or comma list; overrides time scope")
  p_osc.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_osc.add_argument("--since", help="ISO time (naive = UTC)")
  p_osc.add_argument("--until", help="ISO time (naive = UTC)")
  p_osc.add_argument("--tz", type=int, help="input offset hours for naive --since/--until (e.g. -7 = PDT); display stays UTC")
  p_osc.set_defaults(fn=cmd_oscillation)

  args = ap.parse_args()
  args.fn(args)


if __name__ == "__main__":
  main()