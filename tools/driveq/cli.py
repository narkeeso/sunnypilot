"""driveq — query comma driving logs (device or local cache).

Base: list routes (by time range or most recent), dump aligned follow-rows.
Layers: phantom / stats / ttc / stops — segment-level FILTERS that answer
"which segments contain behavior X". Runs against ANY realdata-style base dir.

GOTCHAS (baked in, do not re-learn):
- realdata = flat `route--hash--seg` dirs; route = first two `--` parts
- wall time = Clocks.wallTimeNanos inside the log (dir mtimes unreliable);
  stale segments (>2h from cluster median) are dropped by scan()
- t is seg-relative seconds (logMonoTime/1e9, monotonic since boot)
- vEgo m/s, vCruise KPH, aTarget m/s^2
- src: 0=cruise 1-3=lead-mpc 4=e2e
- qlog ~20Hz fast scans (default); rlog ~100Hz slow, only small windows

Usage:
  driveq list [--recent N | --since ISO --until ISO] [--tz HH] [--base DIR]
    Routes table: route, start/end UTC, segs, minutes. Default recent 3.

  driveq dump --route PREFIX --segs "0-5,8" [--fast|--full] [--window T0:T1]
              [--jsonl] [--base DIR]
    Aligned follow-rows (t, vEgo, aTarget, src, dRel, vRel, ttc) + flip count.
    This is the "profile" layer — use it to trace a window after a filter
    flags a segment.

  driveq phantom [--recent N | --since ISO --until ISO]
                 [--min-decel -0.6] [--min-v 2.0] [--lead-sane 20.0]
    Segments with phantom braking: aTarget <= min-decel while no close lead
    (absent or dRel > lead-sane). Columns: segment, samples, aMin, v@evt,
    dRel@evt. dRel=nan means NO lead at all (true phantom); a number means
    far-lead early braking (the other annoyance class).

  driveq stats [--recent N | --since ISO --until ISO]
    Per-segment: rows, episodes (consecutive a<=-0.12 runs), minA, flips
    (brake-state transitions — the highway-pulse metric), brk% (share of time
    a<=-0.12), lead% (share with a radar lead), dominant src.

  driveq ttc --route PREFIX --segs "0-5" [--series]
    Default: noise summary — ttc range, raw |ttc_dot| med/p90/p99/max, and
    lowpassed same. --series: per-sample rows (t, v, dRel, ttc, dot_raw, dot_lp).
    Use for sizing controller deadbands; the gap-law deadband is 0.1 s/s.

  driveq stops [--recent N | --since ISO --until ISO]
    Per-segment: stop count (vEgo crossing <0.4) and deepest aTarget in the
    1s before each stop (the "bite" metric).

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
  seg_stats,
  stop_profile,
  ttc,
  ttc_series,
)
from driveq.reader import iter_msgs


def _routes(args, default_recent=3):
  routes = scan(args.base)
  tz = getattr(args, "tz", None)
  since = to_epoch(args.since, tz=tz) if getattr(args, "since", None) else None
  until = to_epoch(args.until, tz=tz) if getattr(args, "until", None) else None
  recent = getattr(args, "recent", None)
  if recent is None and since is None and until is None:
    recent = default_recent
  return filter_routes(routes, since=since, until=until, recent=recent)


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


def _phantom_in_seg(base, route, seg, min_decel, min_v, lead_sane):
  """Count phantom samples: aTarget <= min_decel with no close lead. Fast (qlog)."""
  events = []
  v = d = np.nan
  lead = False
  for s, t, msg in iter_msgs(base, route, [seg], ("carState", "radarState", "longitudinalPlan"), fast=True):
    w = msg.which()
    if w == "carState":
      v = msg.carState.vEgo
    elif w == "radarState":
      lo = msg.radarState.leadOne
      lead = lo.present
      d = lo.dRel if lo.present else np.nan
    elif w == "longitudinalPlan":
      a = msg.longitudinalPlan.aTarget
      if a <= min_decel and v > min_v and ((not lead) or (not np.isnan(d) and d > lead_sane)):
        events.append((t, a, v, d))
  return events


def cmd_phantom(args):
  """Print segments with phantom braking in scope.

  Phantom = aTarget <= --min-decel while v > --min-v and no close lead
  (lead absent, or dRel > --lead-sane). Columns: segment, sample count,
  deepest decel, v at first event, dRel at first event.
  dRel=nan → true phantom (model braking, nothing on radar);
  dRel>0 at 45-95m → far-lead early braking (the other annoyance class).
  """
  out = _routes(args)
  total = 0
  seg_hits = 0
  print(f"{'segment':<30}{'samples':>8}{'aMin':>7}{'v@evt':>7}{'dRel':>7}")
  for r in out:
    for seg in r.segs:
      ev = _phantom_in_seg(args.base, r.prefix, seg, args.min_decel, args.min_v, args.lead_sane)
      if ev:
        amin = min(e[1] for e in ev)
        ev0 = ev[0]
        print(f"{r.prefix}--{seg:<4}{len(ev):>8}{amin:>7.2f}{ev0[2]:>7.1f}"
              f"{('nan' if np.isnan(ev0[3]) else f'{ev0[3]:.1f}'):>7}")
        total += len(ev)
        seg_hits += 1
  print(f"\n{seg_hits} segments with phantom braking, {total} samples (a<={args.min_decel}, "
        f"v>{args.min_v}, lead absent or >{args.lead_sane}m)")


def cmd_stats(args):
  """Print per-segment metrics in scope.

  Columns: rows (follow-row count), eps (consecutive a<=-0.12 runs),
  minA (deepest decel), flips (brake-state transitions — pulse metric),
  brk% (share of time braking), lead% (share with a radar lead), src
  (dominant source). Use flips + brk% to find pulse/yoyo segments.
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
  measured radar noise should sit well under it.
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
  """Print stop events per segment in scope.

  Columns: segment, stop count (vEgo crossing below 0.4 m/s), deepest aTarget
  in the 1s before each stop (the "final bite" metric). Use dump --window
  around a flagged stop to profile the approach.
  """
  out = _routes(args)
  print(f"{'segment':<30}{'stops':>6}{'deepest@stop':>13}")
  for r in out:
    for seg in r.segs:
      rows = follow_rows(args.base, r.prefix, [seg], fast=True)
      if not rows:
        continue
      prof = stop_profile(rows)
      if not prof:
        continue
      amin = min(p[1] for p in prof)
      print(f"{r.prefix}--{seg:<4}{len(prof):>6}"
            f"{('nan' if np.isnan(amin) else f'{amin:.3f}'):>13}")


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
                             description="Phantom = aTarget <= min-decel, v > min-v, no close "
                                         "lead. dRel=nan at the event = true phantom; dRel 45-95m "
                                         "= far-lead early braking.",
                             formatter_class=argparse.RawDescriptionHelpFormatter)
  p_phantom.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_phantom.add_argument("--since", help="ISO time (naive = UTC)")
  p_phantom.add_argument("--until", help="ISO time (naive = UTC)")
  p_phantom.add_argument("--tz", type=int, help="input offset hours for naive --since/--until (e.g. -7 = PDT); display stays UTC")
  p_phantom.add_argument("--min-decel", type=float, default=-0.6, help="brake threshold (default -0.6)")
  p_phantom.add_argument("--min-v", type=float, default=2.0, help="min speed m/s (default 2.0)")
  p_phantom.add_argument("--lead-sane", type=float, default=20.0, help="lead farther than this = no lead (m)")
  p_phantom.set_defaults(fn=cmd_phantom)

  p_stats = sub.add_parser("stats", help="per-segment metrics (episodes, flips, brake_pct, src)",
                           description="Per segment: rows, episodes (a<=-0.12 runs), minA, flips "
                                       "(pulse metric), brk% (time braking), lead%, dominant src. "
                                       "High flips+brk% = pulse/yoyo segments.",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
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
                           description="Per segment: stop count (vEgo < 0.4 crossing) and deepest "
                                       "aTarget in the 1s before each stop (the 'bite' metric).",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
  p_stops.add_argument("--recent", type=int, help="most recent N routes (default 3)")
  p_stops.add_argument("--since", help="ISO time (naive = UTC)")
  p_stops.add_argument("--until", help="ISO time (naive = UTC)")
  p_stops.add_argument("--tz", type=int, help="input offset hours for naive --since/--until (e.g. -7 = PDT); display stays UTC")
  p_stops.set_defaults(fn=cmd_stops)

  args = ap.parse_args()
  args.fn(args)


if __name__ == "__main__":
  main()