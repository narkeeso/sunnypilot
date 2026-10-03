"""Route discovery and time filtering for comma realdata (device or local cache).

Facts baked in from real-world use:
- realdata holds FLAT segment dirs `route--hash--seg` (no route grouping)
- device clock is UTC; naive ISO inputs are interpreted as UTC
- segment dir mtimes are the reliable wall-clock anchor (epoch)
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from openpilot.tools.lib.logreader import LogReader

from driveq.reader import logfile


@dataclass
class Route:
  prefix: str
  segs: list
  start_epoch: float
  end_epoch: float


def _seg_wallclock(base, route, seg):
  """Wall-clock (epoch) from the Clocks message inside the log. Expensive
  (~0.25s/segment: full read until the clocks message) — only for disambiguating
  routes whose mtimes are inconsistent. Falls back to dir mtime."""
  fn = logfile(base, route, seg, fast=True)
  if fn is not None:
    try:
      for msg in LogReader(fn):
        if msg.which() == "clocks":
          nanos = msg.clocks.wallTimeNanos
          if nanos > 0:
            return nanos / 1e9
    except (ValueError, OSError):
      pass
  return os.stat(os.path.join(base, f"{route}--{seg}")).st_mtime


def scan(base):
  """Group flat `route--hash--seg` dirs into Route objects. FAST PATH: segment
  dir mtimes (instant) + median guard, which is correct except for the rare
  stale-dir route (same hash, old recording — seen in realdata). For those,
  read the Clocks message from first/mid/last segs to disambiguate."""
  routes = {}
  for d in os.listdir(base):
    parts = d.split("--")
    if len(parts) != 3:
      continue
    prefix = f"{parts[0]}--{parts[1]}"
    try:
      seg = int(parts[2])
    except ValueError:
      continue
    path = os.path.join(base, d)
    if not os.path.isdir(path):
      continue
    r = routes.get(prefix)
    if r is None:
      r = Route(prefix, [], None, None)
      routes[prefix] = r
    r.segs.append(seg)
  for r in routes.values():
    r.segs.sort()
    mtimes = [os.stat(os.path.join(base, f"{r.prefix}--{s}")).st_mtime for s in r.segs]
    if len(mtimes) >= 3:
      med = float(np.median(mtimes))
      spread = max(mtimes) - min(mtimes)
      if spread < 7200.0:
        good = [(t, s) for t, s in zip(mtimes, r.segs)]
      else:
        # suspicious: some segs are weeks out of family — use log clocks to
        # find the true cluster (fast only on the 3 probes)
        probes = {r.segs[0], r.segs[len(r.segs) // 2], r.segs[-1]}
        wall = {s: _seg_wallclock(base, r.prefix, s) for s in probes}
        med = float(np.median(list(wall.values())))
        good = [(t, s) for t, s in zip(mtimes, r.segs) if abs(t - med) < 7200.0]
        if len(good) < 2:
          # clocks disagreed too; fall back to mtimes minus the worst outlier
          good = [(t, s) for t, s in zip(mtimes, r.segs) if abs(t - med) < 3 * spread]
    else:
      good = [(t, s) for t, s in zip(mtimes, r.segs)]
    r.segs = [s for _, s in good]
    r.start_epoch = min(t for t, _ in good)
    r.end_epoch = max(t for t, _ in good)
  return routes


def to_epoch(iso, tz=None):
  """ISO string -> epoch seconds. Explicit offset or Z wins; naive strings are
  UTC by default, or interpreted as `tz` (offset in hours, e.g. -7 for PDT)
  when given — so "2026-08-22T11:00:00 --tz -7" means 11am Pacific."""
  dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone(timedelta(hours=tz)) if tz is not None else timezone.utc)
  return dt.timestamp()


def filter_routes(routes, since=None, until=None, recent=None):
  """Filter by wall-clock window (epoch), or take the most recent N."""
  out = sorted(routes.values(), key=lambda r: r.start_epoch)
  if since is not None:
    out = [r for r in out if r.end_epoch >= since]
  if until is not None:
    out = [r for r in out if r.start_epoch <= until]
  if recent is not None:
    out = out[-recent:]
  return out