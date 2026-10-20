"""Route discovery and time filtering for comma realdata (device or local cache).

Facts baked in from real-world use:
- realdata holds FLAT segment dirs `route--hash--seg` (no route grouping)
- device clock is UTC; naive ISO inputs are interpreted as UTC
- segment dir mtimes are the wall-clock anchor (epoch); outliers from an
  unsynced clock are ignored for start/end, never dropped as segments
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class Route:
  prefix: str
  segs: list
  start_epoch: float
  end_epoch: float


def scan(base):
  """Group flat `route--hash--seg` dirs into Route objects (all segments kept).

  start/end come from segment dir mtimes near the median mtime: a segment
  recorded before the clock synced can carry a months-off mtime (seen on
  000001d4: seg 29 dated Jul 28), and taking min/max over it would make the
  route overlap every --since/--until window back to that date."""
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
    # a real element (not a midpoint between two) so `near` is never empty
    med = sorted(mtimes)[len(mtimes) // 2]
    # segments are 60 s apart, so the whole route fits within ~75 s x n of the median
    near = [t for t in mtimes if abs(t - med) < 75.0 * len(mtimes)]
    r.start_epoch = min(near)
    r.end_epoch = max(near)
  return routes


def to_epoch(iso, tz=None):
  """ISO string -> epoch seconds. Explicit offset or Z wins; naive strings are
  UTC by default, or interpreted as `tz` (offset in hours, e.g. -7 for PDT)
  when given — so "2026-08-22T11:00:00 --tz -7" means 11am Pacific."""
  dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone(timedelta(hours=tz)) if tz is not None else timezone.utc)
  return dt.timestamp()


def filter_routes(routes, since=None, until=None, recent=None, prefixes=None):
  """Filter by explicit route prefixes, wall-clock window (epoch), or most recent N."""
  if prefixes:
    missing = [p for p in prefixes if p not in routes]
    if missing:
      raise SystemExit(f"route not found: {', '.join(missing)}")
    return [routes[p] for p in prefixes]
  out = sorted(routes.values(), key=lambda r: r.start_epoch)
  if since is not None:
    out = [r for r in out if r.end_epoch >= since]
  if until is not None:
    out = [r for r in out if r.start_epoch <= until]
  if recent is not None:
    out = out[-recent:]
  return out