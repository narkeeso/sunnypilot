"""Message streaming from segment logs.

Gotchas baked in:
- segment dirs are `route--hash--seg`; logs match `startswith("rlog"/"qlog")`
  (`.endswith(".rlog.zst")` does NOT match)
- qlog = reduced (~20Hz, fast scans); rlog = full (~100Hz, slow — only for
  small windows)
- t = logMonoTime/1e9 seconds, RELATIVE to the first message of each segment
  (logMonoTime is monotonic since boot, not wall-clock)
- longitudinalPlanSource is a capnp enum: use `.raw`
- radar leadOne may be absent (dRel/vRel NaN) — ~19% of samples on highway
"""
import os

from openpilot.tools.lib.logreader import LogReader


def logfile(base, route, seg, fast=True):
  d = os.path.join(base, f"{route}--{seg}")
  if not os.path.isdir(d):
    return None
  pat = "qlog" if fast else "rlog"
  for f in os.listdir(d):
    if f.startswith(pat) and f.endswith(".zst"):
      return os.path.join(d, f)
  return None


def iter_msgs(base, route, segs, msgs=(), fast=True):
  """Yield (seg, t_s, msg). msgs filters by message type. t is seg-relative seconds."""
  for seg in segs:
    fn = logfile(base, route, seg, fast)
    if fn is None:
      continue
    t0 = None
    for msg in LogReader(fn):
      if msgs and msg.which() not in msgs:
        continue
      t = msg.logMonoTime / 1e9
      if t0 is None:
        t0 = t
      yield seg, t - t0, msg