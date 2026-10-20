"""Inspect camera messages in a replay window for sync problems. usage: check_window.py ROUTE SEGS START_S LEN_S"""
import os, sys
os.environ["REPLAY_START_S"], os.environ["REPLAY_MAX_S"] = sys.argv[3], sys.argv[4]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_replay_sp as h
msgs, idx = h.load(sys.argv[1], [int(s) for s in sys.argv[2].split(",")])
cams = {c: [(m.logMonoTime, getattr(m, c).frameId, getattr(m, c).timestampSof) for m in msgs if m.which() == c] for c in h.CAMS}
for c, v in cams.items():
  ids = [x[1] for x in v]
  gaps = [(a, b) for a, b in zip(ids, ids[1:]) if b != a + 1]
  print(f"{c}: {len(v)} msgs, frameIds {ids[0]}..{ids[-1]}, gaps {gaps[:5]}, in idx_map {sum(i in idx[c] for i in ids)}")
n, w = cams["narrowRoadCameraState"], cams["wideRoadCameraState"]
dsof = [abs(a[2] - b[2]) / 1e6 for a, b in zip(n, w)]
bad = [(i, round(d, 1)) for i, d in enumerate(dsof) if d > 10]
print(f"narrow/wide SOF mismatch >10ms (by order): {len(bad)} of {len(dsof)}; first: {bad[:5]}")
