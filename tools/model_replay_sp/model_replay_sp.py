"""Replay recorded drive segments through sunnypilot's modeld_tinygrad with a chosen model (runs on device).

usage: model_replay_sp.py ROUTE SEGS(comma list) MODEL_SHORT_NAMES(comma list) OUT_DIR

The model is chosen via ModelManager_ActiveBundle inside process_replay's isolated
params prefix, so the device's real settings are untouched. Video is decoded with PyAV
(pip --target dir in REPLAY_PYAV_DIR, default /data/media/0/model_replay/pyav; /data/tmp is wiped on reboot), frames mapped frameId -> (segment, index) via *EncodeIdx.
Writes one JSON per model: per modelV2 frame the model's accel request and lead estimate.
"""
import copy
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.environ.get("REPLAY_PYAV_DIR", "/data/media/0/model_replay/pyav"))  # the device has no PyAV/ffmpeg
import av  # noqa: E402
import numpy as np  # noqa: E402

from openpilot.common.params import Params  # noqa: E402
from openpilot.tools.lib.logreader import LogReader  # noqa: E402
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process  # noqa: E402
from openpilot.sunnypilot.models.fetcher import ModelParser  # noqa: E402

BASE = os.environ.get("REPLAY_LOG_DIR", "/data/media/0/realdata")
MODEL_TIME_LIMIT_S = 600  # a healthy 35 s window takes ~200 s
# Each padded frame also goes to plain shared memory, which the replay shim's copy_frames reads instead of the
# ION VisionIPC buffer (filled from the CPU without a cache sync). Tagged so parallel replays don't collide.
TAG = os.environ.get("REPLAY_TAG", "default")
SHM = {"narrowRoadCameraState": f"/dev/shm/replay_{TAG}_img.bin", "wideRoadCameraState": f"/dev/shm/replay_{TAG}_big_img.bin"}
CAMS = {  # camera state service -> (video file, encode index service)
  "narrowRoadCameraState": ("fcamera.hevc", "narrowRoadEncodeIdx"),
  "wideRoadCameraState": ("ecamera.hevc", "wideRoadEncodeIdx"),
}


class SegFrames:
  """FrameReader stand-in: get(frameId) -> flat nv12 bytes, decoding sequentially per segment file."""
  pix_fmt = "nv12"

  def __init__(self, paths, idx_map, shm_path):
    self.paths, self.idx_map, self.shm_path = paths, idx_map, shm_path
    with av.open(paths[min(paths)]) as c:
      s = c.streams.video[0]
      self.w, self.h = s.codec_context.width, s.codec_context.height
    self.cur_seg, self.it, self.pos, self.last = None, None, -1, None
    self.blank = np.zeros(self.w * self.h * 3 // 2, dtype=np.uint8)

  def _open(self, seg):
    self.container = av.open(self.paths[seg])
    self.container.streams.video[0].thread_type = "AUTO"
    self.it, self.pos, self.cur_seg = self.container.decode(video=0), -1, seg

  def get(self, frame_id):
    # process_replay renumbers camera frameIds from the start of the replay, so fall back to position
    if not hasattr(self, "keys"):
      self.keys, self.first_req = sorted(self.idx_map), frame_id
    if frame_id not in self.idx_map and 0 <= frame_id - self.first_req < len(self.keys):
      frame_id = self.keys[frame_id - self.first_req]
    loc = self.idx_map.get(frame_id)
    if loc is None:
      img = self.last if self.last is not None else self.blank
      self._publish(img)
      return img
    seg, idx = loc
    if seg != self.cur_seg or idx <= self.pos:
      self._open(seg)
    frame = None
    while self.pos < idx:
      try:
        frame = next(self.it)
      except StopIteration:
        img = self.last if self.last is not None else self.blank
        self._publish(img)
        return img
      self.pos += 1
    self.last = frame.to_ndarray(format="nv12").reshape(-1)
    self._publish(self.last)
    return self.last

  def _publish(self, img):
    # same padding process_replay applies before vipc send
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    w, h = self.w, self.h
    stride, y_height, _, yuv_size = get_nv12_info(w, h)
    out = np.zeros(yuv_size, dtype=np.uint8)
    plane = out[:stride * (y_height + h // 2)].reshape(-1, stride)
    plane[:h, :w] = img[:h * w].reshape(h, w)
    uv0 = stride * y_height // stride
    plane[uv0:uv0 + h // 2, :w] = img[h * w:].reshape(h // 2, w)
    out.tofile(self.shm_path)


def load(route, segs):
  msgs, idx_maps = [], {cam: {} for cam in CAMS}
  for seg in segs:
    rl = [f for f in os.listdir(f"{BASE}/{route}--{seg}") if f.startswith("rlog")][0]
    for m in LogReader(f"{BASE}/{route}--{seg}/{rl}"):
      msgs.append(m)
      w = m.which()
      for cam, (_, idx_svc) in CAMS.items():
        if w == idx_svc:
          e = getattr(m, w)
          idx_maps[cam][e.frameId] = (seg, e.segmentId)
  msgs.sort(key=lambda m: m.logMonoTime)
  if os.environ.get("REPLAY_MAX_S"):  # replay only a window: REPLAY_START_S (default 0) + REPLAY_MAX_S seconds
    t0 = next(m.logMonoTime for m in msgs if m.which() == 'narrowRoadCameraState')  # initData carries an older route-start time
    t_start = t0 + float(os.environ.get("REPLAY_START_S", 0)) * 1e9
    t_end = t_start + float(os.environ["REPLAY_MAX_S"]) * 1e9
    # keep startup messages, but only once: a second segment's initData/carParams in the replay stalls modeld
    seen_startup = set()
    def keep(m):
      w = m.which()
      if w in ('carParams', 'initData'):
        if w in seen_startup:
          return False
        seen_startup.add(w)
        return True
      return t_start <= m.logMonoTime < t_end
    msgs = [m for m in msgs if keep(m)]
    # a time cut can drop one camera's half of a frame pair; then every pair is off by a frame (50 ms) and
    # modeld never syncs them, so keep only frameIds both cameras have
    both = set.intersection(*({getattr(m, cam).frameId for m in msgs if m.which() == cam} for cam in CAMS))
    msgs = [m for m in msgs if m.which() not in CAMS or getattr(m, m.which()).frameId in both]
  first_kept = {cam: min(getattr(m, cam).frameId for m in msgs if m.which() == cam) for cam in CAMS}
  idx_maps = {cam: {k: v for k, v in idx_maps[cam].items() if k >= first_kept[cam]} for cam in CAMS}
  return msgs, idx_maps


def make_frs(route, segs, idx_maps):
  return {cam: SegFrames({s: f"{BASE}/{route}--{s}/{CAMS[cam][0]}" for s in segs}, idx_maps[cam], SHM[cam]) for cam in CAMS}


def bundle_param(short_name):
  for b in ModelParser.parse_models(Params().get("ModelManager_ModelsCache")):
    if b.internalName == short_name:
      return b.to_dict()
  raise SystemExit(f"model {short_name} not in catalog")


def main():
  route, segs, models, out_dir = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3].split(","), sys.argv[4]
  os.makedirs(out_dir, exist_ok=True)
  msgs, idx_maps = load(route, segs)
  cfg = copy.deepcopy(get_process_config("modeld"))
  cfg.proc_name = "modeld_tinygrad"
  cfg.timeout = 180  # the first step includes tinygrad GPU compile, which can exceed process_replay's default 30 s
  # replayed modelV2 frames come back renumbered from 1, so align them to these by order
  logged = [(m.modelV2.frameId, m.modelV2.action.desiredAcceleration, m.logMonoTime) for m in msgs if m.which() == "modelV2"]
  car = [(m.logMonoTime, m.carState.vEgo, m.carState.brakePressed) for m in msgs if m.which() == "carState"][::10]
  def _too_long(signum, frame):
    raise TimeoutError("model replay exceeded its time limit")
  signal.signal(signal.SIGALRM, _too_long)

  for short in models:
    out_path = f"{out_dir}/{route}_{'-'.join(map(str, segs))}_{short}.json"
    if os.path.exists(out_path):  # resumable: keep results from an earlier run
      print(f"{short}: already done", flush=True)
      continue
    frs = make_frs(route, segs, idx_maps)  # fresh decoders per model
    t0 = time.monotonic()
    # process_replay waits forever if modeld dies while loading (e.g. a corrupt pkl), so cap each model
    signal.alarm(MODEL_TIME_LIMIT_S)
    try:
      out = replay_process(cfg, msgs, frs=frs, custom_params={"ModelManager_ActiveBundle": bundle_param(short)},
                           disable_progress=True)
    except Exception as e:  # e.g. model pkl missing or incompatible: skip it, keep the other models
      print(f"{short}: FAILED {type(e).__name__}: {e}", flush=True)
      continue
    finally:
      signal.alarm(0)
    rows = []
    for m in out:
      if m.which() != "modelV2":
        continue
      mv = m.modelV2
      ld = mv.leadsV3[0] if len(mv.leadsV3) else None
      rows.append({
        "mono": m.logMonoTime, "frameId": mv.frameId, "aReq": mv.action.desiredAcceleration,
        "shouldStop": mv.action.shouldStop,
        "leadProb": ld.prob if ld else None, "leadX": ld.x[0] if ld else None,
        "leadV": ld.v[0] if ld else None, "leadA": ld.a[0] if ld else None,
      })
    dt = time.monotonic() - t0
    json.dump({"route": route, "segs": segs, "model": short, "rows": rows, "logged": logged, "car": car, "secs": dt},
              open(out_path, "w"))
    print(f"{short}: {len(rows)} modelV2 frames in {dt:.0f}s", flush=True)


if __name__ == "__main__":
  main()
