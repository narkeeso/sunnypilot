# Replay-only shim: offroad, AGNOS takes cores 4-7 offline, so modeld pinning itself to core 7 fails.
import os
_set_affinity = os.sched_setaffinity
def _sched_setaffinity(pid, cores):
  try:
    _set_affinity(pid, cores)
  except OSError:
    pass
os.sched_setaffinity = _sched_setaffinity

if os.environ.get("REPLAY_COPYFRAMES") == "1":
  # Replay-only: read each frame from the shared-memory file model_replay_sp.py writes, instead of the
  # ION VisionIPC buffer the replay fills from the CPU without a cache sync. Possibly unnecessary since the
  # frameId fix in model_replay_sp.py, but the validated runs used it.
  import importlib.abc
  import importlib.util
  import sys

  def _patch_adapters(mod):
    import numpy as np
    from tinygrad import Tensor
    orig = mod.LegacyModelAdapter.copy_frames
    def copy_frames(self, bufs):
      if self.chestnut:
        return orig(self, bufs)
      for key in bufs.keys():
        yuv_size = self.frame_buf_params[key][3]
        path = f"/dev/shm/replay_{os.environ.get('REPLAY_TAG', 'default')}_{key}.bin"
        host = np.fromfile(path, dtype=np.uint8, count=yuv_size) if os.path.exists(path) else np.zeros(yuv_size, dtype=np.uint8)
        self.full_frames[key] = Tensor(host, device=self.WARP_DEV).realize()
    mod.LegacyModelAdapter.copy_frames = copy_frames

  class _AdapterHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
      if name != "openpilot.sunnypilot.modeld_v2.model_adapters":
        return None
      sys.meta_path.remove(self)
      spec = importlib.util.find_spec(name)
      exec_module = spec.loader.exec_module
      def patched_exec(module):
        exec_module(module)
        _patch_adapters(module)
      spec.loader.exec_module = patched_exec
      return spec

  sys.meta_path.insert(0, _AdapterHook())
