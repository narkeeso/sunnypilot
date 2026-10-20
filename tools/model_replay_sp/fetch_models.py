"""Download/verify catalog models by short name (sha256), without activating them. Re-downloads any mismatch.
usage: fetch_models.py SHORT_NAME [...]   (run on device with PYTHONPATH=/data/openpilot)
"""
import hashlib, os, sys, requests
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
params = Params()
root = Paths.model_root()
bundles = params.get("ModelManager_ModelsCache")["bundles"]
active_before = params.get("ModelManager_ActiveBundle")
for x in bundles:
  if x["short_name"] not in sys.argv[1:]:
    continue
  for m in x["models"]:
    a = m["artifact"]; url = a["download_uri"]["url"]; sha = a["download_uri"]["sha256"].lower()
    path = os.path.join(root, a["file_name"])
    if os.path.exists(path) and hashlib.file_digest(open(path, "rb"), "sha256").hexdigest() == sha:
      print(x["short_name"], "already present", flush=True); continue
    part = path + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
      r.raise_for_status()
      with open(part, "wb") as f:
        for chunk in r.iter_content(1 << 20):
          f.write(chunk)
    if hashlib.file_digest(open(part, "rb"), "sha256").hexdigest() != sha:
      os.remove(part); print(x["short_name"], "HASH MISMATCH, removed", flush=True); continue
    os.rename(part, path)
    print(x["short_name"], "ok", round(os.path.getsize(path) / 1e6), "MB", flush=True)
print("active bundle unchanged:", params.get("ModelManager_ActiveBundle") == active_before)
