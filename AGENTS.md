# AGENTS.md

Fork-specific facts for agents working on narkeeso/sunnypilot. Read before reasoning about longitudinal behavior.

## The car is camera-only for longitudinal

- Car: `TOYOTA_RAV4_TSS2_2023`, a radar-ACC Toyota (`ToyotaFlags.RADAR_ACC` in opendbc `toyota/values.py`).
- With alpha longitudinal on, openpilot **silences the radar** (`DISABLE_RADAR` flag → `disable_ecu(... addr=0x750, sub_addr=0xf, ENABLE_RX_DISABLE_TX)` in opendbc `toyota/interface.py`). Stock Toyota AEB / Pre-Collision is off while openpilot controls speed.
- openpilot can't parse TSS2 radar-ACC tracks anyway (`radarUnavailable: True`).
- So **the MPC follows the camera's lead estimate**, never radar. `radarState.leadOne.radar` is always False (0% radar-matched across all logged drives). Do not reason about radar tracks, radar `aLeadTau` (1.5), or "radar takes over" for this car.
- Vision lead path: `get_RadarState_from_vision` in `openpilot/selfdrive/controls/radard.py`. `aLeadTau` is stock 0.3. Setting it to 0.0 (assume a braking lead keeps braking) was tried 2026-10-19 and reverted: no felt improvement, and it risks over-braking for leads slowing to turn off.
- Without alpha long, Toyota's own radar ACC controls speed and openpilot only steers.

## Longitudinal planner shape

E2E is always on (upstream removed Experimental Mode and DEC). The planner takes the **min** of three candidates every frame: MPC (camera lead), cruise (set speed), and the model's `desiredAcceleration`. Fork layer, in order:

1. `E2EBiasController.apply()` (`openpilot/sunnypilot/selfdrive/controls/lib/e2e_bias.py`): speed bias + Approach Grace on the model's accel.
2. `E2EBiasController.apply_mpc()`: MPC braking-onset ramp, tied to the bias slider.
3. Upstream `AccelBoost` (gas-override learned boost) stacks on top of the biased model accel.

Harsh stops behind cars come from late camera recognition of a stopping lead, then the MPC catching up. The limit is the model's anticipation, not planner tuning: an extra stop margin made it worse in simulation, and `aLeadTau` 0.0 didn't help on the road. Compare models' stopped-lead recognition instead.

## Logs and analysis

- `tools/driveq`: drive-log query CLI. Run on device with `/usr/local/venv/bin/python` and `PYTHONPATH=/data/openpilot`, from `/data/openpilot/tools`.
- qlogs do **not** contain `modelV2`; `longitudinalPlan` is ~2 Hz and `carState` ~10 Hz in qlogs. Use rlogs for anything needing the model's raw accel.
- Use `carState.aEgo` (measured) over `longitudinalPlan.aTarget` (requested) for felt-braking metrics, and drop stops where `brakePressed` or not `carControl.longActive`.
- `driveq`'s "radar lead" wording and `lead%` assume radar; on this car every lead is vision.

## Replaying drives through different models (`tools/model_replay_sp`)

Runs recorded drive footage through any downloaded sunnypilot model on the device, offroad, so models can be
compared on the exact same scenarios instead of by feel. Validated 2026-10-19: replaying the model that was live
on a drive reproduces its logged `desiredAcceleration` with correlation 1.000, mean error 0.03 m/s^2.

- Runs **on the device** (models are tinygrad pkls compiled for its GPU). Never touches the real active model,
  settings, or `/data/openpilot`: model choice goes through `ModelManager_ActiveBundle` inside process_replay's
  isolated `OPENPILOT_PREFIX`.
- **`/data/tmp` is wiped on every reboot.** Keep PyAV and results under `/data/media/0/model_replay/`.
- One-time setup: `/usr/local/venv/bin/python -m pip install --target /data/media/0/model_replay/pyav av` (device has
  no ffmpeg/PyAV; `REPLAY_PYAV_DIR` overrides the location).
- Routine for a new drive (on device, `cd /data/openpilot/tools/model_replay_sp`, `PYTHONPATH=/data/openpilot`, venv python):
  1. `list_stops.py --jobs ROUTE > jobs.txt` (one 35 s window per stop behind a car).
  2. Smoke pass (see below), then `run_batch.sh jobs.txt TSFM,TSFDOM,... /data/media/0/model_replay/out`
     (two windows at a time; ~200 s per model per window; resumable: rerun to redo only missing results).
  3. Copy `out/*/*.json` off the device and run `score_stops.py out/*/*.json`. Scorer checks: `test_score_stops.py`.
- **Before a batch, run `fetch_models.py <models...>`**: it sha256-checks every model file against the catalog and
  re-downloads mismatches, without activating anything (the model menu's download activates the model). Seen:
  a truncated TSFM pkl, and models stored only as `.pkl.chunkNNofMM` pieces from an older model manager (old
  builds; current modeld only loads the whole `.pkl`). A bad pkl makes modeld die while loading, and
  process_replay then waits forever; the harness caps each model at 10 min and skips existing results, so a
  rerun into the same OUT_DIR resumes.

**Before any long batch, run a smoke pass (~10 min):** every model on one short window, and every window with one
model, then the replay-vs-live check on a window from each drive. Most failures (bad pkl, step timeout, broken
frame pairs) only showed up hours into the first full batch because this was skipped.

Pitfalls that cost hours, all handled in the tool:
- process_replay **renumbers camera frameIds from 1**. Index video by position from the first frame in the window,
  not by the logged frameId (EncodeIdx), or every lookup misses and the model sees blank frames.
- Symptom of blank input: replayed output is frozen and **identical across different segments**. Always validate
  by replaying the drive's own live model and comparing to its logged output before trusting a comparison.
- sunnypilot runs `modeld_tinygrad` (`sunnypilot/modeld_v2`), which has no process_replay config; reuse `modeld`'s.
- Shell env: `PATH=/usr/local/venv/bin:$PATH` (modeld's launcher calls `python3`), and the `replayshim` on
  `PYTHONPATH`: offroad, cores 4-7 are offline so modeld's pin to core 7 fails.
- `initData` carries an older route-start logMonoTime; anchor windows on the first road camera frame.
- A time-based window cut can drop one camera's half of a road/wide frame pair; every pair is then 50 ms off
  and modeld never syncs them (process_replay TimeoutException on every model). Keep only frameIds both cameras
  have. Verify a window with `check_window.py ROUTE SEGS START_S LEN_S` (narrow/wide frameIds must match one-to-one).
- Run long jobs detached (`setsid`) on device, and prefer the LAN IP over the ssh.comma.ai proxy: under replay
  load the proxy drops sessions, sometimes silently, so a "stop" sent through it may never run. After stopping a
  batch, verify with `ps` that no replays remain before starting another; two batches at once overload the 4
  online cores and every model times out. Kill the replay's session (`pkill -s <pid>`) afterwards or orphaned
  `modeld_tinygrad` processes keep ~275 MB each.
- Never `pgrep -f`/`pkill -f <pattern>` inside an ssh command string: the pattern matches the ssh shell itself
  (waits never end; kills hit your own command). Put such logic in a script file and run that.
- `run_batch.sh` starts `onroad_guard.sh`, which stops all replay sessions if `IsOnroad` becomes 1, so a replay
  never competes with the live driving stack. Keep it that way for any other long replay job.
- process_replay's per-step timeout (30 s default) is too short for modeld's first step, which includes the tinygrad
  GPU compile; it fails some windows and not others (TimeoutException within the first ~20 messages, every model).
  The harness sets 180 s. This was first misdiagnosed as old drives being unreplayable; it isn't.
- Replay is open-loop: models see what the car actually did, fine for "when does it see the stop", not for feel.

- `Params.put` is non-blocking by default: a short script that exits right after `put` silently loses the write. Use
  `put(..., block=True)` (as the model manager does), then read the file back to confirm.

## Planned: per-drive "Enable openpilot" (stock car by default)

Goal: every drive starts fully stock (radar ACC + Pre-Collision/AEB untouched) so other drivers never need to change
settings; one UI button enables openpilot with alpha long for that drive only (one-way until ignition off).
Researched 2026-10-20 (code read on device, nothing built yet):
- CarParams, DISABLE_RADAR and the radar knockout (`CarInterface.init()` -> `disable_ecu`) happen once per onroad
  session at card start (`selfdrive/car/card.py`); panda safety mode is set once from CarParams. No live switch: the
  practical switch is the existing `OnroadCycleRequested` restart (~1 s offroad, `hardwared.py`).
- `OffroadMode` ("always offroad") gives a fully stock car but only resets at manager start (needs re-arming on
  ignition off), and offroad nothing reads CAN, so the trigger must be a UI button. Toyota port parses no
  SET/RES/main buttons; on RADAR_ACC cars not even the gap button.
- Enabling mid-drive silences the radar at that moment (stock ACC/AEB drop): gate the button to standstill/low speed.
- `CarInterface.deinit()` is never called anywhere; the radar returns only when Tester Present stops (UDS session
  timeout). Before building, test parked: silence radar (alpha long onroad), then stop/unplug the comma, and check
  stock ACC/PCS return without dash warnings or DTCs.
- Sketch (~40-80 lines, fork-only): param that re-arms `OffroadMode` on ignition off; button clears it and sets a
  per-drive alpha-long opt-in (cleared on ignition off, not on offroad, or the 1 s cycle wipes it); card reads
  `AlphaLongitudinalEnabled or opt-in`. Existing fallback: `ToyotaEnforceStockLongitudinal` (global, not per drive).

## Repo and device

- Personal branch: `staging-personal`. Upstream: `sunnypilot/sunnypilot` `master`. Sync by merging upstream into a `sync-YYYYMMDD` branch, then fast-forward.
- Git LFS objects live on sunnypilot's Hugging Face store (`.lfsconfig`); pushes to the fork don't need to upload them (`git push --no-verify` if the LFS hook asks for HF credentials).
- Upstream syncs can bump the required AGNOS version (`launch_env.sh`); the device then installs the OS on boot before building.
- Device SSH: `comma@<lan-ip>` at home, or via `ssh.comma.ai` proxy (see `docs/how-to/connect-to-comma.md`), keys from GitHub user `narkeeso`.
