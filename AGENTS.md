# AGENTS.md

Fork-specific facts for agents working on narkeeso/sunnypilot. Read before reasoning about longitudinal behavior.

## The car is camera-only for longitudinal

- Car: `TOYOTA_RAV4_TSS2_2023`, a radar-ACC Toyota (`ToyotaFlags.RADAR_ACC` in opendbc `toyota/values.py`).
- With alpha longitudinal on, openpilot **silences the radar** (`DISABLE_RADAR` flag → `disable_ecu(... addr=0x750, sub_addr=0xf, ENABLE_RX_DISABLE_TX)` in opendbc `toyota/interface.py`). Stock Toyota AEB / Pre-Collision is off while openpilot controls speed.
- openpilot can't parse TSS2 radar-ACC tracks anyway (`radarUnavailable: True`).
- So **the MPC follows the camera's lead estimate**, never radar. `radarState.leadOne.radar` is always False (0% radar-matched across all logged drives). Do not reason about radar tracks, radar `aLeadTau` (1.5), or "radar takes over" for this car.
- Vision lead path: `get_RadarState_from_vision` in `openpilot/selfdrive/controls/radard.py`. This fork sets its `aLeadTau` to 0.0 (stock 0.3) so the MPC assumes a braking lead keeps braking.
- Without alpha long, Toyota's own radar ACC controls speed and openpilot only steers.

## Longitudinal planner shape

E2E is always on (upstream removed Experimental Mode and DEC). The planner takes the **min** of three candidates every frame: MPC (camera lead), cruise (set speed), and the model's `desiredAcceleration`. Fork layer, in order:

1. `E2EBiasController.apply()` (`openpilot/sunnypilot/selfdrive/controls/lib/e2e_bias.py`): speed bias + Approach Grace on the model's accel.
2. `E2EBiasController.apply_mpc()`: MPC braking-onset ramp, tied to the bias slider.
3. Upstream `AccelBoost` (gas-override learned boost) stacks on top of the biased model accel.

Harsh stops behind cars come from late camera recognition of a stopping lead, then the MPC catching up. Moving the stop target closer (extra stop margin) made it worse in simulation.

## Logs and analysis

- `tools/driveq`: drive-log query CLI. Run on device with `/usr/local/venv/bin/python` and `PYTHONPATH=/data/openpilot`, from `/data/openpilot/tools`.
- qlogs do **not** contain `modelV2`; `longitudinalPlan` is ~2 Hz and `carState` ~10 Hz in qlogs. Use rlogs for anything needing the model's raw accel.
- Use `carState.aEgo` (measured) over `longitudinalPlan.aTarget` (requested) for felt-braking metrics, and drop stops where `brakePressed` or not `carControl.longActive`.
- `driveq`'s "radar lead" wording and `lead%` assume radar; on this car every lead is vision.

## Repo and device

- Personal branch: `staging-personal`. Upstream: `sunnypilot/sunnypilot` `master`. Sync by merging upstream into a `sync-YYYYMMDD` branch, then fast-forward.
- Git LFS objects live on sunnypilot's Hugging Face store (`.lfsconfig`); pushes to the fork don't need to upload them (`git push --no-verify` if the LFS hook asks for HF credentials).
- Upstream syncs can bump the required AGNOS version (`launch_env.sh`); the device then installs the OS on boot before building.
- Device SSH: `comma@<lan-ip>` at home, or via `ssh.comma.ai` proxy (see `docs/how-to/connect-to-comma.md`), keys from GitHub user `narkeeso`.
