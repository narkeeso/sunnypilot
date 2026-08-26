"""
E2E speed bias tuning.

Self-contained controller: the bias algorithm, its hot-reloadable params, and the
model-change reset all live here so the planner's merge surface against upstream
is a single apply() call.

Mechanisms (all driven by one LongitudinalE2EBias slider, strength -20..20):

1. Speed bias + fade (apply): the personality axis. Adds b = strength * 0.01
   m/s^2 to the model's e2e accel request on open road (positive = hold speed).
   Stands down linearly as the model's OWN braking request grows: full bias at
   a >= 0, zero at a <= -|bias|. So the model's gentle early ease-off on
   approach is never argued with — the bias stops fighting the moment the model
   starts slowing. Pure e2e signal, no state, no gating. Negative bias (favor
   the model) fades out of braking the same way, never adding braking.

2. MPC braking-onset ramp (apply_mpc): smooths the lead-following path, which
   the e2e bias does not touch. Ramp rate 0.4-2.5 m/s^3 scaled by the same
   slider; caps how fast the MPC's brake request can change per frame so a lead
   ahead triggers a gradual ease-off instead of a stiff brake. Emergency
   (bypass = fcw/should_stop) keeps the raw request — the MPC stays the hard
   floor. Planner tracks a_mpc_prev.

3. Strength -> values mapping (_strength_to_bias, strength_to_mpc_ramp): one
   slider, two interpretations — bias magnitude and MPC ramp rate.

4. Model-change reset (_check_model_change): safety. When
   ModelManager_ActiveBundle identity changes (user swapped models), bias
   resets to 0 and LongitudinalE2EBiasTunedFor is stamped, so a bias tuned
   against one model's behavior never applies silently to another's.
"""

import json

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

DEFAULT_BIAS = 0.0
BIAS_STEPS = 20
BIAS_STEP_SIZE = 0.01
MPC_RAMP_MIN = 0.4   # m/s^3 at full strength: -0.3 m/s^2 over ~0.75s, clearly gentle
MPC_RAMP_MAX = 2.5   # m/s^3 at strength 1: -0.3 m/s^2 over ~0.12s, near stock
APPROACH_GRACE_STEPS = 20
APPROACH_GRACE_VREL = -1.0  # m/s: lead closing faster than this trips the stand-down


def strength_to_mpc_ramp(strength):
  """Ramp rate (m/s^3, lower = smoother MPC braking onset) for a given strength.
  None = no smoothing (stock). Reuses the same slider as the speed bias: both trim
  how much the model's natural feel is trusted over the robot's stiffness."""
  try:
    s = max(0, min(BIAS_STEPS, int(float(strength))))
  except (TypeError, ValueError):
    s = 0
  if s == 0:
    return None
  return float(np.interp(s, [1, BIAS_STEPS], [MPC_RAMP_MAX, MPC_RAMP_MIN]))


class E2EBiasController:
  REFRESH_PERIOD = int(PARAMS_UPDATE_PERIOD / DT_MDL)

  def __init__(self, params=None):
    self._params = params or Params()
    self._tick = 0
    self._e2e_bias = DEFAULT_BIAS
    self._mpc_ramp = None
    self._approach_grace = 0

  def apply(self, a_target_e2e: float, lead_closing: bool = False) -> float:
    """Add the speed bias to the model's desired acceleration.

    lead_closing: fused lead present and closing (radarState.leadOne.present
    AND vRel < -1). When true, the bias's speed hold is scaled down by the
    Approach Grace setting so the model's early ease-off on a closing lead
    reads through cleanly (the driver-like "bleed on closure" feel)."""
    self._tick += 1
    if self._tick % self.REFRESH_PERIOD == 0:
      self._refresh()
    b = self._e2e_bias
    if b == 0.0:
      return a_target_e2e
    if lead_closing and self._approach_grace > 0:
      b *= (1.0 - self._approach_grace / APPROACH_GRACE_STEPS)
      if b == 0.0:
        return a_target_e2e
    # Fade the bias out as soon as the model starts braking: full while the model
    # requests accel >= 0, zero once it requests -|bias| or less, linear in between.
    # The model's gentle early ease-off (a in [-|bias|, 0)) is left fully in charge —
    # the bias stops arguing with it the moment the approach begins. The blend width
    # scales with the bias so the slider's strength maps directly onto the hold.
    # Stops stay identical to stock; negative bias (favouring the model) fades out of
    # braking the same way so it never adds braking either.
    bias_scale = np.clip((a_target_e2e + abs(b)) / abs(b), 0.0, 1.0)
    return a_target_e2e + b * bias_scale

  def apply_mpc(self, a_target_mpc: float, a_mpc_prev: float, dt: float, bypass: bool = False) -> float:
    """Smooth the MPC's braking onset when a strength is set, so a lead ahead triggers
    a gradual ease-off instead of a stiff brake. Emergency (bypass) keeps the raw
    request — the MPC stays the hard floor. Correlated to the same strength slider."""
    if bypass or self._mpc_ramp is None:
      return a_target_mpc
    return max(a_target_mpc, a_mpc_prev - self._mpc_ramp * dt)

  def _refresh(self):
    self._check_model_change()
    strength = self._params.get("LongitudinalE2EBias")
    self._e2e_bias = self._strength_to_bias(strength)
    self._mpc_ramp = strength_to_mpc_ramp(strength)
    grace = self._params.get("LongitudinalApproachGrace")
    try:
      self._approach_grace = max(0, min(APPROACH_GRACE_STEPS, int(float(grace))))
    except (TypeError, ValueError):
      self._approach_grace = 0

  def _strength_to_bias(self, strength):
    try:
      steps = max(-BIAS_STEPS, min(BIAS_STEPS, int(float(strength))))
    except (TypeError, ValueError):
      steps = 0
    return steps * BIAS_STEP_SIZE

  def _check_model_change(self):
    raw_bundle = self._params.get("ModelManager_ActiveBundle")
    if not raw_bundle:
      return
    try:
      # JSON-type params come back already-parsed from Params.get(); accept both forms.
      bundle = json.loads(raw_bundle) if isinstance(raw_bundle, (str, bytes)) else raw_bundle
      identity = f"{bundle['internalName']}:{bundle['generation']}"
    except (json.JSONDecodeError, KeyError, TypeError):
      return
    if self._params.get("LongitudinalE2EBiasTunedFor") != identity:
      self._params.put("LongitudinalE2EBias", 0)
      self._params.put("LongitudinalApproachGrace", 0)
      self._params.put("LongitudinalE2EBiasTunedFor", identity)
      self._e2e_bias = DEFAULT_BIAS
      self._approach_grace = 0
