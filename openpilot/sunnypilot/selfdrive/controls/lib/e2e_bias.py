"""
E2E speed bias tuning.

Self-contained controller: the bias algorithm, its hot-reloadable params, and the
model-change reset all live here so the planner's merge surface against upstream
is a single apply() call.
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

  def apply(self, a_target_e2e: float) -> float:
    """Add the speed bias to the model's desired acceleration."""
    self._tick += 1
    if self._tick % self.REFRESH_PERIOD == 0:
      self._refresh()
    b = self._e2e_bias
    if b == 0.0:
      return a_target_e2e
    # Fade the bias out as model braking grows: full while the model requests at or above
    # -|bias|, zero once it requests -2*|bias| or less, linear in between. The blend width
    # scales with the bias so the slider's strength maps directly onto the hold. Stops stay
    # identical to stock; negative bias (favouring the model) fades out of braking the same
    # way so it never adds braking either.
    bias_scale = np.clip((a_target_e2e + 2.0 * abs(b)) / abs(b), 0.0, 1.0)
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
      self._params.put("LongitudinalE2EBiasTunedFor", identity)
      self._e2e_bias = DEFAULT_BIAS
