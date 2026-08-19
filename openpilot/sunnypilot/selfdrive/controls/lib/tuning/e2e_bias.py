"""
E2E longitudinal personality (personal tuning).

North star: the E2E model drives by what it *predicts*, not by what the driver
wants. When the model is too conservative (surrenders speed early) or too
assertive (holds the go pedal past comfortable), this module trims that
confidence toward the driver's preference.

One `LongitudinalE2EBias` slider = one personality axis:
  positive — assertive: hold set speed on the open road, and ease off to keep a
             bigger gap when following a slower car (never stacking decel onto
             the model's own braking)
  zero     — stock model behaviour
  negative — conservative: prefer the model's own (lower) acceleration

Implementation is self-contained so the planner's merge surface against
upstream stays a single apply() call: speed bias + lead-gate, spacing bleed,
MPC braking-onset ramp (all driven by the same slider), hot-reloadable params,
and a model-change reset.
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
LEAD_FADE_HEADWAY_MIN = 1.0   # seconds headway: bias fully off below this
LEAD_FADE_HEADWAY_MAX = 2.5   # seconds headway: bias fully on above this
BLEED_HEADWAY_MIN = 1.0       # seconds headway: spacing bleed fades in above this
BLEED_HEADWAY_MAX = 3.0       # seconds headway: spacing bleed fades out above this
BLEED_FADE = 0.5              # headway (s) over which the bleed fades at band edges
BLEED_BRAKE_BLEND = 0.2       # m/s^2 window over which the bleed steps out of braking


def lead_gate(headway):
  """1.0 when far / no lead (bias full), 0.0 when a lead is close (bias off).

  The bias is a cruise behavior — hold set speed. On approach it must stand down so
  the model's natural early deceleration comes through instead of being held back."""
  if headway is None or np.isinf(headway):
    return 1.0
  return float(np.clip((headway - LEAD_FADE_HEADWAY_MIN) /
                       (LEAD_FADE_HEADWAY_MAX - LEAD_FADE_HEADWAY_MIN), 0.0, 1.0))


def bleed_factor(headway):
  """1.0 while following a lead (1-3s headway), 0.0 when far / no lead / braking.

  The spacing bleed eases off to maintain a bigger gap behind a slower car; it must
  step out near braking and when there is nothing to follow."""
  if headway is None or np.isinf(headway):
    return 0.0
  lo = np.clip((headway - BLEED_HEADWAY_MIN) / BLEED_FADE, 0.0, 1.0)
  hi = np.clip((BLEED_HEADWAY_MAX - headway) / BLEED_FADE, 0.0, 1.0)
  return float(lo * hi)


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
    self._a_mpc_prev = 0.0

  def reset_state(self):
    """Clear per-drive state. Call on engage/disengage so a re-engage can't
    slew the MPC ramp from a stale previous-cycle value."""
    self._a_mpc_prev = 0.0

  def apply(self, a_target_e2e: float, lead_drel: float | None = None, v_ego: float | None = None) -> float:
    """Add the speed bias + spacing bleed to the model's desired acceleration.

    One slider, two halves of the same personality:
    - positive strength: hold set speed on open road (bias), ease off to maintain a
      bigger gap when following a slower car (bleed)
    - negative strength: favor the model's own (lower) acceleration
    """
    self._tick += 1
    if self._tick % self.REFRESH_PERIOD == 0:
      self._refresh()
    b = self._e2e_bias
    if b == 0.0:
      return a_target_e2e
    headway = lead_drel / v_ego if (lead_drel is not None and v_ego) else None
    gate = lead_gate(headway)
    # Speed-hold: fades out as model braking grows (full at >= -|bias|, zero at <= -2*|bias|)
    # and stands down on lead approach. Stops stay identical to stock.
    bias_scale = np.clip((a_target_e2e + 2.0 * abs(b)) / abs(b), 0.0, 1.0) * gate
    # Spacing bleed: ease off to widen the gap when following; steps out of braking so
    # it never stacks decel onto a stop. Only the positive (assertive) side bleeds.
    if b > 0.0:
      bleed_safety = np.clip((a_target_e2e + BLEED_BRAKE_BLEND) / BLEED_BRAKE_BLEND, 0.0, 1.0)
      bias_scale -= bleed_factor(headway) * bleed_safety
    return a_target_e2e + b * bias_scale

  def apply_mpc(self, a_target_mpc: float, dt: float, bypass: bool = False) -> float:
    """Smooth the MPC's braking onset when a strength is set, so a lead ahead triggers
    a gradual ease-off instead of a stiff brake. Emergency (bypass) keeps the raw
    request — the MPC stays the hard floor. Correlated to the same strength slider.
    State (previous-cycle value) lives here, not in the planner."""
    if bypass or self._mpc_ramp is None:
      self._a_mpc_prev = a_target_mpc
      return a_target_mpc
    out = max(a_target_mpc, self._a_mpc_prev - self._mpc_ramp * dt)
    self._a_mpc_prev = out
    return out

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
