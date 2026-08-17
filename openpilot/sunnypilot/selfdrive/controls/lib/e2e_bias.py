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


class E2EBiasController:
  REFRESH_PERIOD = int(PARAMS_UPDATE_PERIOD / DT_MDL)

  def __init__(self, params=None):
    self._params = params or Params()
    self._tick = 0
    self._e2e_bias = DEFAULT_BIAS

  def apply(self, a_target_e2e: float) -> float:
    """Add the speed bias to the model's desired acceleration."""
    self._tick += 1
    if self._tick % self.REFRESH_PERIOD == 0:
      self._refresh()
    # Fade the bias out as model braking grows: full while the model requests at or above
    # -bias, zero once it requests -2*bias or less, linear in between (0.3 m/s^2 blend width).
    # Stops stay identical to stock; the bias only trims speed-hold during neutral driving.
    bias_scale = np.clip((a_target_e2e + 2.0 * self._e2e_bias) / 0.3, 0.0, 1.0)
    return a_target_e2e + self._e2e_bias * bias_scale

  def _refresh(self):
    self._check_model_change()
    self._e2e_bias = self._strength_to_bias(self._params.get("LongitudinalE2EBias"))

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
      bundle = json.loads(raw_bundle)
      identity = f"{bundle['internalName']}:{bundle['generation']}"
    except (json.JSONDecodeError, KeyError, TypeError):
      return
    if self._params.get("LongitudinalE2EBiasTunedFor") != identity:
      self._params.put("LongitudinalE2EBias", "0")
      self._params.put("LongitudinalE2EBiasTunedFor", identity)
      self._e2e_bias = DEFAULT_BIAS
