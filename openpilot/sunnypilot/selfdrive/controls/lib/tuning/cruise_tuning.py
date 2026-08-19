"""
Overspeed-scaled set-speed approach decel (personal tuning).

Replaces the upstream fixed A_CRUISE_MIN floor with a ramp. Small reductions
(driver lowering the set speed a few mph) coast gently; large overspeeds
(downhill) get full authority quickly so the car doesn't run away on a descent.
"""

import numpy as np

A_CRUISE_MIN = -1.2        # absolute decel cap (m/s^2)
A_CRUISE_BLEED_MIN = -0.3  # gentle floor for small overspeeds (m/s^2)
A_CRUISE_SCALE = 0.15      # m/s^2 per m/s of overspeed in the gentle region
A_CRUISE_RAMP_OS = 2.5     # m/s over set speed: above this, authority ramps up
A_CRUISE_RAMP_GAIN = 0.8   # extra m/s^2 per m/s of overspeed past A_CRUISE_RAMP_OS


def decel_min(v_ego: float, v_cruise: float) -> float:
  """Cruise-candidate decel floor for the current overspeed, both in m/s."""
  overspeed = max(v_ego - v_cruise, 0.0)
  cap = A_CRUISE_SCALE * overspeed + A_CRUISE_RAMP_GAIN * max(overspeed - A_CRUISE_RAMP_OS, 0.0)
  return float(np.clip(-cap, A_CRUISE_MIN, A_CRUISE_BLEED_MIN))
