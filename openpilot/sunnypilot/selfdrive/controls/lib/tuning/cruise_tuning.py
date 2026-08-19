"""
Overspeed-scaled cruise decel (personal tuning).

North star: lowering the set speed should feel like easing off the throttle,
not hitting the brakes. Stock saturated a fixed -1.2 m/s^2 the moment speed
passed the set speed, which read as a harsh brake when the driver reduced the
set limit (logged -1.13 at ~40 mph on a 2 mph reduction). But a descent still
needs authority, or the car creeps well past the limit you set.

This module ties braking authority to *how far* over the set speed the car is:
  small overspeeds (driver nudges the set speed down) — gentle, >= -0.3
  large overspeeds (a long downhill)                 — ramps to full -1.2 by
    ~10 mph over, so the car holds the set limit on sustained grades
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
