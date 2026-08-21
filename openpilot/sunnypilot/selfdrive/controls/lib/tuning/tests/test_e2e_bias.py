import json
import unittest

from openpilot.sunnypilot.selfdrive.controls.lib.tuning.e2e_bias import (
  DEFAULT_BIAS, E2EBiasController, bleed_factor, lead_gate, ramp_rate, strength_to_mpc_ramp,
)


class MockParams:
  def __init__(self, values=None):
    self.values = values or {}

  def get(self, key):
    return self.values.get(key)

  def put(self, key, value):
    self.values[key] = value


def make_controller(bias, params_values=None, strength=None):
  values = dict(params_values or {})
  if strength is not None:
    values.setdefault("LongitudinalE2EBias", str(strength))
  c = E2EBiasController(params=MockParams(values))
  c._e2e_bias = bias
  return c


class TestE2EBiasController(unittest.TestCase):
  def test_full_bias_above_fade_window(self):
    c = make_controller(bias=0.15)
    self.assertAlmostEqual(c.apply(0.5), 0.65, places=6)

  def test_no_bias_at_2x_braking(self):
    c = make_controller(bias=0.15)
    self.assertAlmostEqual(c.apply(-0.3), -0.3, places=6)

  def test_linear_fade_midpoint(self):
    c = make_controller(bias=0.15)
    # at -1.5 * bias the blend is halfway through, so half of the bias is added
    self.assertAlmostEqual(c.apply(-0.225), -0.225 + 0.15 * 0.5, places=6)

  def test_full_bias_at_minus_bias(self):
    c = make_controller(bias=0.15)
    self.assertAlmostEqual(c.apply(-0.15), 0.0, places=6)

  def test_zero_bias_at_minus_2bias(self):
    c = make_controller(bias=0.13)
    self.assertAlmostEqual(c.apply(-0.26), -0.26, places=6)

  def test_fade_width_scales_with_bias(self):
    # stronger bias must reach full hold deeper into the bleed region
    strong = make_controller(bias=0.2)
    weak = make_controller(bias=0.05)
    # at -0.1 bleed: strong is still inside its full window, weak is already fully faded
    self.assertAlmostEqual(strong.apply(-0.1), -0.1 + 0.2, places=6)
    self.assertAlmostEqual(weak.apply(-0.1), -0.1, places=6)
    # at the same half-fade point relative to each: -1.5*bias
    self.assertAlmostEqual(strong.apply(-0.3), -0.3 + 0.1, places=6)
    self.assertAlmostEqual(weak.apply(-0.075), -0.075 + 0.025, places=6)

  def test_zero_bias_is_noop(self):
    c = make_controller(bias=0.0)
    self.assertAlmostEqual(c.apply(-0.3), -0.3, places=6)
    self.assertAlmostEqual(c.apply(0.5), 0.5, places=6)

  def test_negative_bias_subtracts(self):
    c = make_controller(bias=-0.1)
    self.assertAlmostEqual(c.apply(0.5), 0.4, places=6)

  def test_negative_bias_never_touches_braking(self):
    c = make_controller(bias=-0.1)
    self.assertAlmostEqual(c.apply(-0.2), -0.2, places=6)

  def test_param_refresh_reads_strength(self):
    c = make_controller(bias=0.0, params_values={"LongitudinalE2EBias": "20"})
    c._tick = c.REFRESH_PERIOD - 1
    self.assertAlmostEqual(c.apply(0.5), 0.7, places=6)

  def test_strength_conversion(self):
    c = E2EBiasController(params=MockParams())
    self.assertAlmostEqual(c._strength_to_bias("20"), 0.2, places=6)
    self.assertAlmostEqual(c._strength_to_bias("-20"), -0.2, places=6)
    self.assertAlmostEqual(c._strength_to_bias("0"), 0.0, places=6)
    self.assertAlmostEqual(c._strength_to_bias("5"), 0.05, places=6)

  def test_strength_clamped(self):
    c = E2EBiasController(params=MockParams())
    self.assertAlmostEqual(c._strength_to_bias("50"), 0.2, places=6)
    self.assertAlmostEqual(c._strength_to_bias("-50"), -0.2, places=6)
    self.assertAlmostEqual(c._strength_to_bias("garbage"), 0.0, places=6)
    self.assertAlmostEqual(c._strength_to_bias(None), 0.0, places=6)

  def test_mpc_ramp_off_at_zero(self):
    assert strength_to_mpc_ramp("0") is None
    assert strength_to_mpc_ramp(None) is None
    assert strength_to_mpc_ramp("garbage") is None

  def test_mpc_ramp_scales_with_strength(self):
    high = strength_to_mpc_ramp("20")
    low = strength_to_mpc_ramp("1")
    assert high is not None and low is not None
    assert high < low
    self.assertAlmostEqual(high, 0.4, places=6)
    self.assertAlmostEqual(low, 2.5, places=6)

  def test_apply_mpc_smooths_braking(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 20  # base rate 0.4 m/s^3 (stock-ish softening); no lead -> no bubble
    c._a_mpc_prev = -0.05
    # mpc wants -0.3, previous output was -0.05: ramped to -0.05 - rate*dt (v low so speed floor < base)
    self.assertAlmostEqual(c.apply_mpc(-0.3, v_ego=5.0, dt=0.05), -0.05 - 0.4 * 0.05, places=6)

  def test_apply_mpc_emergency_bypasses(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 20
    c._a_mpc_prev = -0.05
    self.assertAlmostEqual(c.apply_mpc(-1.0, v_ego=10.0, dt=0.05, bypass=True), -1.0, places=6)

  def test_apply_mpc_no_ramp_is_stock(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 0
    c._a_mpc_prev = -0.05
    self.assertAlmostEqual(c.apply_mpc(-0.3, v_ego=10.0, dt=0.05), -0.3, places=6)

  def test_apply_mpc_release_not_limited(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 8
    c._a_mpc_prev = -0.3
    self.assertAlmostEqual(c.apply_mpc(-0.1, v_ego=10.0, dt=0.05), -0.1, places=6)

  def test_apply_mpc_safety_bubble_inside_2s_is_raw(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 20
    c._a_mpc_prev = 0.0
    # lead at 1.5s time-to-contact: ramp bypassed, raw authority immediately
    v = 15.0
    for _ in range(3):
      out = c.apply_mpc(-1.7, v_ego=v, lead_drel=1.5 * v, dt=0.05)
    self.assertAlmostEqual(out, -1.7, places=6)

  def test_ramp_rate_stock_and_bubble(self):
    assert ramp_rate(0, 20.0, None) is None
    # inside 2s TTC of a real lead -> None
    assert ramp_rate(20, 15.0, 20.0) is None   # ttc = 1.33s

  def test_ramp_rate_speed_floor_raises_rate(self):
    # no lead: rate = max(base, v*1.5/20). higher speed -> higher floor
    lo = ramp_rate(7, 10.0, None)
    hi = ramp_rate(7, 30.0, None)
    assert lo is not None and hi is not None
    self.assertAlmostEqual(lo, max(strength_to_mpc_ramp(7), 10 * 1.5 / 20.0), places=6)
    assert hi > lo

  def test_ramp_rate_far_lead_no_urgency(self):
    # lead far enough that TTC >= RAMP_TTC_REF -> urgency = 1 -> equals no-lead
    no_lead = ramp_rate(7, 10.0, None)
    far = ramp_rate(7, 10.0, 60.0)   # ttc = 6s
    self.assertAlmostEqual(no_lead, far, places=6)

  def test_apply_model_smooths_phantom(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 20
    c._a_model_prev = 0.0
    # phantom: model slams -1.7 with no lead. First cycle is slew-limited, not -1.7
    out = c.apply_model(-1.7, v_ego=5.0, dt=0.05)
    assert out > -1.7 + 1e-9
    self.assertAlmostEqual(out, -0.4 * 0.05, places=6)  # expects negative ~ -0.02

  def test_apply_model_reaches_full_decel_over_time(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 20
    c._a_model_prev = 0.0
    v = 5.0
    out = 0.0
    for _ in range(200):  # 10s of 0.05s cycles at rate 0.4 -> reaches the -1.7 demand
      out = c.apply_model(-1.7, v_ego=v, dt=0.05)
    self.assertAlmostEqual(out, -1.7, places=6)

  def test_apply_model_safety_bubble_raw(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 20
    c._a_model_prev = 0.0
    v = 15.0
    for _ in range(3):
      out = c.apply_model(-1.7, v_ego=v, lead_drel=1.5 * v, dt=0.05)
    self.assertAlmostEqual(out, -1.7, places=6)

  def test_reset_state_clears_ramp_prev(self):
    c = E2EBiasController(params=MockParams())
    c._strength = 20
    c.apply_model(-1.7, v_ego=10.0, dt=0.05)
    assert c._a_model_prev < -0.001
    c.reset_state()
    self.assertEqual(c._a_model_prev, 0.0)
    self.assertEqual(c._a_mpc_prev, 0.0)

  def test_lead_gate_far_and_no_lead(self):
    assert lead_gate(None) == 1.0
    assert lead_gate(float("inf")) == 1.0
    self.assertAlmostEqual(lead_gate(3.0), 1.0, places=6)

  def test_lead_gate_close(self):
    self.assertAlmostEqual(lead_gate(0.5), 0.0, places=6)
    self.assertAlmostEqual(lead_gate(1.75), 0.5, places=6)

  def test_bias_stands_down_on_approach(self):
    c = make_controller(bias=0.15)
    # far lead (headway 3s) -> full bias
    self.assertAlmostEqual(c.apply(0.5, lead_drel=90.0, v_ego=30.0), 0.65, places=6)
    # close lead (headway 0.5s) -> no bias
    self.assertAlmostEqual(c.apply(0.5, lead_drel=15.0, v_ego=30.0), 0.5, places=6)
    # no lead -> full bias
    self.assertAlmostEqual(c.apply(0.5), 0.65, places=6)

  def test_bias_fades_with_lead_headway(self):
    c = make_controller(bias=0.15)
    # headway 1.75s: hold faded to half AND bleed active -> 0.5 + 0.15*(0.5 - 1.0)
    self.assertAlmostEqual(c.apply(0.5, lead_drel=52.5, v_ego=30.0), 0.5 + 0.15 * (0.5 - 1.0), places=6)

  def test_bleed_factor_follow_band(self):
    assert bleed_factor(None) == 0.0
    assert bleed_factor(float("inf")) == 0.0
    self.assertAlmostEqual(bleed_factor(1.5), 1.0, places=6)
    self.assertAlmostEqual(bleed_factor(2.0), 1.0, places=6)
    self.assertAlmostEqual(bleed_factor(0.5), 0.0, places=6)
    self.assertAlmostEqual(bleed_factor(3.5), 0.0, places=6)

  def test_bleed_eases_off_when_following(self):
    c = make_controller(bias=0.1)
    # following a lead at 2s headway, model cruising (~0) -> output pushed below model
    out = c.apply(0.0, lead_drel=60.0, v_ego=30.0)
    assert out < 0.0, f"expected bleed to push below 0, got {out}"
    # far / no lead -> no bleed (hold applies)
    self.assertAlmostEqual(c.apply(0.0), 0.0 + 0.1, places=6)
    self.assertAlmostEqual(c.apply(0.0, lead_drel=120.0, v_ego=30.0), 0.0 + 0.1, places=6)

  def test_bleed_steps_out_of_braking(self):
    c = make_controller(bias=0.1)
    # model braking hard -> bleed fades, never stacks decel
    out = c.apply(-0.3, lead_drel=60.0, v_ego=30.0)
    # at -0.3 the model-braking fade kills the bleed; bias fade also at zero -> -0.3
    self.assertAlmostEqual(out, -0.3, places=6)

  def test_negative_bias_no_bleed(self):
    c = make_controller(bias=-0.1)
    # no lead -> full negative bias, no bleed component
    self.assertAlmostEqual(c.apply(0.0), -0.1, places=6)
    # the lead-gate stands down negative bias too on approach (2s headway)
    self.assertAlmostEqual(c.apply(0.0, lead_drel=60.0, v_ego=30.0), -0.1 * 2.0 / 3.0, places=6)

  def test_model_change_resets_bias_to_default(self):
    bundle = json.dumps({"internalName": "modelA", "generation": 1})
    c = make_controller(bias=0.2, params_values={"ModelManager_ActiveBundle": bundle}, strength=20)
    c._tick = c.REFRESH_PERIOD - 1
    c.apply(0.5)
    assert c._params.values["LongitudinalE2EBias"] == 0
    assert c._params.values["LongitudinalE2EBiasTunedFor"] == "modelA:1"
    assert c._e2e_bias == DEFAULT_BIAS

  def test_model_change_resets_bias_with_parsed_dict(self):
    # real Params.get() returns JSON-type keys already-parsed (dict), not a string
    bundle = {"internalName": "modelA", "generation": 1}
    c = make_controller(bias=0.2, params_values={"ModelManager_ActiveBundle": bundle}, strength=20)
    c._tick = c.REFRESH_PERIOD - 1
    c.apply(0.5)
    assert c._params.values["LongitudinalE2EBias"] == 0
    assert c._params.values["LongitudinalE2EBiasTunedFor"] == "modelA:1"
    assert c._e2e_bias == DEFAULT_BIAS

  def test_same_model_keeps_bias(self):
    bundle = json.dumps({"internalName": "modelA", "generation": 1})
    c = make_controller(bias=0.2, params_values={
      "ModelManager_ActiveBundle": bundle,
      "LongitudinalE2EBiasTunedFor": "modelA:1",
    }, strength=20)
    c._tick = c.REFRESH_PERIOD - 1
    c.apply(0.5)
    assert c._params.values["LongitudinalE2EBias"] == "20"

  def test_missing_bundle_no_reset(self):
    c = make_controller(bias=0.2, strength=20)
    c._tick = c.REFRESH_PERIOD - 1
    c.apply(0.5)
    assert c._params.values["LongitudinalE2EBias"] == "20"

  def test_invalid_bundle_no_reset(self):
    c = make_controller(bias=0.2, params_values={"ModelManager_ActiveBundle": "{not json"}, strength=20)
    c._tick = c.REFRESH_PERIOD - 1
    c.apply(0.5)
    assert c._params.values["LongitudinalE2EBias"] == "20"


if __name__ == "__main__":
  unittest.main()
