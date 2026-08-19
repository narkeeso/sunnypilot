import json
import unittest

from openpilot.sunnypilot.selfdrive.controls.lib.tuning.e2e_bias import DEFAULT_BIAS, E2EBiasController, bleed_factor, lead_gate, strength_to_mpc_ramp


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
    c._mpc_ramp = 0.4
    c._a_mpc_prev = -0.05
    # mpc wants -0.3, previous output was -0.05: ramped to -0.05 - 0.4*dt
    self.assertAlmostEqual(c.apply_mpc(-0.3, dt=0.05), -0.05 - 0.4 * 0.05, places=6)

  def test_apply_mpc_emergency_bypasses(self):
    c = E2EBiasController(params=MockParams())
    c._mpc_ramp = 0.8
    c._a_mpc_prev = -0.05
    self.assertAlmostEqual(c.apply_mpc(-1.0, dt=0.05, bypass=True), -1.0, places=6)

  def test_apply_mpc_no_ramp_is_stock(self):
    c = E2EBiasController(params=MockParams())
    c._mpc_ramp = None
    c._a_mpc_prev = -0.05
    self.assertAlmostEqual(c.apply_mpc(-0.3, dt=0.05), -0.3, places=6)

  def test_apply_mpc_release_not_limited(self):
    c = E2EBiasController(params=MockParams())
    c._mpc_ramp = 0.8
    c._a_mpc_prev = -0.3
    self.assertAlmostEqual(c.apply_mpc(-0.1, dt=0.05), -0.1, places=6)

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
