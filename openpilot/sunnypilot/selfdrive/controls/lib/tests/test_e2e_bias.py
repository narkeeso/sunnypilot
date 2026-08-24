import json
import unittest

from openpilot.sunnypilot.selfdrive.controls.lib.e2e_bias import DEFAULT_BIAS, E2EBiasController, strength_to_mpc_ramp


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

  def test_no_bias_at_full_braking(self):
    c = make_controller(bias=0.15)
    self.assertAlmostEqual(c.apply(-0.3), -0.3, places=6)

  def test_linear_fade_midpoint(self):
    c = make_controller(bias=0.15)
    # at -0.5 * bias the blend is halfway through, so half of the bias is added
    self.assertAlmostEqual(c.apply(-0.075), -0.075 + 0.15 * 0.5, places=6)

  def test_full_bias_at_zero_braking(self):
    c = make_controller(bias=0.15)
    self.assertAlmostEqual(c.apply(0.0), 0.15, places=6)

  def test_zero_bias_at_minus_bias(self):
    c = make_controller(bias=0.13)
    self.assertAlmostEqual(c.apply(-0.13), -0.13, places=6)

  def test_fade_width_scales_with_bias(self):
    # stronger bias keeps more hold deeper into the braking region
    strong = make_controller(bias=0.2)
    weak = make_controller(bias=0.05)
    # at -0.1 braking: strong is halfway faded, weak is fully faded
    self.assertAlmostEqual(strong.apply(-0.1), -0.1 + 0.1, places=6)
    self.assertAlmostEqual(weak.apply(-0.1), -0.1, places=6)
    # at the same half-fade point relative to each: -0.5*bias
    self.assertAlmostEqual(strong.apply(-0.1), -0.1 + 0.1, places=6)
    self.assertAlmostEqual(weak.apply(-0.025), -0.025 + 0.025, places=6)

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
    # mpc wants -0.3, previous output was -0.05: ramped to -0.05 - 0.4*dt
    self.assertAlmostEqual(c.apply_mpc(-0.3, -0.05, dt=0.05), -0.05 - 0.4 * 0.05, places=6)

  def test_apply_mpc_emergency_bypasses(self):
    c = E2EBiasController(params=MockParams())
    c._mpc_ramp = 0.8
    self.assertAlmostEqual(c.apply_mpc(-1.0, -0.05, dt=0.05, bypass=True), -1.0, places=6)

  def test_apply_mpc_no_ramp_is_stock(self):
    c = E2EBiasController(params=MockParams())
    c._mpc_ramp = None
    self.assertAlmostEqual(c.apply_mpc(-0.3, -0.05, dt=0.05), -0.3, places=6)

  def test_apply_mpc_release_not_limited(self):
    c = E2EBiasController(params=MockParams())
    c._mpc_ramp = 0.8
    self.assertAlmostEqual(c.apply_mpc(-0.1, -0.3, dt=0.05), -0.1, places=6)

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
