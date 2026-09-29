"""Verify that SAC training uses the same 0.1 V absolute-target resolution as Modbus."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from cathodic_rl.config.settings import SAC_MAX_DELTA_VOLTAGE
from cathodic_rl.env.cathodic_env import CathodicProtectionEnv
from cathodic_rl.evaluate import evaluation_model_path
from cathodic_rl.model_metadata import write_sac_metadata
from cathodic_rl.voltage_quantization import quantized_target_delta, quantize_voltage
from shared.control_core import load_model_metadata


class VoltageResolutionTests(unittest.TestCase):
    def test_absolute_target_uses_half_up_quantization(self):
        self.assertEqual(quantized_target_delta(42.3, -0.05, 0.1), (42.3, 0.0))
        self.assertEqual(quantized_target_delta(42.3, -0.06, 0.1), (42.2, -0.1))
        self.assertEqual(quantized_target_delta(42.3, 0.05, 0.1), (42.4, 0.1))
        self.assertEqual(quantize_voltage(44.25, 0.1), 44.3)

    def test_sac_action_extremes_request_half_volt(self):
        env = CathodicProtectionEnv(action_mode="continuous")
        try:
            env.reset(seed=42)
            self.assertEqual(quantize_voltage(env.output_voltage, 0.1), env.output_voltage)
            self.assertEqual(env._convert_action(np.asarray([-1.0], dtype=np.float32)), -0.5)
            self.assertEqual(env._convert_action(np.asarray([1.0], dtype=np.float32)), 0.5)
            self.assertEqual(SAC_MAX_DELTA_VOLTAGE, 0.5)
        finally:
            env.close()

    def test_model_validity_guard_limits_post_quantization_delta(self):
        env = CathodicProtectionEnv(action_mode="continuous")
        try:
            env.output_voltage, env.output_current, env.pipe_potential = 45.0, 5.0, -1600.0
            env.environment_model.reset_history([-1600.0] * 7)
            _, _, _, _, info = env.step(np.asarray([1.0], dtype=np.float32))
            self.assertEqual(info["requested_delta_voltage"], 0.5)
            self.assertEqual(info["quantized_delta_voltage"], 0.5)
            self.assertEqual(info["safety_delta_voltage"], 0.5)
            self.assertTrue(info["model_limit_hit"])
            self.assertAlmostEqual(info["effective_delta_voltage"], 0.1)
        finally:
            env.close()

    def test_candidate_metadata_contains_half_volt_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "candidate.zip"
            artifact.write_bytes(b"candidate")
            metadata = load_model_metadata(write_sac_metadata(artifact))
        self.assertEqual(metadata.max_delta_voltage, 0.5)
        self.assertEqual(metadata.model_path.name, "candidate.zip")

    def test_evaluation_defaults_to_new_candidate_path(self):
        self.assertIn("dv050_step010_candidate", evaluation_model_path())


if __name__ == "__main__":
    unittest.main()
