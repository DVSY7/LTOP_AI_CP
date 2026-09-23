"""정책 방향·목표 유지·포화율 Gate 검증."""

import numpy as np
import unittest

from cathodic_rl.evaluation.policy_gate import evaluate_policy


class DirectionalPolicy:
    def predict(self, observation, deterministic=True):
        potential = float(observation[2]) * 3000.0 - 3000.0
        if potential < -1610.0:
            action = -0.5
        elif potential > -1590.0:
            action = 0.5
        else:
            action = 0.0
        return np.asarray([action], dtype=np.float32), None


class CollapsedPolicy:
    def predict(self, observation, deterministic=True):
        return np.asarray([-1.0], dtype=np.float32), None


class PolicyGateTests(unittest.TestCase):
    def test_directional_policy_passes(self):
        result = evaluate_policy(DirectionalPolicy())
        self.assertTrue(result.passed)
        self.assertEqual(result.saturation_rate, 0.0)

    def test_collapsed_policy_fails(self):
        result = evaluate_policy(CollapsedPolicy())
        self.assertFalse(result.passed)
        self.assertFalse(result.direction_passed)
        self.assertFalse(result.target_hold_passed)
        self.assertEqual(result.saturation_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
