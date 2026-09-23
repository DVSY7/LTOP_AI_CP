"""Gymnasium 환경의 현재 DQN·SAC 계약을 검증한다."""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from cathodic_rl.config.settings import MAX_EPISODE_STEPS
from cathodic_rl.env.cathodic_env import CathodicProtectionEnv


class EnvironmentContractTests(unittest.TestCase):
    def test_discrete_reset_and_step_follow_gymnasium_contract(self):
        env = CathodicProtectionEnv(action_mode="discrete")
        try:
            observation, info = env.reset(seed=42)
            self.assertEqual(observation.shape, (3,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(info["current_step"], 0)

            next_observation, _, terminated, truncated, info = env.step(1)
            self.assertTrue(env.observation_space.contains(next_observation))
            self.assertEqual(info["current_step"], 1)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        finally:
            env.close()

    def test_continuous_reset_and_step_follow_policy_state_contract(self):
        env = CathodicProtectionEnv(action_mode="continuous")
        try:
            observation, _ = env.reset(seed=42)
            self.assertEqual(observation.shape, (4,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(env.observation_space.contains(observation))

            next_observation, _, _, _, info = env.step(
                np.asarray([0.0], dtype=np.float32))
            self.assertEqual(next_observation.shape, (4,))
            self.assertTrue(env.observation_space.contains(next_observation))
            self.assertIn("safety_delta_voltage", info)
        finally:
            env.close()

    def test_episode_is_truncated_at_configured_limit(self):
        env = CathodicProtectionEnv(action_mode="discrete")
        try:
            env.reset(seed=42)
            env.current_step = MAX_EPISODE_STEPS - 1
            _, _, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["current_step"], MAX_EPISODE_STEPS)
        finally:
            env.close()

    def test_gymnasium_environment_checker_passes(self):
        env = CathodicProtectionEnv(action_mode="continuous")
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
