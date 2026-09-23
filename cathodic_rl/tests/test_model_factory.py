"""DQN·SAC 모델 생성과 환경 연결을 검증한다."""

import unittest

from stable_baselines3 import DQN, SAC

from cathodic_rl.env.cathodic_env import CathodicProtectionEnv
from cathodic_rl.models.dqn_model import create_dqn_model
from cathodic_rl.models.sac_model import create_sac_model


class ModelFactoryTests(unittest.TestCase):
    def test_dqn_model_accepts_discrete_environment_observation(self):
        env = CathodicProtectionEnv(action_mode="discrete")
        try:
            model = create_dqn_model(env)
            self.assertIsInstance(model, DQN)
            observation, _ = env.reset(seed=42)
            action, _ = model.predict(observation, deterministic=True)
            self.assertTrue(env.action_space.contains(action))
        finally:
            env.close()

    def test_sac_model_accepts_continuous_environment_observation(self):
        env = CathodicProtectionEnv(action_mode="continuous")
        try:
            model = create_sac_model(env)
            self.assertIsInstance(model, SAC)
            observation, _ = env.reset(seed=42)
            action, _ = model.predict(observation, deterministic=True)
            self.assertTrue(env.action_space.contains(action))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
