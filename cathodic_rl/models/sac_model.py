"""전기방식 제어용 SAC 모델 생성."""

import gymnasium as gym
from stable_baselines3 import SAC

from config.settings import DEFAULT_RANDOM_SEED


def create_sac_model(
    env: gym.Env,
) -> SAC:
    """CathodicProtectionEnv를 학습할 SAC 모델을 생성한다."""

    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        ent_coef=0.1,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        seed=DEFAULT_RANDOM_SEED,
    )