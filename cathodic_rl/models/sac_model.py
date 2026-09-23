"""전기방식 제어용 SAC 모델 생성."""

import gymnasium as gym
from stable_baselines3 import SAC

from cathodic_rl.config.settings import (
    DEFAULT_RANDOM_SEED,
    SAC_BUFFER_SIZE,
    SAC_ENT_COEF,
    SAC_LEARNING_STARTS,
)


def create_sac_model(
    env: gym.Env,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
) -> SAC:
    """CathodicProtectionEnv를 학습할 SAC 모델을 생성한다."""

    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        ent_coef=SAC_ENT_COEF,
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=SAC_LEARNING_STARTS,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )
