"""전기방식 제어용 DQN 모델 생성."""

import gymnasium as gym
from stable_baselines3 import DQN

from cathodic_rl.config.settings import DEFAULT_RANDOM_SEED


def create_dqn_model(
    env: gym.Env,
) -> DQN:
    """CathodicProtectionEnv를 학습할 DQN 모델을 생성한다."""

    return DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        seed=DEFAULT_RANDOM_SEED,
    )
