"""CathodicProtectionEnv 에피소드 종료 테스트."""
import _path_setup

from config.settings import (
    ACTION_HOLD,
    MAX_EPISODE_STEPS,
)
from env.cathodic_env import CathodicProtectionEnv


def test_episode_truncation() -> None:
    """최대 Step 도달 시 에피소드가 종료되는지 확인한다."""

    env = CathodicProtectionEnv()
    env.reset()

    print("\n[에피소드 최대 Step 테스트]")

    for step_number in range(1, MAX_EPISODE_STEPS + 1):
        _, _, terminated, truncated, info = env.step(
            ACTION_HOLD
        )

        assert terminated is False
        assert info["current_step"] == step_number

        if step_number < MAX_EPISODE_STEPS:
            assert truncated is False
        else:
            assert truncated is True

    print(
        f"최대 Step: {MAX_EPISODE_STEPS}\n"
        f"마지막 current_step: {info['current_step']}\n"
        f"terminated: {terminated}\n"
        f"truncated: {truncated}\n"
        f"결과: 통과"
    )

    env.close()


if __name__ == "__main__":
    test_episode_truncation()

    print("\n에피소드 종료 테스트 통과")