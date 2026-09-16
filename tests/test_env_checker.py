"""Gymnasium 환경 전체 규격 테스트."""
from gymnasium.utils.env_checker import check_env

from env.cathodic_env import CathodicProtectionEnv


import numpy as np

from env.cathodic_env import CathodicProtectionEnv


def test_step_determinism_debug():

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )

    # 같은 Action 사용
    action = np.array(
        [0.25],
        dtype=np.float32,
    )


    # ========================================================
    # 첫 번째 실행
    # ========================================================

    env.reset(
        seed=42
    )

    _, reward_1, _, _, info_1 = (
        env.step(
            action
        )
    )


    # ========================================================
    # 같은 Seed + 같은 Action으로 다시 실행
    # ========================================================

    env.reset(
        seed=42
    )

    _, reward_2, _, _, info_2 = (
        env.step(
            action
        )
    )


    print(
        "\n===== Step Determinism 확인 ====="
    )

    print(
        f"Reward 1 : {reward_1}"
    )

    print(
        f"Reward 2 : {reward_2}"
    )


    print(
        "\n===== Info 비교 ====="
    )


    # --------------------------------------------------------
    # 각 Info 항목 비교
    # --------------------------------------------------------

    for key in info_1.keys():

        value_1 = info_1[key]
        value_2 = info_2[key]

        same = (
            value_1
            ==
            value_2
        )

        print(
            f"{key:25s} | "
            f"{value_1!s:25s} | "
            f"{value_2!s:25s} | "
            f"same={same}"
        )


    env.close()


if __name__ == "__main__":

    test_step_determinism_debug()


def test_environment_format() -> None:
    """환경이 Gymnasium 인터페이스 규칙을 따르는지 확인한다."""

    env = CathodicProtectionEnv()

    print("\n[Gymnasium 환경 규격 테스트]")

    check_env(
        env,
        skip_render_check=True,
    )

    env.close()

    print(
        "관측공간, 행동공간, reset(), step() 규격\n"
        "결과: 통과"
    )


if __name__ == "__main__":
    test_environment_format()

    print("\nGymnasium 환경 전체 검사 통과")