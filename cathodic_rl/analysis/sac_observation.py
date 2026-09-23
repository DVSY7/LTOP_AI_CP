# tests/test_sac_observation.py

import numpy as np

from cathodic_rl.env.cathodic_env import CathodicProtectionEnv


def print_observation(
    title,
    observation,
    env,
):
    """
    SAC Observation 4개를 보기 쉽게 출력한다.
    """

    # 정규화된 Trend를 다시 실제 mV 변화량으로 복원
    normalized_trend = float(observation[3])

    tb_trend = (
        normalized_trend * 20.0
    ) - 10.0

    print()
    print(title)

    print(
        f"Observation = {observation}"
    )

    print(
        f"실제 상태 | "
        f"V={env.output_voltage:.3f} V | "
        f"I={env.output_current:.3f} A | "
        f"TB={env.pipe_potential:.3f} mV"
    )

    print(
        f"TB Trend = "
        f"{tb_trend:+.3f} mV"
    )


def main():

    print()
    print("=" * 80)
    print("4-State SAC Observation 확인")
    print("=" * 80)

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )


    # ========================================================
    # RESET
    # ========================================================

    observation, info = env.reset(
        seed=42
    )

    print_observation(
        title="[RESET]",
        observation=observation,
        env=env,
    )


    # ========================================================
    # STEP
    #
    # 진단을 위해 같은 +0.5 Action을 반복한다.
    #
    # normalized Action +0.5
    # → 현재 ±0.05V 설정에서는
    # → ΔV 약 +0.025V
    # ========================================================

    action = np.array(
        [0.5],
        dtype=np.float32,
    )

    for step in range(1, 6):

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print_observation(
            title=f"[STEP {step}]",
            observation=observation,
            env=env,
        )

        print(
            f"Reward = {reward:+.5f}"
        )

        if terminated or truncated:
            break


    env.close()


if __name__ == "__main__":
    main()
