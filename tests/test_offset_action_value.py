# tests/test_offset_action_value.py

import numpy as np

from env.cathodic_env import CathodicProtectionEnv


# ============================================================
# 테스트 설정
# ============================================================

TEST_STEPS = 20

# SAC 최대 Action ±0.05V 기준
TEST_ACTIONS = {
    "DECREASE (-0.05V)": -1.0,
    "HOLD     ( 0.00V)":  0.0,
    "INCREASE (+0.05V)": +1.0,
}

# 실제 Episode에서도 도달 가능한 수준을 기준으로 비교
TEST_OFFSETS = [
    -0.10,
     0.00,
    +0.10,
]


def run_case(
    env,
    initial_state,
    initial_offset,
    action_value,
):
    """
    동일한 V / I / TB / History에서
    control_current_offset과 Action만 변경하여
    20-step 누적 Reward를 계산한다.
    """

    # 환경 기본 Reset
    env.reset()

    # --------------------------------------------------------
    # 동일한 외부 상태 강제 설정
    # --------------------------------------------------------
    env.output_voltage = initial_state["voltage"]
    env.output_current = initial_state["current"]
    env.pipe_potential = initial_state["tb"]

    # 모든 CASE에서 동일한 TB History 사용
    initial_tb_history = [
        initial_state["tb"]
    ] * 7

    env.environment_model.reset_history(
        initial_tb_history
    )

    # --------------------------------------------------------
    # 이번 테스트의 핵심
    # offset만 강제로 다르게 설정
    # --------------------------------------------------------
    env.environment_model.control_current_offset = (
        initial_offset
    )

    total_reward = 0.0

    for step in range(TEST_STEPS):

        action = np.array(
            [action_value],
            dtype=np.float32,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "final_tb": env.pipe_potential,
        "final_voltage": env.output_voltage,
        "final_offset": (
            env.environment_model.control_current_offset
        ),
    }


def test_offset_action_value():

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )

    # --------------------------------------------------------
    # 두 방향 모두 검사
    #
    # BELOW:
    #   너무 음수 → 원래는 -ΔV가 올바른 방향
    #
    # ABOVE:
    #   덜 음수 → 원래는 +ΔV가 올바른 방향
    # --------------------------------------------------------

    test_states = {
        "BELOW (-1618 mV)": {
            "voltage": 43.50,
            "current": 5.94,
            "tb": -1618.0,
        },

        "ABOVE (-1568 mV)": {
            "voltage": 43.50,
            "current": 5.94,
            "tb": -1568.0,
        },
    }

    for state_name, initial_state in test_states.items():

        print()
        print("=" * 90)
        print(f"초기 상태 : {state_name}")
        print(
            f"V={initial_state['voltage']:.2f} V | "
            f"I={initial_state['current']:.2f} A | "
            f"TB={initial_state['tb']:.1f} mV"
        )
        print("=" * 90)

        for offset in TEST_OFFSETS:

            print()
            print(
                f"===== Initial Offset = "
                f"{offset:+.3f} A ====="
            )

            results = []

            for label, action_value in TEST_ACTIONS.items():

                result = run_case(
                    env=env,
                    initial_state=initial_state,
                    initial_offset=offset,
                    action_value=action_value,
                )

                result["label"] = label

                results.append(result)

            # Reward 높은 순서
            results.sort(
                key=lambda x: x["total_reward"],
                reverse=True,
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):

                print(
                    f"{rank}위 | "
                    f"{result['label']:<21} | "
                    f"Reward={result['total_reward']:+9.3f} | "
                    f"Final TB={result['final_tb']:+8.1f} | "
                    f"Final V={result['final_voltage']:.3f} | "
                    f"Final Offset="
                    f"{result['final_offset']:+.3f}"
                )

    env.close()


if __name__ == "__main__":
    test_offset_action_value()