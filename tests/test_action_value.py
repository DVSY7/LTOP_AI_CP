# tests/test_action_value.py

import numpy as np

from env.cathodic_env import CathodicProtectionEnv


# ============================================================
# 테스트 설정
# ============================================================

TEST_STEPS = 20

# 현재 SAC 최대 제어량을 ±0.05V로 설정했다고 했으므로
# 세 가지 고정 정책을 비교
TEST_ACTIONS = {
    "DECREASE (-0.05V)": -1.0,
    "HOLD     ( 0.00V)":  0.0,
    "INCREASE (+0.05V)": +1.0,
}


def run_fixed_action_test(env, initial_state, action_value, label):
    """
    동일한 초기 상태에서 하나의 Action을 계속 반복하면서
    최종 TB와 누적 Reward를 확인한다.
    """

    # 환경 Reset
    obs, info = env.reset()

    # --------------------------------------------------------
    # 테스트용 초기 상태 강제 지정
    # --------------------------------------------------------
    env.output_voltage = initial_state["voltage"]
    env.output_current = initial_state["current"]
    env.pipe_potential = initial_state["tb"]

    # TB History도 현재 TB를 기준으로 동일하게 초기화
    #
    # 이번 테스트 목적은 서로 다른 Action의 효과 비교이므로
    # 세 실험의 History 조건을 동일하게 맞춘다.
    initial_tb_history = [initial_state["tb"]] * 7
    env.environment_model.reset_history(initial_tb_history)

    total_reward = 0.0

    print()
    print(f"--- {label} ---")

    for step in range(1, TEST_STEPS + 1):

        # continuous action은 [-1, +1]
        action = np.array([action_value], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"Step {step:02d} | "
            f"V={env.output_voltage:.3f} | "
            f"TB={env.pipe_potential:.1f} | "
            f"Reward={reward:+8.3f}"
        )

        if terminated or truncated:
            break

    print(
        f"결과 | "
        f"최종 V={env.output_voltage:.3f} V | "
        f"최종 TB={env.pipe_potential:.1f} mV | "
        f"누적 Reward={total_reward:+.3f}"
    )

    return {
        "label": label,
        "final_tb": env.pipe_potential,
        "total_reward": total_reward,
    }


def test_action_value():

    # SAC용 continuous 환경
    env = CathodicProtectionEnv(action_mode="continuous")

    # ========================================================
    # 테스트할 두 가지 상태
    #
    # BELOW : 너무 음수 → 전압 감소가 올바른 방향
    # ABOVE : 덜 음수   → 전압 증가가 올바른 방향
    # ========================================================

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
        print("=" * 70)
        print(f"초기 상태 : {state_name}")
        print("=" * 70)

        results = []

        for label, action_value in TEST_ACTIONS.items():

            result = run_fixed_action_test(
                env=env,
                initial_state=initial_state,
                action_value=action_value,
                label=label,
            )

            results.append(result)

        # ----------------------------------------------------
        # 누적 Reward가 높은 순서 출력
        # ----------------------------------------------------
        results.sort(
            key=lambda x: x["total_reward"],
            reverse=True
        )

        print()
        print("===== 누적 Reward 순위 =====")

        for rank, result in enumerate(results, start=1):

            print(
                f"{rank}위 | "
                f"{result['label']:<20} | "
                f"최종 TB={result['final_tb']:.1f} | "
                f"누적 Reward={result['total_reward']:+.3f}"
            )

    env.close()


if __name__ == "__main__":
    test_action_value()