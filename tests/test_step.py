"""CathodicProtectionEnv.step() 테스트."""
import _path_setup

import numpy as np

from config.settings import (
    ACTION_TO_DELTA_VOLTAGE,
    INITIAL_OUTPUT_VOLTAGE,
    INITIAL_PIPE_POTENTIAL,
    TEMP_CURRENT_PER_VOLT,
    TEMP_POTENTIAL_CHANGE_PER_VOLT,
    ACTION_INCREASE,
    MAX_EPISODE_STEPS,
    TARGET_POTENTIAL_MAX,
    TARGET_POTENTIAL_MIN,
)
from env.cathodic_env import CathodicProtectionEnv


ACTION_NAMES = {
    0: "출력전압 감소",
    1: "출력전압 유지",
    2: "출력전압 증가",
}


def test_state_change() -> None:
    """Action에 따른 전압·전류·방식전위 변화를 확인한다."""

    env = CathodicProtectionEnv()

    rewards: dict[int, float] = {}
    
    print("\n[Action별 상태 변화 테스트]")

    for action in (0, 1, 2):

        env.reset()

        observation, reward, terminated, truncated, info = env.step(action)

        # 현재 Action의 Reward 저장
        rewards[action] = reward

        delta_voltage = ACTION_TO_DELTA_VOLTAGE[action]

        expected_voltage = (
            INITIAL_OUTPUT_VOLTAGE + delta_voltage
        )
        expected_current = (
            expected_voltage * TEMP_CURRENT_PER_VOLT
        )
        expected_potential = (
            INITIAL_PIPE_POTENTIAL
            + delta_voltage
            * TEMP_POTENTIAL_CHANGE_PER_VOLT
        )

        actual_voltage = float(observation[0])
        actual_current = float(observation[1])
        actual_potential = float(observation[2])

        # float32 계산 오차를 고려하여 np.isclose() 사용
        assert np.isclose(actual_voltage, expected_voltage)
        assert np.isclose(actual_current, expected_current)
        assert np.isclose(actual_potential, expected_potential)

        assert np.isclose(
            info["output_voltage"],
            expected_voltage,
        )
        assert np.isclose(
            info["output_current"],
            expected_current,
        )
        assert np.isclose(
            info["pipe_potential"],
            expected_potential,
        )

        assert info["current_step"] == 1
        assert terminated is False
        assert truncated is False

        print(
            f"Action {action} ({ACTION_NAMES[action]})\n"
            f"  전압: {INITIAL_OUTPUT_VOLTAGE:.1f}"
            f" → {actual_voltage:.1f} V\n"
            f"  전류: {actual_current:.2f} A\n"
            f"  전위: {INITIAL_PIPE_POTENTIAL:.1f}"
            f" → {actual_potential:.1f} mV\n"
            f"  Reward: {reward:.5f}\n"
            f"  결과: 통과"
        )
        
    assert rewards[2] > rewards[1] > rewards[0]

    print(
        "\n[Reward 방향성 테스트]\n"
        "증가 Action > 유지 Action > 감소 Action\n"
        "결과: 통과"
    )
    env.close()

def test_reach_target_potential() -> None:
    """증가 Action 반복 시 목표 방식전위 구간에 도달하는지 확인한다."""

    env = CathodicProtectionEnv()
    env.reset()

    previous_potential = INITIAL_PIPE_POTENTIAL
    previous_reward: float | None = None

    print("\n[목표 방식전위 도달 테스트]")

    for step_number in range(1, MAX_EPISODE_STEPS + 1):
        observation, reward, terminated, truncated, info = env.step(
            ACTION_INCREASE
        )

        current_potential = float(observation[2])

        # 증가 Action 적용 시 전위가 음의 방향으로 이동해야 함
        assert current_potential < previous_potential

        # 목표에 가까워지는 동안 Reward가 개선되어야 함
        if previous_reward is not None:
            assert reward > previous_reward

        if (
            TARGET_POTENTIAL_MIN
            <= current_potential
            <= TARGET_POTENTIAL_MAX
        ):
            assert reward > 0.0
            assert terminated is False
            assert truncated is False

            print(
                f"목표 구간 도달 Step: {step_number}\n"
                f"  출력전압: {float(observation[0]):.1f} V\n"
                f"  출력전류: {float(observation[1]):.2f} A\n"
                f"  방식전위: {current_potential:.1f} mV\n"
                f"  Reward: {reward:.5f}\n"
                f"  결과: 통과"
            )

            env.close()
            return

        assert terminated is False
        assert truncated is False

        previous_potential = current_potential
        previous_reward = reward

    env.close()

    raise AssertionError(
        "최대 Step 안에 목표 방식전위 구간에 도달하지 못했습니다."
    )

def test_invalid_action() -> None:
    """정의되지 않은 Action이 차단되는지 확인한다."""

    env = CathodicProtectionEnv()
    env.reset()

    print("\n[잘못된 Action 테스트]")

    invalid_action = 3

    try:
        env.step(invalid_action)
    except ValueError as error:
        print(
            f"Action {invalid_action} 정상 차단\n"
            f"  오류 내용: {error}\n"
            f"  결과: 통과"
        )
    else:
        raise AssertionError(
            f"유효하지 않은 Action {invalid_action}이 "
            "차단되지 않았습니다."
        )
    finally:
        print("\n모든 step() 상태 변화 테스트 통과")
        env.close()


def test_step() -> None:
    test_state_change()
    test_reach_target_potential()
    test_invalid_action()
    

if __name__ == "__main__":
    test_step()
    