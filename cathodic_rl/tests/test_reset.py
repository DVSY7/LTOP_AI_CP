# cathodic_rl/test_reset.py - 파일경로

"""CathodicProtectionEnv의 reset() 동작을 확인하는 테스트."""

import _path_setup

import numpy as np

from config.settings import (
    INITIAL_OUTPUT_CURRENT,
    INITIAL_OUTPUT_VOLTAGE,
    INITIAL_PIPE_POTENTIAL,
)
from env.cathodic_env import CathodicProtectionEnv


def test_reset() -> None:
    """환경 초기화 결과를 확인한다."""

    # 환경 생성
    env = CathodicProtectionEnv(render_mode="human")

    # 환경 초기화
    observation, info = env.reset(seed=42)

    print("=== reset() 테스트 결과 ===")

    print("\n1. Observation")
    print(observation)

    print("\n2. Observation 자료형")
    print(observation.dtype)

    print("\n3. Observation 형태")
    print(observation.shape)

    print("\n4. Observation 공간 포함 여부")
    print(env.observation_space.contains(observation))

    print("\n5. Info")
    print(info)

    # --------------------------------------------------------
    # 예상값 검사
    # --------------------------------------------------------

    expected_observation = np.array(
        [
            INITIAL_OUTPUT_VOLTAGE,
            INITIAL_OUTPUT_CURRENT,
            INITIAL_PIPE_POTENTIAL,
        ],
        dtype=np.float32,
    )

    assert isinstance(
        observation,
        np.ndarray,
    ), "Observation은 numpy.ndarray여야 합니다."

    assert (
        observation.dtype == np.float32
    ), "Observation의 자료형은 float32여야 합니다."

    assert observation.shape == (
        3,
    ), "Observation은 3개의 상태값을 가져야 합니다."

    assert np.array_equal(
        observation,
        expected_observation,
    ), (
        "초기 Observation 값이 settings.py의 초기 설정값과 "
        "일치하지 않습니다."
    )

    assert env.observation_space.contains(
        observation
    ), "Observation이 observation_space 범위를 벗어났습니다."

    assert (
        info["current_step"] == 0
    ), "reset 후 current_step은 0이어야 합니다."

    assert (
        info["output_voltage"] == INITIAL_OUTPUT_VOLTAGE
    ), "초기 출력전압이 설정값과 다릅니다."

    assert (
        info["output_current"] == INITIAL_OUTPUT_CURRENT
    ), "초기 출력전류가 설정값과 다릅니다."

    assert (
        info["pipe_potential"] == INITIAL_PIPE_POTENTIAL
    ), "초기 방식전위가 설정값과 다릅니다."

    print("\n모든 reset() 테스트를 통과했습니다.")

    env.render()
    env.close()


if __name__ == "__main__":
    test_reset()