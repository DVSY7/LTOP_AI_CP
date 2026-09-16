# """전기방식 강화학습 환경의 보상 계산 함수."""

# from config.settings import (
#     TARGET_POTENTIAL_MAX,
#     TARGET_POTENTIAL_MIN,
# )

# def calculate_reward(
#     pipe_potential: float,
#     output_voltage: float,
#     output_current: float,
#     applied_delta_voltage: float,
# ) -> float:
#     """방식전위, 소비전력 및 출력변화를 이용해 보상을 계산한다.

#     현재 프로토타입용 임시 보상함수이며,
#     실제 적용 전 현장 기준과 학습 결과에 따라 조정해야 한다.
#     """

#     # 1. 목표 방식전위 도달 여부 또는 목표 범위까지의 거리
#     if TARGET_POTENTIAL_MIN <= pipe_potential <= TARGET_POTENTIAL_MAX:
#         potential_reward = 10.0

#     elif pipe_potential > TARGET_POTENTIAL_MAX:
#         # 미방식 방향: 예) -700mV
#         distance = pipe_potential - TARGET_POTENTIAL_MAX
#         potential_reward = -distance / 100.0

#     else:
#         # 과방식 방향: 예) -1000mV
#         distance = TARGET_POTENTIAL_MIN - pipe_potential
#         potential_reward = -distance / 100.0

#     # 2. 소비전력 패널티
#     power_penalty = (
#         0.01 * output_voltage * output_current
#     )

#     # 3. 급격하거나 불필요한 출력변화 패널티
#     action_penalty = (
#         0.05 * abs(applied_delta_voltage)
#     )

#     reward = (
#         potential_reward
#         - power_penalty
#         - action_penalty
#     )

#     return float(reward)

"""전기방식 강화학습 환경의 보상 계산 함수."""

from config.settings import (
    TARGET_POTENTIAL_MAX,
    TARGET_POTENTIAL_MIN,
)


def calculate_dqn_reward(
    pipe_potential: float,
    output_voltage: float,
    output_current: float,
    applied_delta_voltage: float,
) -> float:
    """방식전위, 소비전력 및 출력변화를 이용해 보상을 계산한다.

    현재 프로토타입용 임시 보상함수이며,
    실제 적용 전 현장 기준과 학습 결과에 따라 조정해야 한다.
    """

    # 1. 방식전위 상태 평가
    if TARGET_POTENTIAL_MIN <= pipe_potential <= TARGET_POTENTIAL_MAX:
        # 목표 방식전위 범위
        potential_reward = 10.0

    elif pipe_potential > TARGET_POTENTIAL_MAX:
        # 미방식 상태
        # 미방식을 가장 위험한 상태로 판단하여 큰 패널티 적용
        distance = pipe_potential - TARGET_POTENTIAL_MAX

        potential_reward = (
            -100.0
            - distance / 100.0
        )

    else:
        # 과방식 상태
        # 목표 범위 이탈 패널티와 거리 패널티 적용
        distance = TARGET_POTENTIAL_MIN - pipe_potential

        potential_reward = (
            -50.0
            - distance / 100.0
        )

    # 2. 소비전력 패널티
    power_penalty = (
        0.01 * output_voltage * output_current
    )

    # 3. 출력변화 패널티
    action_penalty = (
        0.05 * abs(applied_delta_voltage)
    )

    reward = (
        potential_reward
        - power_penalty
        - action_penalty
    )

    return float(reward)

def calculate_sac_reward(
    previous_pipe_potential: float,
    pipe_potential: float,
    output_voltage: float,
    output_current: float,
    applied_delta_voltage: float,
) -> float:
    """SAC 연속제어용 보상함수."""

    target_potential = (
        TARGET_POTENTIAL_MIN
        +
        TARGET_POTENTIAL_MAX
    ) / 2.0

    previous_distance = abs(
        previous_pipe_potential - target_potential
    )

    current_distance = abs(
        pipe_potential - target_potential
    )

    # 현재 위치에 대한 보상
    if TARGET_POTENTIAL_MIN <= pipe_potential <= TARGET_POTENTIAL_MAX:
        potential_reward = 10.0

    elif pipe_potential > TARGET_POTENTIAL_MAX:
        distance = pipe_potential - TARGET_POTENTIAL_MAX
        potential_reward = (
            -100.0
            - distance / 100.0
        )

    else:
        distance = TARGET_POTENTIAL_MIN - pipe_potential
        potential_reward = (
            -50.0
            - distance / 100.0
        )

    # 이전 Step보다 목표 중심에 가까워졌는지 평가
    progress_reward = (
        previous_distance - current_distance
    )

    power_penalty = (
        0.01 * output_voltage * output_current
    )

    action_penalty = (
        0.05 * abs(applied_delta_voltage)
    )

    reward = (
        potential_reward
        + progress_reward
        - power_penalty
        - action_penalty
    )

    return float(reward)