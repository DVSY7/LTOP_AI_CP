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

from cathodic_rl.config.settings import (
    SAC_MAX_DELTA_VOLTAGE,
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
    """
    SAC 연속제어용 Reward V2.

    핵심 목표
    ---------
    1. 목표 중심(-1600mV)에 가까워지는 행동을 명확하게 보상
    2. 목표에서 멀어지는 행동을 명확하게 패널티
    3. 목표 범위에서는 안정적인 유지를 보상
    4. 목표 범위에서 불필요한 제어는 억제
    """

    # --------------------------------------------------------
    # 1. 목표 중심 계산
    # --------------------------------------------------------

    target_potential = (
        TARGET_POTENTIAL_MIN
        + TARGET_POTENTIAL_MAX
    ) / 2.0


    # --------------------------------------------------------
    # 2. 이전 / 현재 목표 중심까지 거리
    # --------------------------------------------------------

    previous_distance = abs(
        previous_pipe_potential
        - target_potential
    )

    current_distance = abs(
        pipe_potential
        - target_potential
    )


    # --------------------------------------------------------
    # 3. 목표 접근 Progress Reward
    #
    # 예:
    #
    # 목표에 5mV 가까워짐
    # → +25
    #
    # 목표에서 5mV 멀어짐
    # → -25
    # --------------------------------------------------------

    progress_reward = (
        previous_distance
        - current_distance
    ) * 5.0


    # --------------------------------------------------------
    # 4. 목표 범위 도달 / 유지 보상
    # --------------------------------------------------------

    in_target = (
        TARGET_POTENTIAL_MIN
        <= pipe_potential
        <= TARGET_POTENTIAL_MAX
    )

    if in_target:
        target_reward = 20.0
    else:
        target_reward = 0.0


    # --------------------------------------------------------
    # 5. Action 방향 보상 / 목표 구간 Action 억제
    #
    # 목표 범위에서는 불필요한 출력변화를
    # 조금 더 강하게 억제한다.
    # --------------------------------------------------------

    normalized_action = max(
        -1.0,
        min(
            1.0,
            applied_delta_voltage / SAC_MAX_DELTA_VOLTAGE,
        ),
    )

    if previous_pipe_potential < TARGET_POTENTIAL_MIN:
        # 과방식: 전압 감소(-)가 올바른 방향이다.
        direction_reward = -5.0 * normalized_action
        action_penalty = 0.0
    elif previous_pipe_potential > TARGET_POTENTIAL_MAX:
        # 미방식: 전압 증가(+)가 올바른 방향이다.
        direction_reward = 5.0 * normalized_action
        action_penalty = 0.0
    else:
        # 목표 구간에서는 HOLD를 명확하게 학습시킨다.
        direction_reward = 0.0
        action_penalty = 5.0 * abs(normalized_action)


    # --------------------------------------------------------
    # 6. 최종 Reward
    # --------------------------------------------------------

    reward = (
        progress_reward
        + target_reward
        + direction_reward
        - action_penalty
    )

    return float(reward)
