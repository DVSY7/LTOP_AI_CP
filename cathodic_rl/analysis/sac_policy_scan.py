# tests/test_sac_policy_scan.py

import numpy as np
import torch

from stable_baselines3 import SAC

from cathodic_rl.config.settings import (
    SAC_MODEL_PATH,
    MIN_OUTPUT_VOLTAGE,
    MAX_OUTPUT_VOLTAGE,
    MIN_OUTPUT_CURRENT,
    MAX_OUTPUT_CURRENT,
    MIN_PIPE_POTENTIAL,
    MAX_PIPE_POTENTIAL,
)


def make_observation(
    voltage: float,
    current: float,
    tb: float,
    offset: float,
) -> np.ndarray:
    """
    실제 상태값
    [V, I, TB, TB Trend]

    ↓

    SAC 입력용 정규화 Observation
    """

    normalized_voltage = (
        voltage - MIN_OUTPUT_VOLTAGE
    ) / (
        MAX_OUTPUT_VOLTAGE - MIN_OUTPUT_VOLTAGE
    )

    normalized_current = (
        current - MIN_OUTPUT_CURRENT
    ) / (
        MAX_OUTPUT_CURRENT - MIN_OUTPUT_CURRENT
    )

    normalized_tb = (
        tb - MIN_PIPE_POTENTIAL
    ) / (
        MAX_PIPE_POTENTIAL - MIN_PIPE_POTENTIAL
    )

    # offset 인자는 기존 분석 호출부 호환용이며 정책 입력에는 사용하지 않는다.

    # ------------------------------------------------------------
    # TB Trend 정규화
    #
    # 이번 Policy Scan에서는 모든 TB에 대해
    # "현재 TB 변화가 없는 상태"를 동일하게 적용한다.
    #
    # 실제값 0 mV
    # -10 ~ +10 mV 정규화 → 0.5
    # ------------------------------------------------------------

    tb_trend = 0.0

    TB_TREND_MIN = -10.0
    TB_TREND_MAX = +10.0

    normalized_tb_trend = (
        tb_trend - TB_TREND_MIN
    ) / (
        TB_TREND_MAX - TB_TREND_MIN
    )


    return np.array(
        [
            normalized_voltage,
            normalized_current,
            normalized_tb,
            normalized_tb_trend,
        ],
        dtype=np.float32,
    )


def get_min_q(
    model,
    observation,
    action,
):
    """
    SAC Twin Critic의 Q1, Q2 중
    작은 값을 반환한다.
    """

    obs_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    action_tensor = torch.tensor(
        [[action]],
        dtype=torch.float32,
    )

    with torch.no_grad():

        q_values = model.critic(
            obs_tensor,
            action_tensor,
        )

        q1 = float(
            q_values[0]
            .cpu()
            .numpy()
            .flatten()[0]
        )

        q2 = float(
            q_values[1]
            .cpu()
            .numpy()
            .flatten()[0]
        )

    return min(q1, q2)


def find_best_critic_action(
    model,
    observation,
):
    """
    -1 ~ +1 Action 범위를 촘촘하게 검사해서
    Critic이 가장 높게 평가하는 Action을 찾는다.

    이번에는 0.05 간격으로 검사한다.
    """

    actions = np.linspace(
        -1.0,
        1.0,
        41,
    )

    best_action = None
    best_q = -np.inf

    for action in actions:

        min_q = get_min_q(
            model=model,
            observation=observation,
            action=float(action),
        )

        if min_q > best_q:

            best_q = min_q
            best_action = float(action)

    return best_action, best_q


def classify_expected_direction(
    tb: float,
) -> str:
    """
    현재 V1 목표범위 기준으로
    기대되는 제어 방향을 표시한다.

    TB < -1610  → 전압 감소
    목표범위    → 유지
    TB > -1590  → 전압 증가
    """

    if tb < -1610:
        return "DECREASE"

    elif tb > -1590:
        return "INCREASE"

    else:
        return "HOLD"


def classify_action_direction(
    action: float,
) -> str:
    """
    Action의 방향을 간단히 분류한다.

    ±0.10 이내는 HOLD로 표시한다.
    """

    if action < -0.10:
        return "DECREASE"

    elif action > 0.10:
        return "INCREASE"

    else:
        return "HOLD"


def main():

    print()
    print("=" * 100)
    print("SAC Policy Scan : TB 변화에 따른 Actor / Critic 비교")
    print("=" * 100)

    # --------------------------------------------------------
    # 학습된 4-State SAC 로드
    # --------------------------------------------------------

    model = SAC.load(
        SAC_MODEL_PATH
    )


    # --------------------------------------------------------
    # 다른 상태는 모두 고정
    #
    # TB만 변화시켜서
    # SAC가 TB 상태를 구분하는지 확인한다.
    # --------------------------------------------------------

    voltage = 43.50
    current = 5.94
    offset = 0.0


    # --------------------------------------------------------
    # 테스트할 TB
    #
    # 목표범위:
    # -1610 ~ -1590 mV
    # --------------------------------------------------------

    tb_values = [
        -1640.0,
        -1630.0,
        -1620.0,
        -1610.0,
        -1600.0,
        -1590.0,
        -1580.0,
        -1570.0,
        -1560.0,
    ]


    print()
    print(
        f"고정 조건 | "
        f"V={voltage:.2f} V | "
        f"I={current:.2f} A | "
        f"Offset={offset:+.3f} A"
    )

    print()
    print(
        f"{'TB':>8} | "
        f"{'Expected':>10} | "
        f"{'Actor':>8} | "
        f"{'Actor Dir':>10} | "
        f"{'Critic':>8} | "
        f"{'Critic Dir':>10} | "
        f"{'Best Q':>10}"
    )

    print("-" * 100)


    # --------------------------------------------------------
    # TB를 하나씩 변경하면서 검사
    # --------------------------------------------------------

    for tb in tb_values:

        observation = make_observation(
            voltage=voltage,
            current=current,
            tb=tb,
            offset=offset,
        )


        # ----------------------------------------------------
        # Actor deterministic Action
        # ----------------------------------------------------

        actor_action, _ = model.predict(
            observation,
            deterministic=True,
        )

        actor_action = float(
            np.asarray(
                actor_action
            ).reshape(-1)[0]
        )


        # ----------------------------------------------------
        # Critic이 가장 높게 평가하는 Action
        # ----------------------------------------------------

        critic_action, best_q = (
            find_best_critic_action(
                model=model,
                observation=observation,
            )
        )


        # ----------------------------------------------------
        # 방향 분류
        # ----------------------------------------------------

        expected_direction = (
            classify_expected_direction(
                tb
            )
        )

        actor_direction = (
            classify_action_direction(
                actor_action
            )
        )

        critic_direction = (
            classify_action_direction(
                critic_action
            )
        )


        # ----------------------------------------------------
        # 출력
        # ----------------------------------------------------

        print(
            f"{tb:8.1f} | "
            f"{expected_direction:>10} | "
            f"{actor_action:+8.3f} | "
            f"{actor_direction:>10} | "
            f"{critic_action:+8.3f} | "
            f"{critic_direction:>10} | "
            f"{best_q:+10.3f}"
        )


    print()
    print("=" * 100)
    print("Policy Scan 완료")
    print("=" * 100)


if __name__ == "__main__":
    main()
