# tests/test_sac_q_curve.py

import numpy as np
import torch

from stable_baselines3 import SAC

from config.settings import (
    SAC_MODEL_PATH,
    MIN_OUTPUT_VOLTAGE,
    MAX_OUTPUT_VOLTAGE,
    MIN_OUTPUT_CURRENT,
    MAX_OUTPUT_CURRENT,
    MIN_PIPE_POTENTIAL,
    MAX_PIPE_POTENTIAL,
)


# ============================================================
# 현재 5-State Observation에서 사용하는 정규화 범위
# ============================================================

OFFSET_MIN = -0.20
OFFSET_MAX = +0.20

TB_TREND_MIN = -10.0
TB_TREND_MAX = +10.0


def make_observation(
    voltage: float,
    current: float,
    tb: float,
    offset: float,
    tb_trend: float,
) -> np.ndarray:
    """
    실제 상태:
    [V, I, TB, Offset, TB Trend]

    ↓

    SAC 입력용 0~1 정규화 Observation
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

    normalized_offset = (
        offset - OFFSET_MIN
    ) / (
        OFFSET_MAX - OFFSET_MIN
    )

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
            normalized_offset,
            normalized_tb_trend,
        ],
        dtype=np.float32,
    )


def get_q_values(
    model,
    observation,
    action,
):
    """
    Twin Critic의 Q1, Q2와
    SAC에서 사용할 보수적인 Min Q를 반환한다.
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

    min_q = min(q1, q2)

    return q1, q2, min_q


def test_case(
    model,
    name,
    voltage,
    current,
    tb,
    offset,
    tb_trend,
):
    """
    하나의 상태에 대해
    Action -1 ~ +1 전체 Q 곡선을 확인한다.
    """

    observation = make_observation(
        voltage=voltage,
        current=current,
        tb=tb,
        offset=offset,
        tb_trend=tb_trend,
    )

    # --------------------------------------------------------
    # 현재 Actor가 실제로 선택하는 Action
    # --------------------------------------------------------

    actor_action, _ = model.predict(
        observation,
        deterministic=True,
    )

    actor_action = float(
        np.asarray(actor_action).reshape(-1)[0]
    )

    print()
    print("=" * 85)
    print(
        f"CASE {name} | "
        f"V={voltage:.2f} | "
        f"I={current:.2f} | "
        f"TB={tb:.1f} | "
        f"Offset={offset:+.3f} | "
        f"Trend={tb_trend:+.1f}"
    )
    print("=" * 85)

    print(
        f"Observation : {observation}"
    )

    print(
        f"Actor Action: {actor_action:+.4f}"
    )

    print()
    print(
        f"{'Action':>8} | "
        f"{'Q1':>10} | "
        f"{'Q2':>10} | "
        f"{'Min Q':>10}"
    )

    print("-" * 50)

    # --------------------------------------------------------
    # -1.0 ~ +1.0
    # 0.1 간격으로 Critic Q 확인
    # --------------------------------------------------------

    actions = np.linspace(
        -1.0,
        1.0,
        21,
    )

    results = []

    for action in actions:

        q1, q2, min_q = get_q_values(
            model=model,
            observation=observation,
            action=float(action),
        )

        results.append(
            (
                float(action),
                q1,
                q2,
                min_q,
            )
        )

        print(
            f"{action:+8.2f} | "
            f"{q1:+10.4f} | "
            f"{q2:+10.4f} | "
            f"{min_q:+10.4f}"
        )


    # --------------------------------------------------------
    # Critic이 가장 높게 평가한 Action
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda x: x[3],
    )

    best_action = best_result[0]
    best_q = best_result[3]


    # --------------------------------------------------------
    # Actor가 선택한 실제 Action의 Q도 계산
    #
    # Critic 최고점과 Actor 위치가
    # 얼마나 차이나는지 확인하기 위함.
    # --------------------------------------------------------

    _, _, actor_q = get_q_values(
        model=model,
        observation=observation,
        action=actor_action,
    )

    q_gap = best_q - actor_q

    print()
    print(
        f"Critic Best Action : "
        f"{best_action:+.2f}"
    )

    print(
        f"Critic Best Min Q  : "
        f"{best_q:+.4f}"
    )

    print(
        f"Actor Action       : "
        f"{actor_action:+.4f}"
    )

    print(
        f"Q @ Actor          : "
        f"{actor_q:+.4f}"
    )

    print(
        f"Best Q - Actor Q   : "
        f"{q_gap:+.4f}"
    )


def main():

    print()
    print("=" * 85)
    print("5-State SAC Critic Q Curve 분석")
    print("=" * 85)

    model = SAC.load(
        SAC_MODEL_PATH
    )


    # ========================================================
    # 모든 조건을 동일하게 고정
    #
    # V       = 43.50 V
    # I       = 5.94 A
    # Offset  = 0 A
    # TB Trend= 0 mV
    #
    # TB 위치만 바꿔서 비교
    # ========================================================

    voltage = 43.50
    current = 5.94
    offset = 0.0
    tb_trend = 0.0


    # --------------------------------------------------------
    # 1. BELOW
    # --------------------------------------------------------

    test_case(
        model=model,
        name="BELOW",
        voltage=voltage,
        current=current,
        tb=-1640.0,
        offset=offset,
        tb_trend=tb_trend,
    )


    # --------------------------------------------------------
    # 2. TARGET
    # --------------------------------------------------------

    test_case(
        model=model,
        name="TARGET",
        voltage=voltage,
        current=current,
        tb=-1600.0,
        offset=offset,
        tb_trend=tb_trend,
    )


    # --------------------------------------------------------
    # 3. ABOVE
    # --------------------------------------------------------

    test_case(
        model=model,
        name="ABOVE",
        voltage=voltage,
        current=current,
        tb=-1560.0,
        offset=offset,
        tb_trend=tb_trend,
    )


    print()
    print("=" * 85)
    print("Q Curve 분석 완료")
    print("=" * 85)


if __name__ == "__main__":
    main()