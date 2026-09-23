# tests/test_sac_q_value.py

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
    실제 물리값
    [V, I, TB, TB Trend]

    ↓

    SAC가 사용하는 정규화 Observation
    [V_norm, I_norm, TB_norm, TB_Trend_norm]
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
    normalized_tb_trend = 0.5

    return np.array(
        [
            normalized_voltage,
            normalized_current,
            normalized_tb,
            normalized_tb_trend,
        ],
        dtype=np.float32,
    )


def get_q_values(
    model,
    observation: np.ndarray,
    action_value: float,
):
    """
    하나의 Observation과 Action에 대해
    SAC Critic의 Q1, Q2를 확인한다.
    """

    obs_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    action_tensor = torch.tensor(
        [[action_value]],
        dtype=torch.float32,
    )

    with torch.no_grad():

        q_values = model.critic(
            obs_tensor,
            action_tensor,
        )

        q1 = float(
            q_values[0].cpu().numpy().flatten()[0]
        )

        q2 = float(
            q_values[1].cpu().numpy().flatten()[0]
        )

    return q1, q2


def run_case(
    model,
    case_name: str,
    voltage: float,
    current: float,
    tb: float,
    offset: float,
):

    print()
    print("=" * 80)
    print(f"{case_name}")
    print("=" * 80)

    print(
        f"실제 상태 | "
        f"V={voltage:.2f} V | "
        f"I={current:.2f} A | "
        f"TB={tb:.1f} mV | "
        f"Offset={offset:+.3f} A"
    )


    # --------------------------------------------------------
    # Observation 생성
    # --------------------------------------------------------

    observation = make_observation(
        voltage=voltage,
        current=current,
        tb=tb,
        offset=offset,
    )

    print()
    print(
        "Observation:",
        observation,
    )


    # --------------------------------------------------------
    # Actor의 deterministic Action 확인
    # --------------------------------------------------------

    actor_action, _ = model.predict(
        observation,
        deterministic=True,
    )

    actor_action = float(
        np.asarray(actor_action).reshape(-1)[0]
    )

    print()
    print(
        f"Actor Action = "
        f"{actor_action:+.4f}"
    )


    # --------------------------------------------------------
    # Critic Q-value 비교
    #
    # -1 = 최대 전압 감소
    #  0 = 유지
    # +1 = 최대 전압 증가
    # --------------------------------------------------------

    actions = {
        "DECREASE": -1.0,
        "HOLD": 0.0,
        "INCREASE": +1.0,
    }

    print()
    print(
        "Critic Q-value"
    )

    results = {}

    for action_name, action_value in actions.items():

        q1, q2 = get_q_values(
            model=model,
            observation=observation,
            action_value=action_value,
        )

        min_q = min(
            q1,
            q2,
        )

        results[action_name] = min_q

        print(
            f"{action_name:<10} | "
            f"Action={action_value:+.1f} | "
            f"Q1={q1:+.4f} | "
            f"Q2={q2:+.4f} | "
            f"MinQ={min_q:+.4f}"
        )


    # --------------------------------------------------------
    # Critic이 가장 높게 평가하는 방향
    # --------------------------------------------------------

    best_action = max(
        results,
        key=results.get,
    )

    print()
    print(
        f"Critic Best Action = "
        f"{best_action}"
    )


def main():

    print()
    print("=" * 80)
    print("4-State SAC Q-value 진단")
    print("=" * 80)


    # --------------------------------------------------------
    # 새로 학습한 4-state SAC 모델 로드
    # --------------------------------------------------------

    model = SAC.load(
        SAC_MODEL_PATH
    )


    # ========================================================
    # CASE 1
    #
    # BELOW
    #
    # TB=-1618 mV
    # 목표보다 너무 음(-)이므로
    # 우리가 기대하는 방향은 DECREASE
    # ========================================================

    run_case(
        model=model,
        case_name="CASE 1 : BELOW",
        voltage=43.50,
        current=5.94,
        tb=-1618.0,
        offset=0.0,
    )


    # ========================================================
    # CASE 2
    #
    # ABOVE
    #
    # TB=-1568 mV
    # 목표보다 덜 음(-)이므로
    # 우리가 기대하는 방향은 INCREASE
    # ========================================================

    run_case(
        model=model,
        case_name="CASE 2 : ABOVE",
        voltage=43.50,
        current=5.94,
        tb=-1568.0,
        offset=0.0,
    )


if __name__ == "__main__":
    main()
