# tests/test_sac_q_value.py

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
# 테스트할 실제 상태
# ============================================================

TEST_STATES = {
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


# ============================================================
# SAC Action
#
# 현재 설정:
# -1.0 → -0.05V
#  0.0 →  0.00V
# +1.0 → +0.05V
# ============================================================

TEST_ACTIONS = {
    "DECREASE (-1.0)": -1.0,
    "HOLD      (0.0)":  0.0,
    "INCREASE (+1.0)": +1.0,
}


def normalize_state(
    voltage: float,
    current: float,
    tb: float,
) -> np.ndarray:
    """
    cathodic_env.py의 SAC Observation 정규화와
    동일한 방식으로 상태를 0~1 범위로 변환한다.
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

    normalized_potential = (
        tb - MIN_PIPE_POTENTIAL
    ) / (
        MAX_PIPE_POTENTIAL - MIN_PIPE_POTENTIAL
    )

    return np.array(
        [
            normalized_voltage,
            normalized_current,
            normalized_potential,
        ],
        dtype=np.float32,
    )


def test_sac_q_value():

    print()
    print("=" * 80)
    print("SAC Critic Q-Value 테스트")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 현재 저장된 SAC 모델 불러오기
    # --------------------------------------------------------

    model = SAC.load(
        SAC_MODEL_PATH
    )

    print()
    print(
        f"SAC 모델 로드 완료: "
        f"{SAC_MODEL_PATH}"
    )


    # --------------------------------------------------------
    # 2. BELOW / ABOVE 각각 테스트
    # --------------------------------------------------------

    for state_name, state in TEST_STATES.items():

        print()
        print("=" * 80)
        print(f"상태: {state_name}")

        print(
            f"실제 상태 | "
            f"V={state['voltage']:.2f} V | "
            f"I={state['current']:.2f} A | "
            f"TB={state['tb']:.1f} mV"
        )

        # ----------------------------------------------------
        # 환경과 동일하게 Observation 정규화
        # ----------------------------------------------------

        observation = normalize_state(
            voltage=state["voltage"],
            current=state["current"],
            tb=state["tb"],
        )

        print(
            "SAC Observation | "
            f"{observation}"
        )


        # ----------------------------------------------------
        # numpy → PyTorch Tensor
        #
        # shape:
        # (3,) → (1, 3)
        # ----------------------------------------------------

        obs_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=model.device,
        ).unsqueeze(0)


        # ----------------------------------------------------
        # 현재 Actor가 실제로 선택하는 Action도 확인
        # ----------------------------------------------------

        predicted_action, _ = model.predict(
            observation,
            deterministic=True,
        )

        print()
        print(
            "Actor deterministic Action: "
            f"{float(np.asarray(predicted_action).item()):+.4f}"
        )

        print()
        print("-" * 80)
        print(
            f"{'Action':<22} | "
            f"{'Q1':>12} | "
            f"{'Q2':>12} | "
            f"{'Min Q':>12}"
        )
        print("-" * 80)


        results = []


        # ----------------------------------------------------
        # 3. 동일 State에서
        #    -1 / 0 / +1 Action의 Q-value 비교
        # ----------------------------------------------------

        with torch.no_grad():

            for action_name, action_value in TEST_ACTIONS.items():

                action_tensor = torch.tensor(
                    [[action_value]],
                    dtype=torch.float32,
                    device=model.device,
                )

                # SAC Critic은 Twin Q Network 사용
                q_values = model.critic(
                    obs_tensor,
                    action_tensor,
                )

                q1 = float(
                    q_values[0]
                    .cpu()
                    .item()
                )

                q2 = float(
                    q_values[1]
                    .cpu()
                    .item()
                )

                # SAC에서는 보수적으로 두 Q 중 작은 값을 사용
                min_q = min(
                    q1,
                    q2,
                )

                results.append(
                    {
                        "action_name": action_name,
                        "q1": q1,
                        "q2": q2,
                        "min_q": min_q,
                    }
                )

                print(
                    f"{action_name:<22} | "
                    f"{q1:>+12.4f} | "
                    f"{q2:>+12.4f} | "
                    f"{min_q:>+12.4f}"
                )


        # ----------------------------------------------------
        # 4. Critic이 가장 높게 평가한 Action 표시
        # ----------------------------------------------------

        best_result = max(
            results,
            key=lambda x: x["min_q"],
        )

        print("-" * 80)

        print(
            "Critic 최고 평가 Action: "
            f"{best_result['action_name']}"
        )


if __name__ == "__main__":
    test_sac_q_value()