from env_model.environment_model import EnvironmentModel
import random


# ============================================================
# Rollout 설정
# ============================================================

NUM_STEPS = 144

INITIAL_VOLTAGE = 43.5608
INITIAL_CURRENT = 5.9100

INITIAL_TB_HISTORY = [
    -1602.0,
    -1600.0,
    -1599.0,
    -1598.0,
    -1595.0,
    -1597.0,
    -1593.0,
]


# ============================================================
# Rollout 실행 함수
# ============================================================

def run_rollout(
    case_name,
    action_function,
):
    """
    하나의 Action 정책으로
    EnvironmentModel을 144 Step 연속 실행한다.
    """

    env_model = EnvironmentModel()

    env_model.reset_history(
        INITIAL_TB_HISTORY
    )

    V_t = INITIAL_VOLTAGE
    I_t = INITIAL_CURRENT

    voltage_history = [V_t]
    current_history = [I_t]
    tb_history = [
        env_model.tb_history[-1]
    ]
    action_history = []


    print()
    print(
        f"===== {case_name} 시작 ====="
    )


    # --------------------------------------------------------
    # 144 Step Rollout
    # --------------------------------------------------------

    for step in range(
        1,
        NUM_STEPS + 1,
    ):

        delta_v = action_function(
            step
        )

        result = (
            env_model.predict_next_state(
                V_t=V_t,
                I_t=I_t,
                delta_v=delta_v,
            )
        )


        # ----------------------------------------------------
        # 결과 저장
        # ----------------------------------------------------

        action_history.append(
            delta_v
        )

        voltage_history.append(
            result["next_voltage"]
        )

        current_history.append(
            result["next_current"]
        )

        tb_history.append(
            result["next_tb"]
        )


        # ----------------------------------------------------
        # 다음 Step 상태 업데이트
        # ----------------------------------------------------

        V_t = result[
            "next_voltage"
        ]

        I_t = result[
            "next_current"
        ]


        # ----------------------------------------------------
        # 24 Step마다 상태 출력
        # 24 Step = 약 4시간
        # ----------------------------------------------------

        if step % 24 == 0:

            print(
                f"Step {step:3d} | "
                f"V={V_t:.4f} V | "
                f"I={I_t:.4f} A | "
                f"TB={result['next_tb']:.3f} mV"
            )


    # ========================================================
    # Rollout 요약
    # ========================================================

    print()
    print(
        f"===== {case_name} 결과 ====="
    )

    print(
        f"Voltage"
        f" | min={min(voltage_history):.4f}"
        f" | max={max(voltage_history):.4f}"
        f" | final={voltage_history[-1]:.4f}"
    )

    print(
        f"Current"
        f" | min={min(current_history):.4f}"
        f" | max={max(current_history):.4f}"
        f" | final={current_history[-1]:.4f}"
    )

    print(
        f"TB"
        f" | min={min(tb_history):.3f}"
        f" | max={max(tb_history):.3f}"
        f" | final={tb_history[-1]:.3f}"
    )

    print(
        f"Action"
        f" | min={min(action_history):+.3f}"
        f" | max={max(action_history):+.3f}"
    )

    print()


# ============================================================
# Case 1
# Delta_V = 0.00 V 고정
# ============================================================

def case_1_action(step):

    return 0.00


# ============================================================
# Case 2
# Delta_V = +0.05 V 고정
# ============================================================

def case_2_action(step):

    return 0.05


# ============================================================
# Case 3
# Delta_V = -0.05 ~ +0.05 V 랜덤
# ============================================================

def case_3_action(step):

    return random.uniform(
        -0.05,
        0.05,
    )


# ============================================================
# 테스트 실행
# ============================================================

if __name__ == "__main__":

    random.seed(
        42
    )

    run_rollout(
        case_name=(
            "Case 1 : Delta_V = 0.00 V"
        ),
        action_function=case_1_action,
    )

    run_rollout(
        case_name=(
            "Case 2 : Delta_V = +0.05 V"
        ),
        action_function=case_2_action,
    )

    run_rollout(
        case_name=(
            "Case 3 : Delta_V Random"
        ),
        action_function=case_3_action,
    )