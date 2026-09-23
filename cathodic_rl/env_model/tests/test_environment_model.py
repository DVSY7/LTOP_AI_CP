"""수동 진단: 결합 환경모델을 여러 Step 실행하며 History 갱신을 확인한다."""

from cathodic_rl.env_model.environment_model import EnvironmentModel


# ============================================================
# Environment Model 불러오기
# ============================================================

env_model = EnvironmentModel()


# ============================================================
# 초기 TB History 설정
# ============================================================

initial_tb_history = [
    -1602.0,
    -1600.0,
    -1599.0,
    -1598.0,
    -1595.0,
    -1597.0,
    -1593.0,
]

env_model.reset_history(
    initial_tb_history
)


# ============================================================
# 초기 상태
# ============================================================

V_t = 43.5608
I_t = 5.9100


# ============================================================
# 테스트할 Delta_V
# ============================================================

test_actions = [
    -0.20,
    -0.10,
    0.00,
    0.10,
    0.20,
]


# ============================================================
# Environment Model 연속 Step 테스트
# ============================================================

print(
    "\n===== Environment Model Multi-Step 테스트 ====="
)

print(
    "\n초기 TB History :"
)

print(
    env_model.tb_history
)


# ============================================================
# 여러 Step 연속 실행
# ============================================================

for step, delta_v in enumerate(
    test_actions,
    start=1,
):

    result = env_model.predict_next_state(
        V_t=V_t,
        I_t=I_t,
        delta_v=delta_v,
    )


    print(
        f"\n===== Step {step} ====="
    )

    print(
        f"Delta_V           : "
        f"{delta_v:+.2f} V"
    )

    print(
        f"현재 TB           : "
        f"{result['current_tb']:.3f} mV"
    )

    print(
        f"다음 전압         : "
        f"{result['next_voltage']:.4f} V"
    )

    print(
        f"전체 Delta_I      : "
        f"{result['delta_i_total']:+.4f} A"
    )

    print(
        f"Baseline Delta_I  : "
        f"{result['delta_i_baseline']:+.4f} A"
    )

    print(
        f"Action Delta_I    : "
        f"{result['delta_i_action']:+.4f} A"
    )

    print(
        f"다음 전류         : "
        f"{result['next_current']:.4f} A"
    )

    print(
        f"자연 TB 예측      : "
        f"{result['tb_natural_next']:.3f} mV"
    )

    print(
        f"TB 제어 보정      : "
        f"{result['delta_tb_control']:+.3f} mV"
    )

    print(
        f"최종 다음 TB      : "
        f"{result['next_tb']:.3f} mV"
    )

    print(
        "현재 TB History   :"
    )

    print(
        env_model.tb_history
    )


    # --------------------------------------------------------
    # 다음 Step의 현재 상태로 업데이트
    # --------------------------------------------------------

    V_t = result[
        "next_voltage"
    ]

    I_t = result[
        "next_current"
    ]
