"""수동 진단: TBModel의 자연 변화와 제어전류 보정 효과를 분리해 출력한다."""

from cathodic_rl.env_model.tb_model import TBModel


# ============================================================
# TB Model 불러오기
# ============================================================

tb_model = TBModel()


# ============================================================
# 테스트용 TB History
# ============================================================

TB_t = -1593.0

TB_lag1 = -1597.0
TB_lag2 = -1595.0
TB_lag3 = -1598.0
TB_lag6 = -1602.0


# ============================================================
# 테스트할 Action Delta_I
# ============================================================

test_delta_i_actions = [
    -0.20,
    -0.10,
    0.00,
    0.10,
    0.20,
]


# ============================================================
# TB Model 테스트
# ============================================================

print(
    "\n===== TB Model 테스트 ====="
)

for delta_i_action in test_delta_i_actions:

    result = tb_model.predict(
        TB_t=TB_t,
        TB_lag1=TB_lag1,
        TB_lag2=TB_lag2,
        TB_lag3=TB_lag3,
        TB_lag6=TB_lag6,
        delta_i_action=delta_i_action,
    )

    print(
        f"\nAction ΔI = "
        f"{delta_i_action:+.3f} A"
    )

    print(
        f"  자연 TB 예측      : "
        f"{result['tb_natural_next']:.3f} mV"
    )

    print(
        f"  제어 TB 보정      : "
        f"{result['delta_tb_control']:+.3f} mV"
    )

    print(
        f"  최종 TB 예측      : "
        f"{result['tb_next']:.3f} mV"
    )
