from env_model.current_model import CurrentModel


# ============================================================
# Current Model 불러오기
# ============================================================

current_model = CurrentModel()


# ============================================================
# 테스트 상태
# ============================================================

V_t = 43.5608
I_t = 5.9100


# 여러 Delta_V에 대한 모델 반응 확인
test_actions = [
    -0.20,
    -0.10,
    0.00,
    0.10,
    0.20,
]


# ============================================================
# 예측 결과 확인
# ============================================================

print("\n===== Current Model 테스트 =====")

for delta_v in test_actions:

    result = current_model.predict(
        V_t=V_t,
        I_t=I_t,
        delta_v=delta_v,
    )

    print(
        f"\nΔV = {delta_v:+.2f} V"
    )

    print(
        f"  전체 ΔI     : "
        f"{result['delta_i_total']:+.4f} A"
    )

    print(
        f"  Baseline ΔI : "
        f"{result['delta_i_baseline']:+.4f} A"
    )

    print(
        f"  Action ΔI   : "
        f"{result['delta_i_action']:+.4f} A"
    )

    print(
        f"  다음 전류    : "
        f"{result['next_current']:.4f} A"
    )