"""수동 진단: 여러 Delta_V에 대한 CurrentModel의 전류 응답을 출력한다.

자동 합격·불합격을 판단하지 않는다. 제어전압 변화의 방향에 따라 `delta_i_action`과
`next_current`가 물리적으로 납득 가능한 방향과 크기를 갖는지 확인할 때 사용한다.
"""

from cathodic_rl.env_model.current_model import CurrentModel


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
