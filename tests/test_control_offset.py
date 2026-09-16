# tests/test_control_offset.py

from env_model.environment_model import EnvironmentModel


# ============================================================
# 동일한 외부 상태
# ============================================================

TEST_VOLTAGE = 43.60
TEST_CURRENT = 5.94
TEST_TB = -1600.0

# 모든 CASE에서 동일한 TB History 사용
TEST_HISTORY = [
    -1600.0,
    -1600.0,
    -1600.0,
    -1600.0,
    -1600.0,
    -1600.0,
    -1600.0,
]

# control_current_offset만 다르게 설정
TEST_OFFSETS = [
    -0.10,
    -0.05,
     0.00,
    +0.05,
    +0.10,
]


def test_control_offset():

    print()
    print("=" * 75)
    print("Control Current Offset 영향 테스트")
    print("=" * 75)

    print(
        f"동일 상태 | "
        f"V={TEST_VOLTAGE:.2f} V | "
        f"I={TEST_CURRENT:.2f} A | "
        f"TB={TEST_TB:.1f} mV"
    )

    print()
    print(
        "Offset      | TB Natural | Control Effect | TB Next"
    )
    print("-" * 75)

    for offset in TEST_OFFSETS:

        # CASE마다 새로운 EnvironmentModel 생성
        # → 이전 CASE의 History/Offset 영향을 완전히 제거
        model = EnvironmentModel()

        # 동일한 TB History 설정
        model.reset_history(
            TEST_HISTORY
        )

        # 이번 테스트에서 확인하려는 값만 강제로 변경
        model.control_current_offset = offset

        # Action은 0V
        #
        # 즉 새로운 제어 명령은 전혀 주지 않고,
        # 기존 offset만으로 TB가 달라지는지 확인한다.
        result = model.predict_next_state(
            V_t=TEST_VOLTAGE,
            I_t=TEST_CURRENT,
            delta_v=0.0,
        )

        print(
            f"{offset:+.3f} A   | "
            f"{result['tb_natural_next']:+10.3f} | "
            f"{result['delta_tb_control']:+14.3f} | "
            f"{result['next_tb']:+10.3f}"
        )


if __name__ == "__main__":
    test_control_offset()