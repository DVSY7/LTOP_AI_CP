# cathodic_rl/test_setting.py - 파일경로

# 설정 값을 가지고 있는 파일에서 정상적으로 임포트 되는지 확인하는 코드입니다.
import _path_setup

from config.settings import (
    ACTION_TO_DELTA_VOLTAGE,
    MAX_OUTPUT_VOLTAGE,
    MIN_OUTPUT_VOLTAGE,
    TARGET_POTENTIAL_MAX,
    TARGET_POTENTIAL_MIN,
)


def test_setting() -> None:
    print("출력전압 범위:")
    print(f"{MIN_OUTPUT_VOLTAGE} V ~ {MAX_OUTPUT_VOLTAGE} V")

    print("\n목표 방식전위 범위:")
    print(f"{TARGET_POTENTIAL_MIN} mV ~ {TARGET_POTENTIAL_MAX} mV")

    print("\n행동별 전압 변화량:")
    for action, delta_voltage in ACTION_TO_DELTA_VOLTAGE.items():
        print(f"행동 {action}: {delta_voltage:+.1f} V")


if __name__ == "__main__":
    test_setting()