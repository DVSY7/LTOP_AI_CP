"""출력전압 Safety Filter 테스트."""
import _path_setup

import numpy as np

from safety.safety_filter import filter_voltage_action


def test_voltage_safety_filter() -> None:
    """정상 범위와 상한·하한에서 Action 제한을 확인한다."""

    test_cases = [
        {
            "name": "정상 범위 증가",
            "current": 10.0,
            "requested": 0.5,
            "expected": 0.5,
        },
        {
            "name": "정상 범위 감소",
            "current": 10.0,
            "requested": -0.5,
            "expected": -0.5,
        },
        {
            "name": "최대전압 초과 차단",
            "current": 60.0,
            "requested": 0.5,
            "expected": 0.0,
        },
        {
            "name": "최소전압 미만 차단",
            "current": 0.0,
            "requested": -0.5,
            "expected": 0.0,
        },
        {
            "name": "최대전압 일부 초과",
            "current": 59.8,
            "requested": 0.5,
            "expected": 0.2,
        },
    ]

    print("\n[출력전압 Safety Filter 테스트]")

    for case in test_cases:
        actual = filter_voltage_action(
            current_voltage=case["current"],
            requested_delta_voltage=case["requested"],
        )

        assert np.isclose(actual, case["expected"])

        print(
            f"{case['name']}\n"
            f"  현재전압: {case['current']:.1f} V\n"
            f"  요청 변화량: {case['requested']:+.1f} V\n"
            f"  적용 변화량: {actual:+.1f} V\n"
            f"  결과: 통과"
        )


if __name__ == "__main__":
    test_voltage_safety_filter()

    print("\n모든 Safety Filter 테스트 통과")