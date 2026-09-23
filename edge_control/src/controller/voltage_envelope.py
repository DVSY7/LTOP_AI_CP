"""운전 전압 범위 밖의 현재 설정값도 안전하게 범위 안으로 되돌린다."""

import math

from shared.control_core import (
    SafetyFilterResult,
    SafetyLimits,
    filter_voltage_request,
)


def filter_with_envelope_recovery(*, current_set_voltage: float,
                                  requested_delta_voltage: float,
                                  limits: SafetyLimits) -> SafetyFilterResult:
    """정상 범위에서는 공통 Safety Filter를 쓰고, 이탈 시 경계값으로 복귀한다."""

    current = float(current_set_voltage)
    requested = float(requested_delta_voltage)
    if not math.isfinite(current) or not math.isfinite(requested):
        raise ValueError("전압 제어 값은 유한한 숫자여야 합니다")

    if current < limits.min_set_voltage:
        target = limits.min_set_voltage
        reason = "current_voltage_below_min"
    elif current > limits.max_set_voltage:
        target = limits.max_set_voltage
        reason = "current_voltage_above_max"
    else:
        return filter_voltage_request(
            current_set_voltage=current,
            requested_delta_voltage=requested,
            limits=limits,
        )

    return SafetyFilterResult(
        current_set_voltage=current,
        requested_delta_voltage=requested,
        requested_set_voltage=current + requested,
        filtered_delta_voltage=target - current,
        filtered_set_voltage=target,
        intervened=True,
        reasons=(reason,),
    )
