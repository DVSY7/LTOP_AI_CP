"""전기방식 제어 Action에 적용하는 안전 필터."""

import numpy as np

from config.settings import (
    MAX_OUTPUT_VOLTAGE,
    MIN_OUTPUT_VOLTAGE,
)


def filter_voltage_action(
    current_voltage: float,
    requested_delta_voltage: float,
) -> float:
    """출력전압 범위를 벗어나지 않는 안전한 변화량을 반환한다.

    Args:
        current_voltage:
            현재 정류기 출력전압 [V].
        requested_delta_voltage:
            Agent가 요청한 출력전압 변화량 [V].

    Returns:
        실제로 적용할 수 있는 안전한 전압 변화량 [V].
    """

    requested_voltage = (
        current_voltage + requested_delta_voltage
    )

    safe_voltage = float(
        np.clip(
            requested_voltage,
            MIN_OUTPUT_VOLTAGE,
            MAX_OUTPUT_VOLTAGE,
        )
    )

    return safe_voltage - current_voltage