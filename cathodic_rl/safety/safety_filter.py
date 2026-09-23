"""학습 환경에서 공통 Safety Filter를 호출하는 호환 계층."""

from pathlib import Path
import sys

from cathodic_rl.config.settings import (
    MAX_OUTPUT_VOLTAGE,
    MIN_OUTPUT_VOLTAGE,
)

# 기존 `cathodic_rl` 디렉터리 직접 실행 방식을 유지하면서 저장소 공통
# 패키지를 사용한다. 패키징 구조가 정리되면 이 경로 보정은 제거할 수 있다.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shared.control_core.safety_filter import SafetyLimits, filter_voltage_request


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

    result = filter_voltage_request(
        current_set_voltage=current_voltage,
        requested_delta_voltage=requested_delta_voltage,
        limits=SafetyLimits(
            min_set_voltage=MIN_OUTPUT_VOLTAGE,
            max_set_voltage=MAX_OUTPUT_VOLTAGE,
            # 기존 학습 환경 동작 보존: 변화량 제한은 Action 변환에서 수행한다.
            max_delta_voltage=None,
        ),
    )
    return result.filtered_delta_voltage
