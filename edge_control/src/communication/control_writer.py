"""자동 제어 요청을 Holding Register에 쓰고 Read-back으로 검증한다."""

from .test_writer import TestWriteError, perform_single_write


def write_control_target(config, target_set_voltage: float) -> dict:
    """설정된 제어 신호에 한 번 Write하고, 같은 주소의 Read-back을 확인한다."""

    if config.control is None or not config.control.enabled:
        raise TestWriteError("자동 제어 Write가 비활성화되어 있습니다")

    result = perform_single_write(
        config,
        config.control.write_signal,
        target_set_voltage,
    )
    return {
        **result,
        "actual_write_voltage": target_set_voltage,
        "read_back_verified": True,
    }
