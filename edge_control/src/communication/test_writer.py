"""승인된 시험에서 설정값 한 건만 기록하는 제한된 Modbus Writer."""

import math
from decimal import Decimal, ROUND_HALF_UP

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException


class TestWriteError(Exception):
    """단일 시험 Write를 안전하게 완료할 수 없음."""


def encode_single_register(value: float, register) -> int:
    """공학값을 가장 가까운 단일 16-bit 원시 레지스터 값으로 변환한다."""

    if register.data_type not in {"uint16", "int16"} or register.count != 1:
        raise TestWriteError(
            "단일 Write 시험은 16-bit 레지스터만 지원합니다"
        )

    if not math.isfinite(value):
        raise TestWriteError(
            "Write 값은 유한한 숫자여야 합니다"
        )

    if register.scale == 0:
        raise TestWriteError(
            "레지스터 scale은 0일 수 없습니다"
        )

    engineering_value = Decimal(str(value))
    offset = Decimal(str(register.offset))
    scale = Decimal(str(register.scale))

    raw_decimal = (
        engineering_value - offset
    ) / scale

    rounded_raw = int(
        raw_decimal.quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )

    minimum, maximum = (
        (0, 65535)
        if register.data_type == "uint16"
        else (-32768, 32767)
    )

    if not minimum <= rounded_raw <= maximum:
        raise TestWriteError(
            "원시 Write 값이 자료형 범위를 벗어납니다"
        )

    return rounded_raw & 0xFFFF


def decode_single_register(raw_value: int, register) -> float:
    """기록된 원시값을 실제 공학값으로 변환한다."""

    decoded_raw = raw_value

    if (
        register.data_type == "int16"
        and raw_value >= 0x8000
    ):
        decoded_raw = raw_value - 0x10000

    applied_value = (
        Decimal(decoded_raw)
        * Decimal(str(register.scale))
        + Decimal(str(register.offset))
    )

    return float(applied_value)


def perform_single_write(
    config,
    signal_name,
    value,
    client_factory=ModbusTcpClient,
):
    """read_write 신호에 한 번 쓰고 같은 주소를 다시 읽어 검증한다."""

    if not config.write_enabled:
        raise TestWriteError(
            "write_enabled가 false입니다"
        )

    registers = {
        register.name: register
        for register in config.registers
    }

    register = registers.get(signal_name)

    if register is None:
        raise TestWriteError(
            f"정의되지 않은 논리 신호입니다: {signal_name}"
        )

    if register.access != "read_write":
        raise TestWriteError(
            f"Write가 허용되지 않은 논리 신호입니다: {signal_name}"
        )

    if register.register_type != "holding":
        raise TestWriteError(
            "단일 Write 시험은 Holding Register만 지원합니다"
        )

    raw_value = encode_single_register(
        value,
        register,
    )

    applied_value = decode_single_register(
        raw_value,
        register,
    )

    client = client_factory(
        config.host,
        port=config.port,
        timeout=config.timeout,
        retries=config.retries,
    )

    try:
        if not client.connect():
            raise TestWriteError(
                "산업용 PC 연결 실패"
            )

        response = client.write_register(
            register.request_address,
            raw_value,
            device_id=config.unit_id,
        )

        if response is None or response.isError():
            raise TestWriteError(
                f"Modbus Write 오류 응답: {response}"
            )

        read_response = client.read_holding_registers(
            register.request_address,
            count=1,
            device_id=config.unit_id,
        )

        if (
            read_response is None
            or read_response.isError()
        ):
            raise TestWriteError(
                f"Write 후 Read-back 오류: {read_response}"
            )

        words = getattr(
            read_response,
            "registers",
            None,
        )

        if words != [raw_value]:
            raise TestWriteError(
                "Write 후 값 불일치: "
                f"expected={raw_value}, actual={words}"
            )

        return {
            "signal": signal_name,
            "requested_value": value,
            "applied_value": applied_value,
            "raw_value": raw_value,
        }

    except (OSError, ModbusException) as exc:
        raise TestWriteError(str(exc)) from exc

    finally:
        client.close()