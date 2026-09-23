"""논리 신호별 원시 레지스터와 공학값 변환."""

from datetime import datetime
import math
import struct
from zoneinfo import ZoneInfo

from ..config_loader import DATA_FORMATS
from .modbus_client import ReadError


def timestamp():
    """JSONL 시각을 한국 표준시 기준 초 단위로 반환한다."""

    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")


def decode(words, register):
    ordered = words if register.word_order == "big" else list(reversed(words))
    payload = b"".join(word.to_bytes(2, register.byte_order) for word in ordered)
    decoded = struct.unpack(">" + DATA_FORMATS[register.data_type][1], payload)[0]
    value = decoded * register.scale + register.offset
    if not math.isfinite(value):
        raise ReadError("변환값이 NaN 또는 무한대입니다")
    return value


class RegisterReader:
    def __init__(self, client, registers):
        self.client = client
        self.registers = {register.name: register for register in registers}

    def read(self, name):
        register = self.registers[name]
        # A cycle record has one timestamp in main.py.  Per-register timestamps
        # only repeat that information and make JSONL records unnecessarily large.
        sample = dict(name=name, address=register.address,
                      request_address=register.request_address,
                      register_type=register.register_type, unit=register.unit,
                      raw=None, value=None, status="error")
        try:
            sample["raw"] = self.client.read(register)
            sample["value"] = decode(sample["raw"], register)
            if register.codes is not None:
                sample["label"] = register.codes.get(sample["value"], "UNKNOWN")
                if sample["label"] == "UNKNOWN":
                    raise ReadError(f"미정의 상태 코드: {sample['value']}")
            sample["status"] = "ok"
        except ReadError as exc:
            sample["error"] = str(exc)
        return sample
