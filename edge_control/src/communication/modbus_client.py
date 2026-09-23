"""읽기 메서드만 노출하는 Modbus TCP 어댑터."""

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException


class ReadError(Exception):
    """해당 수집 주기를 무효 처리해야 하는 통신/데이터 오류."""


class ReadOnlyClient:
    def __init__(self, config, client_factory=ModbusTcpClient):
        self._client = client_factory(config.host, port=config.port,
                                      timeout=config.timeout, retries=config.retries)
        self._unit_id = config.unit_id
        self._connected = False

    def read(self, register):
        try:
            if not self._connected:
                self._connected = bool(self._client.connect())
                if not self._connected:
                    raise ReadError("산업용 PC 연결 실패")
            method = (self._client.read_holding_registers
                      if register.register_type == "holding"
                      else self._client.read_input_registers)
            response = method(register.request_address, count=register.count,
                              device_id=self._unit_id)
            if response is None or response.isError():
                raise ReadError(f"Modbus 오류 응답: {response}")
            words = getattr(response, "registers", None)
            if not isinstance(words, (list, tuple)) or len(words) != register.count:
                raise ReadError("레지스터 응답 길이 불일치")
            if any(type(word) is not int or not 0 <= word <= 65535 for word in words):
                raise ReadError("유효하지 않은 원시 레지스터 값")
            return list(words)
        except (OSError, ModbusException) as exc:
            raise ReadError(str(exc)) from exc

    def close(self):
        self._connected = False
        self._client.close()
