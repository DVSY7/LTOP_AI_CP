"""장비별 설정을 YAML에서 읽고 검증한다."""

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from shared.control_core import SafetyLimits


class ConfigError(ValueError):
    """미확정 또는 잘못된 설정."""


DATA_FORMATS = {"uint16": (1, "H"), "int16": (1, "h"),
                "uint32": (2, "I"), "int32": (2, "i"), "float32": (2, "f")}


@dataclass(frozen=True)
class Register:
    """YAML의 한 논리 신호를 Modbus 읽기·쓰기 사양으로 표현한다."""
    name: str
    address: int
    request_address: int
    register_type: str
    data_type: str
    byte_order: str
    word_order: str
    scale: float
    offset: float
    unit: str
    codes: dict[int, str] | None = None
    access: str = "read"

    @property
    def count(self):
        return DATA_FORMATS[self.data_type][0]


@dataclass(frozen=True)
class Config:
    """YAML 검증을 통과한 edge 실행 설정의 불변 묶음이다."""
    profile: str
    write_enabled: bool
    host: str
    port: int
    unit_id: int
    timeout: float
    retries: int
    poll_interval: float
    reconnect_interval: float
    log_file: Path
    max_bytes: int
    backup_count: int
    record_interval: float
    registers: tuple[Register, ...]
    safety_limits: SafetyLimits | None
    ai_enabled: bool
    ai_shadow_mode: bool
    ai_deterministic: bool
    model_metadata_file: Path | None
    control: "ControlSettings | None" = None


@dataclass(frozen=True)
class ExploreSettings:
    """EXPLORE 전압 범위, 변화 폭, Hold 패턴 설정이다."""
    pattern: str
    min_voltage: float
    max_voltage: float
    step_voltage: float
    hold_every_steps: int
    hold_steps: int


@dataclass(frozen=True)
class ControlSettings:
    """자동 Write 대상과 Write 간격, EXPLORE 설정이다."""
    enabled: bool
    write_signal: str
    write_interval_seconds: float
    explore: ExploreSettings | None


def load_config(path):
    """YAML을 읽어 형식·범위·레지스터 접근 권한을 검증한 Config로 반환한다."""
    path = Path(path).resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"설정 파일 읽기 실패: {exc}") from exc
    errors = []

    def get(key, kind, *, choices=None, minimum=None, maximum=None):
        value = data
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        valid = isinstance(value, kind) and not isinstance(value, bool)
        if kind is bool:
            valid = type(value) is bool
        if valid and isinstance(value, (int, float)):
            valid = math.isfinite(value)
        if valid and kind is str:
            valid = bool(value.strip())
        if valid and choices is not None:
            valid = value in choices
        if valid and minimum is not None:
            valid = value >= minimum
        if valid and maximum is not None:
            valid = value <= maximum
        if not valid:
            errors.append(f"{key}: 미확정 또는 잘못된 값 ({value!r})")
            return None
        return value

    if get("enabled", bool) is not True:
        errors.append("enabled: 비활성 프로파일은 실행할 수 없습니다")
    write_enabled = get("write_enabled", bool)
    profile = get("profile", str)
    host = get("connection.host", str)
    port = get("connection.port", int, minimum=1, maximum=65535)
    unit_id = get("connection.unit_id", int, minimum=0, maximum=255)
    timeout = get("connection.timeout_seconds", (int, float), minimum=0.001)
    retries = get("connection.retries", int, minimum=0)
    base = get("address_base", int, choices=(0, 1))
    poll = get("poll_interval_seconds", (int, float), minimum=0.001)
    reconnect = get("reconnect_interval_seconds", (int, float), minimum=0.001)
    log_file = get("logging.file", str)
    max_bytes = get("logging.max_bytes", int, minimum=1)
    backups = get("logging.backup_count", int, minimum=1)
    logging_data = data.get("logging") if isinstance(data, dict) else None
    record_interval = poll
    if isinstance(logging_data, dict) and "record_interval_seconds" in logging_data:
        configured_interval = get(
            "logging.record_interval_seconds", (int, float), minimum=0.001)
        if configured_interval is not None:
            record_interval = float(configured_interval)
    ai_enabled = False
    ai_shadow_mode = True
    ai_deterministic = True
    model_metadata_file = None
    ai_data = data.get("ai") if isinstance(data, dict) else None
    if ai_data is not None:
        if not isinstance(ai_data, dict):
            errors.append("ai: AI 설정 매핑이어야 합니다")
        else:
            ai_enabled = get("ai.enabled", bool)
            ai_shadow_mode = get("ai.shadow_mode", bool)
            ai_deterministic = get("ai.deterministic", bool)
            metadata_file = get("ai.metadata_file", str)
            if metadata_file is not None:
                model_metadata_file = (path.parent / metadata_file).resolve()
    safety_limits = None
    safety_data = data.get("safety") if isinstance(data, dict) else None
    if safety_data is not None:
        if not isinstance(safety_data, dict):
            errors.append("safety: 설정전압 한계 매핑이어야 합니다")
        else:
            min_voltage = get("safety.min_set_voltage", (int, float))
            max_voltage = get("safety.max_set_voltage", (int, float))
            max_delta = get("safety.max_delta_voltage", (int, float), minimum=0.000001)
            if min_voltage is not None and max_voltage is not None and max_delta is not None:
                try:
                    safety_limits = SafetyLimits(
                        min_set_voltage=float(min_voltage),
                        max_set_voltage=float(max_voltage),
                        max_delta_voltage=float(max_delta),
                    )
                except ValueError as exc:
                    errors.append(f"safety: {exc}")
    registers = []
    mode_labels = {
        "remote_mode": {"LOCAL", "REMOTE"},
        "operation_status": {"OFF", "ON"},
        "operation_mode": {"AI", "MANUAL", "EXPLORE"},
        "tb_status": {"NORMAL", "ERR_BAT", "ERR_COM", "ERR_VT"}
    }
    configured = data.get("registers", {}) if isinstance(data, dict) else {}
    if not isinstance(configured, dict):
        raise ConfigError("registers는 논리 이름별 매핑이어야 합니다")
    # 전압/전류는 모든 수집 프로파일의 필수값이다. 나머지는 장비별로
    # 정의된 경우에만 읽는다. 운전 상태 3종은 일부만 빠지면 안전한
    # 모드 판정이 불가능하므로 한 항목이 있으면 세 항목을 모두 요구한다.
    names = ["rectifier_voltage", "rectifier_current"]
    for optional_name in ("set_voltage", "tb_potential", "tb_status"):
        if optional_name in configured:
            names.append(optional_name)
    operation_names = ("remote_mode", "operation_status", "operation_mode")
    if any(name in configured for name in operation_names):
        names.extend(operation_names)
    
    for name in names:
        prefix = f"registers.{name}."
        address = get(prefix + "address", int, minimum=0, maximum=65536)
        reg_type = get(prefix + "register_type", str, choices=("holding", "input"))
        dtype = get(prefix + "data_type", str, choices=DATA_FORMATS)
        byte_order = get(prefix + "byte_order", str, choices=("big", "little"))
        word_order = get(prefix + "word_order", str, choices=("big", "little"))
        scale = get(prefix + "scale", (int, float))
        offset = get(prefix + "offset", (int, float))
        unit = get(prefix + "unit", str)
        codes = None
        register_data = configured.get(name, {})
        access = register_data.get("access", "read") if isinstance(register_data, dict) else None
        if access not in {"read", "read_write"}:
            errors.append(prefix + "access: read 또는 read_write여야 합니다")
        if name in mode_labels:
            codes = get(prefix + "codes", dict)
            if codes is not None:
                valid_codes = (
                    len(codes) == len(mode_labels[name])
                    and all(type(code) is int and 0 <= code <= 65535 for code in codes)
                    and all(isinstance(label, str) for label in codes.values())
                    and set(codes.values()) == mode_labels[name]
                )
                if not valid_codes:
                    errors.append(prefix + "codes: 코드와 상태 이름을 확인하세요")
            if dtype != "uint16" or scale != 1 or offset != 0:
                errors.append(prefix + "상태 코드는 uint16, scale=1, offset=0이어야 합니다")
        if scale == 0:
            errors.append(prefix + "scale: 0은 허용하지 않습니다")
        if address is not None and base is not None and dtype is not None:
            request_address = address - base
            if not 0 <= request_address <= 65536 - DATA_FORMATS[dtype][0]:
                errors.append(prefix + "address: 요청 레지스터 범위 초과")
            registers.append(Register(name, address, request_address, reg_type,
                                      dtype, byte_order, word_order, scale, offset, unit, codes, access))
    control = None
    control_data = data.get("control") if isinstance(data, dict) else None
    if control_data is not None:
        if not isinstance(control_data, dict):
            errors.append("control: mapping이어야 합니다")
        else:
            control_enabled = get("control.enabled", bool)
            write_signal = get("control.write_signal", str)
            write_interval = get(
                "control.write_interval_seconds", (int, float), minimum=0.001)
            explore = None
            explore_data = control_data.get("explore")
            if explore_data is not None:
                if not isinstance(explore_data, dict):
                    errors.append("control.explore: mapping이어야 합니다")
                else:
                    pattern = get("control.explore.pattern", str,
                                  choices=("up_down", "hold"))
                    min_voltage = get("control.explore.min_voltage", (int, float))
                    max_voltage = get("control.explore.max_voltage", (int, float))
                    step_voltage = get("control.explore.step_voltage", (int, float),
                                       minimum=0.000001)
                    hold_every_steps = get("control.explore.hold_every_steps", int,
                                           minimum=1)
                    hold_steps = get("control.explore.hold_steps", int, minimum=1)
                    if min_voltage is not None and max_voltage is not None:
                        if min_voltage >= max_voltage:
                            errors.append("control.explore: min_voltage는 max_voltage보다 작아야 합니다")
                        elif safety_limits is not None and not (
                            safety_limits.min_set_voltage <= min_voltage
                            and max_voltage <= safety_limits.max_set_voltage
                        ):
                            errors.append("control.explore: safety 전압 범위 안에 있어야 합니다")
                    if all(value is not None for value in (
                        pattern, min_voltage, max_voltage, step_voltage,
                        hold_every_steps, hold_steps,
                    )):
                        explore = ExploreSettings(
                            pattern=pattern,
                            min_voltage=float(min_voltage),
                            max_voltage=float(max_voltage),
                            step_voltage=float(step_voltage),
                            hold_every_steps=hold_every_steps,
                            hold_steps=hold_steps,
                        )
            if control_enabled and write_enabled is not True:
                errors.append("control.enabled: 실제 Write에는 write_enabled: true가 필요합니다")
            if control_enabled and explore is None:
                errors.append("control.explore: EXPLORE 자동 제어 설정이 필요합니다")
            configured_registers = {register.name: register for register in registers}
            target_register = configured_registers.get(write_signal)
            if control_enabled and (
                target_register is None
                or target_register.access != "read_write"
                or target_register.register_type != "holding"
            ):
                errors.append("control.write_signal: read_write Holding Register여야 합니다")
            if control_enabled and ai_enabled and not ai_shadow_mode and not write_enabled:
                errors.append("ai.shadow_mode: false에는 write_enabled: true가 필요합니다")
            if control_enabled is not None and write_signal is not None and write_interval is not None:
                control = ControlSettings(
                    enabled=control_enabled,
                    write_signal=write_signal,
                    write_interval_seconds=float(write_interval),
                    explore=explore,
                )

    if errors:
        raise ConfigError("설정 확인이 필요합니다:\n- " + "\n- ".join(errors))
    return Config(profile, write_enabled, host, port, unit_id, timeout, retries, poll, reconnect,
                  (path.parent / log_file).resolve(), max_bytes, backups, record_interval, tuple(registers),
                  safety_limits, ai_enabled, ai_shadow_mode, ai_deterministic,
                  model_metadata_file, control)
