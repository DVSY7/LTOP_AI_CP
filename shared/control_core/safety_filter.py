"""설정전압의 절대 범위와 주기당 변화량을 제한하는 순수 함수."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SafetyLimits:
    min_set_voltage: float
    max_set_voltage: float
    max_delta_voltage: float | None = None

    def __post_init__(self):
        values = (self.min_set_voltage, self.max_set_voltage)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("설정전압 한계는 유한한 숫자여야 합니다")
        if self.min_set_voltage >= self.max_set_voltage:
            raise ValueError("최소 설정전압은 최대 설정전압보다 작아야 합니다")
        if self.max_delta_voltage is not None:
            if not math.isfinite(self.max_delta_voltage) or self.max_delta_voltage <= 0:
                raise ValueError("최대 변화량은 0보다 큰 유한한 숫자여야 합니다")


@dataclass(frozen=True)
class SafetyFilterResult:
    current_set_voltage: float
    requested_delta_voltage: float
    requested_set_voltage: float
    filtered_delta_voltage: float
    filtered_set_voltage: float
    intervened: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}은 유한한 숫자여야 합니다")
    return value


def filter_voltage_request(
    *,
    current_set_voltage: float,
    requested_delta_voltage: float,
    limits: SafetyLimits,
) -> SafetyFilterResult:
    """요청 변화량을 제한하고 요청값·적용값·제한 사유를 모두 반환한다."""

    current = _require_finite("current_set_voltage", current_set_voltage)
    requested_delta = _require_finite("requested_delta_voltage", requested_delta_voltage)
    if not limits.min_set_voltage <= current <= limits.max_set_voltage:
        raise ValueError("현재 설정전압이 Safety Filter 절대 범위를 벗어났습니다")

    requested_set = current + requested_delta
    filtered_delta = requested_delta
    reasons: list[str] = []

    if limits.max_delta_voltage is not None:
        if filtered_delta > limits.max_delta_voltage:
            filtered_delta = limits.max_delta_voltage
            reasons.append("delta_above_max")
        elif filtered_delta < -limits.max_delta_voltage:
            filtered_delta = -limits.max_delta_voltage
            reasons.append("delta_below_min")

    filtered_set = current + filtered_delta
    absolute_limit_applied = False
    if filtered_set > limits.max_set_voltage:
        filtered_set = limits.max_set_voltage
        reasons.append("voltage_above_max")
        absolute_limit_applied = True
    elif filtered_set < limits.min_set_voltage:
        filtered_set = limits.min_set_voltage
        reasons.append("voltage_below_min")
        absolute_limit_applied = True

    # 절대전압 제한이 없으면 뺄셈 재계산으로 생기는 부동소수점 오차를 피한다.
    applied_delta = filtered_set - current if absolute_limit_applied else filtered_delta
    return SafetyFilterResult(
        current_set_voltage=current,
        requested_delta_voltage=requested_delta,
        requested_set_voltage=requested_set,
        filtered_delta_voltage=applied_delta,
        filtered_set_voltage=filtered_set,
        intervened=bool(reasons),
        reasons=tuple(reasons),
    )
