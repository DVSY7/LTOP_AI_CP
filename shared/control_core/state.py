"""학습과 엣지가 공유하는 기본 측정 State의 필드 순서와 단위."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class StateField:
    name: str
    unit: str


# 모델 메타데이터와 비교할 기준 순서다. 임의로 정렬하거나 바꾸지 않는다.
BASE_STATE_FIELDS = (
    StateField("rectifier_voltage", "V"),
    StateField("rectifier_current", "A"),
    StateField("tb_potential", "mV"),
)

POLICY_STATE_FIELDS = BASE_STATE_FIELDS + (StateField("tb_trend", "mV/cycle"),)


@dataclass(frozen=True)
class BaseState:
    rectifier_voltage: float
    rectifier_current: float
    tb_potential: float

    @property
    def vector(self) -> tuple[float, float, float]:
        return tuple(getattr(self, field.name) for field in BASE_STATE_FIELDS)

    def to_dict(self) -> dict:
        return {
            "field_order": [field.name for field in BASE_STATE_FIELDS],
            "units": {field.name: field.unit for field in BASE_STATE_FIELDS},
            "values": asdict(self),
            "vector": list(self.vector),
        }


def build_base_state(values: dict[str, object]) -> BaseState:
    """현재 주기의 측정값으로 순서가 고정된 3-field State를 만든다."""

    if not isinstance(values, dict):
        raise ValueError("State values는 매핑이어야 합니다")
    converted = {}
    for field in BASE_STATE_FIELDS:
        if field.name not in values:
            raise ValueError(f"State 필드 누락: {field.name}")
        value = values[field.name]
        if isinstance(value, bool):
            raise ValueError(f"State 필드는 숫자여야 합니다: {field.name}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"State 필드는 숫자여야 합니다: {field.name}") from exc
        if not math.isfinite(number):
            raise ValueError(f"State 필드는 유한한 숫자여야 합니다: {field.name}")
        converted[field.name] = number
    return BaseState(**converted)


@dataclass(frozen=True)
class PolicyState:
    rectifier_voltage: float
    rectifier_current: float
    tb_potential: float
    tb_trend: float

    @property
    def vector(self) -> tuple[float, float, float, float]:
        return tuple(getattr(self, field.name) for field in POLICY_STATE_FIELDS)

    def to_dict(self) -> dict:
        return {
            "field_order": [field.name for field in POLICY_STATE_FIELDS],
            "units": {field.name: field.unit for field in POLICY_STATE_FIELDS},
            "values": asdict(self),
            "vector": list(self.vector),
        }


def build_policy_state(values: dict[str, object], tb_trend: float) -> PolicyState:
    base = build_base_state(values)
    trend = float(tb_trend)
    if not math.isfinite(trend):
        raise ValueError("tb_trend는 유한한 숫자여야 합니다")
    return PolicyState(*base.vector, trend)
