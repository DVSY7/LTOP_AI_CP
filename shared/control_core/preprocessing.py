"""학습과 엣지가 공유하는 4차원 정책 State 정규화."""

from dataclasses import dataclass

from .state import PolicyState


@dataclass(frozen=True)
class StateBounds:
    voltage: tuple[float, float]
    current: tuple[float, float]
    potential: tuple[float, float]
    tb_trend: tuple[float, float]


def _normalize(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if lower >= upper:
        raise ValueError("State 정규화 최솟값은 최댓값보다 작아야 합니다")
    return min(1.0, max(0.0, (value - lower) / (upper - lower)))


def normalize_policy_state(state: PolicyState, bounds: StateBounds) -> tuple[float, ...]:
    return (
        _normalize(state.rectifier_voltage, bounds.voltage),
        _normalize(state.rectifier_current, bounds.current),
        _normalize(state.tb_potential, bounds.potential),
        _normalize(state.tb_trend, bounds.tb_trend),
    )
