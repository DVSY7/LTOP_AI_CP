"""EXPLORE 모드에서 연속 전압 Action 후보를 만들고 안전 한계를 적용한다."""

from dataclasses import asdict, dataclass
import random

from shared.control_core import SafetyLimits
from .voltage_envelope import filter_with_envelope_recovery


@dataclass(frozen=True)
class ExploreAction:
    source: str
    original_delta_voltage: float
    filtered_delta_voltage: float
    target_set_voltage: float
    safety_intervened: bool
    safety_reasons: tuple[str, ...]
    actual_write_voltage: float | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        for name in (
            "original_delta_voltage",
            "filtered_delta_voltage",
            "target_set_voltage",
        ):
            result[name] = round(result[name], 2)
        result["safety_reasons"] = list(self.safety_reasons)
        return result


class ExploreController:
    """허용 변화량 안에서 연속 균등분포로 탐색 Action을 생성한다."""

    def __init__(self, limits: SafetyLimits, random_source=None):
        if limits.max_delta_voltage is None:
            raise ValueError("EXPLORE에는 max_delta_voltage 설정이 필요합니다")
        self.limits = limits
        self.random_source = random_source or random.SystemRandom()

    def propose(self, current_set_voltage: float) -> ExploreAction:
        max_delta = self.limits.max_delta_voltage
        requested_delta = float(self.random_source.uniform(-max_delta, max_delta))
        filtered = filter_with_envelope_recovery(
            current_set_voltage=current_set_voltage,
            requested_delta_voltage=requested_delta,
            limits=self.limits,
        )
        return ExploreAction(
            source="EXPLORE",
            original_delta_voltage=filtered.requested_delta_voltage,
            filtered_delta_voltage=filtered.filtered_delta_voltage,
            target_set_voltage=filtered.filtered_set_voltage,
            safety_intervened=filtered.intervened,
            safety_reasons=filtered.reasons,
            # Read-only 단계에서는 실제 Write 값이 존재하지 않는다.
            actual_write_voltage=None,
        )


class PatternExploreController:
    """고정 전압 범위에서 반복 가능한 EXPLORE 경로를 만든다."""

    def __init__(self, limits: SafetyLimits, *, pattern: str,
                 step_voltage: float, hold_every_steps: int, hold_steps: int):
        if pattern not in {"up_down", "hold"}:
            raise ValueError("지원하지 않는 EXPLORE pattern입니다")
        if limits.max_delta_voltage is None or not 0 < step_voltage <= limits.max_delta_voltage:
            raise ValueError("step_voltage는 Safety 최대 변화량 안에 있어야 합니다")
        self.limits = limits
        self.pattern = pattern
        self.step_voltage = step_voltage
        self.hold_every_steps = hold_every_steps
        self.hold_steps = hold_steps
        self._direction = 1.0
        self._moves_since_hold = 0
        self._holds_remaining = 0

    def propose(self, current_set_voltage: float) -> ExploreAction:
        if self._holds_remaining > 0:
            requested_delta = 0.0
            self._holds_remaining -= 1
        else:
            if current_set_voltage >= self.limits.max_set_voltage:
                self._direction = -1.0
            elif current_set_voltage <= self.limits.min_set_voltage:
                self._direction = 1.0
            requested_delta = self._direction * self.step_voltage
            requested_target = current_set_voltage + requested_delta
            if requested_target >= self.limits.max_set_voltage:
                requested_delta = self.limits.max_set_voltage - current_set_voltage
                self._direction = -1.0
            elif requested_target <= self.limits.min_set_voltage:
                requested_delta = self.limits.min_set_voltage - current_set_voltage
                self._direction = 1.0

            self._moves_since_hold += 1
            if self.pattern == "hold" and self._moves_since_hold >= self.hold_every_steps:
                self._moves_since_hold = 0
                self._holds_remaining = self.hold_steps

        filtered = filter_with_envelope_recovery(
            current_set_voltage=current_set_voltage,
            requested_delta_voltage=requested_delta,
            limits=self.limits,
        )
        return ExploreAction(
            source="EXPLORE",
            original_delta_voltage=filtered.requested_delta_voltage,
            filtered_delta_voltage=filtered.filtered_delta_voltage,
            target_set_voltage=filtered.filtered_set_voltage,
            safety_intervened=filtered.intervened,
            safety_reasons=filtered.reasons,
            actual_write_voltage=None,
        )


def build_explore_action(*, states, values, guard, controller):
    """현재 주기가 EXPLORE이면 Action 기록을 만들고, 차단 상태도 명시한다."""

    if not states or states.get("operation_mode") != "EXPLORE":
        return None
    if not guard.action_allowed:
        return {"source": "EXPLORE", "status": "blocked",
                "reasons": list(guard.reasons), "actual_write_voltage": None}
    if controller is None:
        return {"source": "EXPLORE", "status": "error",
                "reasons": ["safety_limits_missing"], "actual_write_voltage": None}
    if not values or "set_voltage" not in values:
        return {"source": "EXPLORE", "status": "error",
                "reasons": ["set_voltage_missing"], "actual_write_voltage": None}
    try:
        action = controller.propose(values["set_voltage"])
    except (TypeError, ValueError) as exc:
        return {"source": "EXPLORE", "status": "error",
                "reasons": ["invalid_set_voltage"], "error": str(exc),
                "actual_write_voltage": None}
    return {"status": "generated", **action.to_dict()}
