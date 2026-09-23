"""SAC 정책을 Shadow Mode로 실행하고 Safety Filter 결과를 기록한다."""

from dataclasses import asdict, dataclass

import numpy as np

from shared.control_core import (
    ModelCompatibilityError,
    PolicyState,
    SafetyLimits,
    StateBounds,
    normalize_policy_state,
    verify_model_artifact,
)
from .voltage_envelope import filter_with_envelope_recovery


@dataclass(frozen=True)
class AiAction:
    source: str
    model_version: str
    normalized_action: float
    original_delta_voltage: float
    filtered_delta_voltage: float
    target_set_voltage: float
    safety_intervened: bool
    safety_reasons: tuple[str, ...]
    normalized_state: tuple[float, ...]
    actual_write_voltage: float | None = None

    def to_dict(self):
        result = asdict(self)
        for name in ("normalized_action", "original_delta_voltage",
                     "filtered_delta_voltage", "target_set_voltage"):
            result[name] = round(result[name], 2)
        result["normalized_state"] = [round(value, 6) for value in self.normalized_state]
        result["safety_reasons"] = list(self.safety_reasons)
        return result


class AiController:
    """검증된 SAC 모델을 호출해 안전 필터가 적용된 목표 전압을 만든다."""
    def __init__(self, model, metadata, safety_limits: SafetyLimits, *, deterministic=True):
        self.model = model
        self.metadata = metadata
        self.safety_limits = safety_limits
        self.deterministic = deterministic
        self.bounds = StateBounds(*metadata.state_bounds)

    def propose(self, processed_state, current_set_voltage: float) -> AiAction:
        """4차원 현장 State를 정규화하고 SAC Action을 전압 변화량으로 변환한다."""
        values = processed_state["values"]
        state = PolicyState(**values)
        normalized = normalize_policy_state(state, self.bounds)
        action, _ = self.model.predict(
            np.asarray(normalized, dtype=np.float32),
            deterministic=self.deterministic,
        )
        normalized_action = float(np.asarray(action).item())
        if not -1.0 <= normalized_action <= 1.0:
            raise ValueError("SAC Action이 -1~1 범위를 벗어났습니다")
        requested_delta = normalized_action * self.metadata.max_delta_voltage
        filtered = filter_with_envelope_recovery(
            current_set_voltage=current_set_voltage,
            requested_delta_voltage=requested_delta,
            limits=self.safety_limits,
        )
        return AiAction(
            source="AI",
            model_version=self.metadata.version,
            normalized_action=normalized_action,
            original_delta_voltage=requested_delta,
            filtered_delta_voltage=filtered.filtered_delta_voltage,
            target_set_voltage=filtered.filtered_set_voltage,
            safety_intervened=filtered.intervened,
            safety_reasons=filtered.reasons,
            normalized_state=normalized,
            actual_write_voltage=None,
        )


def load_ai_controller(metadata, safety_limits, *, deterministic=True):
    verify_model_artifact(metadata)
    try:
        from stable_baselines3 import SAC
        model = SAC.load(str(metadata.model_path))
    except Exception as exc:
        raise ModelCompatibilityError(f"SAC 모델 로드 실패: {exc}") from exc
    if model.observation_space.shape != (4,):
        raise ModelCompatibilityError("SAC 모델 입력 차원이 4가 아닙니다")
    if model.action_space.shape != (1,):
        raise ModelCompatibilityError("SAC 모델 Action 차원이 1이 아닙니다")
    return AiController(model, metadata, safety_limits, deterministic=deterministic)


def build_ai_action(*, states, values, guard, processed_state, controller):
    if not states or states.get("operation_mode") != "AI":
        return None
    if not guard.action_allowed:
        return {"source": "AI", "status": "blocked", "reasons": list(guard.reasons),
                "actual_write_voltage": None}
    if not processed_state or processed_state.get("status") != "valid":
        status = processed_state.get("status") if processed_state else "unavailable"
        return {"source": "AI", "status": "warming_up", "reasons": [status],
                "actual_write_voltage": None}
    if controller is None:
        return {"source": "AI", "status": "error", "reasons": ["model_unavailable"],
                "actual_write_voltage": None}
    if not values or "set_voltage" not in values:
        return {"source": "AI", "status": "error", "reasons": ["set_voltage_missing"],
                "actual_write_voltage": None}
    try:
        action = controller.propose(processed_state, values["set_voltage"])
    except (TypeError, ValueError) as exc:
        return {"source": "AI", "status": "error", "reasons": ["inference_failed"],
                "error": str(exc), "actual_write_voltage": None}
    return {"status": "generated", **action.to_dict()}
