"""연속된 정상 측정 주기로 현장 관측 가능한 4차원 State를 만든다."""

from shared.control_core import build_base_state, build_policy_state


class StateProcessor:
    def __init__(self):
        self._previous_tb = None

    def process(self, *, values, state_usable: bool):
        if not state_usable:
            self._previous_tb = None
            return None
        try:
            base = build_base_state(values)
        except ValueError as exc:
            self._previous_tb = None
            return {"status": "invalid", "error": str(exc)}

        current_tb = base.tb_potential
        if self._previous_tb is None:
            self._previous_tb = current_tb
            return {
                "status": "warming_up",
                **base.to_dict(),
                "normalized_vector": None,
                "reason": "previous_tb_unavailable",
            }

        trend = current_tb - self._previous_tb
        self._previous_tb = current_tb
        state = build_policy_state(values, trend)
        return {"status": "valid", **state.to_dict(), "normalized_vector": None}
