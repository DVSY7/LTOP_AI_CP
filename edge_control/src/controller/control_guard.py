"""수집된 상태가 제어 판단에 사용 가능한지 보수적으로 판정한다."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GuardResult:
    state_usable: bool
    action_allowed: bool
    write_allowed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def evaluate_control_guard(
    *,
    cycle_valid: bool,
    states: dict[str, str] | None,
    write_enabled: bool,
) -> GuardResult:
    """통신·운전모드·TB 상태를 확인하고 차단 사유를 반환한다.

    Action 후보를 만들 수 있는 모드는 AI와 EXPLORE이다.
    write_enabled가 false이면 Action 허용 여부와 관계없이 Write를 차단한다.
    """

    reasons: list[str] = []

    if not cycle_valid:
        reasons.append("invalid_cycle")
        return GuardResult(False, False, False, tuple(reasons))

    states = states or {}
    required_states = ("remote_mode", "operation_status", "operation_mode", "tb_status")
    missing = [name for name in required_states if name not in states]
    if missing:
        reasons.extend(f"missing_{name}" for name in missing)
        return GuardResult(False, False, False, tuple(reasons))

    if states["tb_status"] != "NORMAL":
        reasons.append(f"tb_status_{states['tb_status'].lower()}")
    if states["remote_mode"] != "REMOTE":
        reasons.append("not_remote")
    if states["operation_status"] != "ON":
        reasons.append("operation_off")

    operation_mode = states["operation_mode"]
    if operation_mode == "MANUAL":
        reasons.append("manual_mode")
    elif operation_mode not in {"AI", "EXPLORE"}:
        reasons.append("unsupported_operation_mode")

    state_usable = not any(
        reason.startswith("tb_status_") or reason.startswith("missing_")
        for reason in reasons
    )
    action_allowed = not reasons

    if not write_enabled:
        reasons.append("write_disabled")
    write_allowed = action_allowed and write_enabled

    return GuardResult(state_usable, action_allowed, write_allowed, tuple(reasons))
