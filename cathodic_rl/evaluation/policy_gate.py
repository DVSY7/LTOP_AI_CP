"""SAC 정책의 방향성, 목표 유지 및 Action 포화를 검사한다."""

import argparse
from dataclasses import asdict, dataclass
import json

import numpy as np
from stable_baselines3 import SAC

from cathodic_rl.config.settings import (
    MAX_OUTPUT_CURRENT,
    MAX_OUTPUT_VOLTAGE,
    MAX_PIPE_POTENTIAL,
    MIN_OUTPUT_CURRENT,
    MIN_OUTPUT_VOLTAGE,
    MIN_PIPE_POTENTIAL,
    SAC_MODEL_PATH,
)


@dataclass(frozen=True)
class PolicyGateResult:
    passed: bool
    above_action: float
    target_action: float
    below_action: float
    direction_passed: bool
    target_hold_passed: bool
    saturation_rate: float
    saturation_passed: bool

    def to_dict(self):
        return asdict(self)


def _normalize(value, lower, upper):
    return (value - lower) / (upper - lower)


def _observation(voltage, current, potential, trend):
    return np.asarray([
        _normalize(voltage, MIN_OUTPUT_VOLTAGE, MAX_OUTPUT_VOLTAGE),
        _normalize(current, MIN_OUTPUT_CURRENT, MAX_OUTPUT_CURRENT),
        _normalize(potential, MIN_PIPE_POTENTIAL, MAX_PIPE_POTENTIAL),
        _normalize(trend, -10.0, 10.0),
    ], dtype=np.float32)


def _predict(model, potential, *, voltage=43.63, current=5.95, trend=0.0):
    action, _ = model.predict(
        _observation(voltage, current, potential, trend), deterministic=True)
    return float(np.asarray(action).item())


def evaluate_policy(model) -> PolicyGateResult:
    below_action = _predict(model, -1650.0)
    target_action = _predict(model, -1600.0)
    above_action = _predict(model, -1570.0)
    direction_passed = below_action < -0.1 and above_action > 0.1
    target_hold_passed = abs(target_action) <= 0.25

    actions = [
        _predict(model, potential, trend=trend)
        for potential in (-1670, -1650, -1610, -1600, -1590, -1570, -1563)
        for trend in (-5.0, 0.0, 5.0)
    ]
    saturation_rate = sum(abs(action) >= 0.95 for action in actions) / len(actions)
    saturation_passed = saturation_rate <= 0.5
    return PolicyGateResult(
        passed=direction_passed and target_hold_passed and saturation_passed,
        above_action=above_action,
        target_action=target_action,
        below_action=below_action,
        direction_passed=direction_passed,
        target_hold_passed=target_hold_passed,
        saturation_rate=saturation_rate,
        saturation_passed=saturation_passed,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAC 정책 배포 전 방향·포화 검사")
    parser.add_argument("--model", default=SAC_MODEL_PATH)
    args = parser.parse_args(argv)
    result = evaluate_policy(SAC.load(args.model))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
