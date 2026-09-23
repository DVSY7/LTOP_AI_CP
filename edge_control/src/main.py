"""산업용 PC 제어 루프의 조립 지점.

설정 로드 → Modbus Read → Guard → State 처리 → EXPLORE 또는 AI Action →
Safety Filter → 조건부 Write → JSONL 로그 순서로 한 제어 주기를 실행한다.
"""

import argparse
import sys
import time

from shared.control_core import (
    ModelCompatibilityError,
    SafetyLimits,
    load_model_metadata,
    verify_model_artifact,
)

from .config_loader import ConfigError, load_config
from .communication.modbus_client import ReadOnlyClient
from .communication.register_reader import RegisterReader, timestamp
from .communication.control_writer import write_control_target
from .communication.test_writer import TestWriteError
from .controller.control_guard import evaluate_control_guard
from .controller.ai_controller import build_ai_action, load_ai_controller
from .controller.explore_controller import (
    ExploreController,
    PatternExploreController,
    build_explore_action,
)
from .logging_setup import emit, setup_logging
from .state.state_processor import StateProcessor


MEASUREMENT_FIELDS = (
    "rectifier_voltage",
    "rectifier_current",
    "set_voltage",
    "tb_potential",
)
ACTION_LOG_FIELDS = (
    "source",
    "status",
    "model_version",
    "normalized_action",
    "original_delta_voltage",
    "filtered_delta_voltage",
    "target_set_voltage",
    "actual_write_voltage",
    "safety_intervened",
    "safety_reasons",
    "remaining_seconds",
    "reasons",
    "write_error",
)


def build_learning_measurements(values, processed_state):
    """학습 transition에 필요한 공학 단위 측정값과 TB trend만 남긴다."""

    measurements = {
        name: round(float(values[name]), 2)
        for name in MEASUREMENT_FIELDS
        if values is not None and name in values
    } if values is not None else None
    if measurements is not None:
        processed_values = (processed_state or {}).get("values") or {}
        if "tb_trend" in processed_values:
            measurements["tb_trend"] = round(float(processed_values["tb_trend"]), 2)
    return measurements


def build_learning_action(control_action):
    """모델 입력 복제본을 제외하고 실제 제어 이력에 필요한 Action만 남긴다."""

    if control_action is None:
        return None
    return {
        name: control_action[name]
        for name in ACTION_LOG_FIELDS
        if name in control_action
    }


def operating_voltage_limits(config):
    """AI와 EXPLORE가 공유하는 시험 운전 전압 범위를 만든다."""

    if config.control is not None and config.control.explore is not None:
        explore = config.control.explore
        max_delta = (
            config.safety_limits.max_delta_voltage
            if config.safety_limits is not None else None
        )
        return SafetyLimits(explore.min_voltage, explore.max_voltage, max_delta)
    return config.safety_limits


def run(config, client, logger, *, ai_controller=None, max_cycles=None,
        sleep=time.sleep, clock=time.monotonic):
    """설정된 주기로 제어 주기를 반복한다.

    `client`와 `logger`를 인자로 받아 실제 장비와 모의 통신 테스트가 같은 흐름을
    사용한다. Read가 실패한 주기는 Guard가 차단하고 연결을 닫은 뒤 재연결 간격만큼
    기다린다. 정상 주기 로그는 `record_interval`마다 남기되, 상태 변경·오류·Write는
    즉시 기록한다.
    """
    reader = RegisterReader(client, config.registers)
    state_processor = StateProcessor()
    explore_controller = None
    if config.control is not None and config.control.explore is not None:
        explore = config.control.explore
        explore_controller = PatternExploreController(
            SafetyLimits(explore.min_voltage, explore.max_voltage, explore.step_voltage),
            pattern=explore.pattern,
            step_voltage=explore.step_voltage,
            hold_every_steps=explore.hold_every_steps,
            hold_steps=explore.hold_steps,
        )
    elif config.safety_limits is not None:
        explore_controller = ExploreController(config.safety_limits)
    cycle = 0
    last_write_attempt_at = None
    last_record_at = None
    last_record_signature = None
    try:
        while max_cycles is None or cycle < max_cycles:
            started = clock()
            cycle += 1
            samples = []
            for register in config.registers:
                sample = reader.read(register.name)
                samples.append(sample)
                if sample["status"] != "ok":
                    break
            valid = len(samples) == len(config.registers) and all(
                sample["status"] == "ok" for sample in samples)
            failed_sample = next(
                (sample for sample in samples if sample["status"] != "ok"), None)
            read_error = (
                {"signal": failed_sample["name"],
                 "message": failed_sample.get("error", "read_failed")}
                if failed_sample is not None else None
            )
            states = (
                {sample["name"]: sample["label"] for sample in samples if "label" in sample}
                if valid
                else None
            )
            guard = evaluate_control_guard(
                cycle_valid=valid,
                states=states,
                write_enabled=config.write_enabled,
            )
            values = {s["name"]: s["value"] for s in samples} if valid else None
            operation_mode = states.get("operation_mode") if states else None
            automatic_mode = (
                operation_mode == "EXPLORE"
                or (operation_mode == "AI" and config.ai_enabled and not config.ai_shadow_mode)
            )
            interval = (
                config.control.write_interval_seconds
                if config.control is not None and config.control.enabled
                else None
            )
            due = interval is not None and (
                last_write_attempt_at is None or started - last_write_attempt_at >= interval
            )
            waiting_for_interval = automatic_mode and interval is not None and not due
            control_action = (
                {"source": operation_mode, "status": "waiting_for_interval",
                 "remaining_seconds": round(interval - (started - last_write_attempt_at), 3),
                 "actual_write_voltage": None}
                if waiting_for_interval
                else build_explore_action(
                    states=states,
                    values=values,
                    guard=guard,
                    controller=explore_controller,
                )
            )
            processed_state = state_processor.process(
                values=values,
                state_usable=guard.state_usable,
            )
            ai_action = None if waiting_for_interval and operation_mode == "AI" else build_ai_action(
                states=states,
                values=values,
                guard=guard,
                processed_state=processed_state,
                controller=ai_controller,
            )
            if ai_action is not None:
                control_action = ai_action
            write_result = None
            should_write = (
                automatic_mode
                and due
                and guard.write_allowed
                and control_action is not None
                and control_action.get("status") == "generated"
                and config.control is not None
                and config.control.enabled
            )
            if should_write:
                last_write_attempt_at = started
                try:
                    write_details = write_control_target(
                        config, control_action["target_set_voltage"])
                    control_action["actual_write_voltage"] = write_details["actual_write_voltage"]
                    control_action["status"] = "written"
                    # 상세 Write 주소·원시값은 설정과 통신 계층의 관심사다.
                    # 운영 로그에는 제어 성공 여부만 남기고 실제 전압은 control_action에 기록한다.
                    write_result = {"read_back_verified": write_details["read_back_verified"]}
                except TestWriteError as exc:
                    control_action["status"] = "write_failed"
                    control_action["actual_write_voltage"] = None
                    control_action["write_error"] = str(exc)
            record = dict(event="cycle", timestamp=timestamp(), cycle=cycle, valid=valid,
                          measurements=build_learning_measurements(values, processed_state),
                          read_error=read_error,
                          states=states,
                          control_action=build_learning_action(control_action),
                          write_result=write_result)
            action_status = (
                control_action.get("status") if control_action is not None else None
            )
            record_signature = (
                valid,
                tuple(sorted((states or {}).items())),
                tuple(guard.reasons),
                action_status,
            )
            event_requires_record = (
                not valid
                or write_result is not None
                or action_status == "write_failed"
                or record_signature != last_record_signature
            )
            if (
                last_record_at is None
                or started - last_record_at >= config.record_interval
                or event_requires_record
            ):
                emit(logger, record)
                last_record_at = started
                last_record_signature = record_signature
            if not valid:
                client.close()
            if max_cycles is not None and cycle >= max_cycles:
                break
            # 정상 주기는 처리 시간을 제외하고 대기. 실패 후에는 재연결 간격 보장.
            delay = max(0, config.poll_interval - (clock() - started))
            sleep(delay if valid else max(delay, config.reconnect_interval))
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="산업용 PC Read-only 수집")
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-config", action="store_true", help="설정만 검증, 통신 없음")
    parser.add_argument("--max-cycles", type=int, help="지정 횟수 수집 후 종료")
    args = parser.parse_args(argv)
    if args.max_cycles is not None and args.max_cycles <= 0:
        parser.error("--max-cycles는 양수여야 합니다")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # if config.write_enabled:
    #     print("Read-only 수집기는 write_enabled: false만 허용합니다", file=sys.stderr)
    #     return 2
    metadata = None
    if config.ai_enabled:
        if config.model_metadata_file is None or config.safety_limits is None:
            print("AI에는 모델 메타데이터와 Safety 설정이 필요합니다", file=sys.stderr)
            return 2
        try:
            metadata = load_model_metadata(config.model_metadata_file)
            verify_model_artifact(metadata)
        except ModelCompatibilityError as exc:
            print(f"AI 모델 검증 실패: {exc}", file=sys.stderr)
            return 2
    if args.check_config:
        ai_mode = (
            f"AI {'Shadow' if config.ai_shadow_mode else 'Write'}={metadata.version}"
            if metadata else "AI disabled"
        )
        write_mode = "automatic Write enabled" if (
            config.control is not None and config.control.enabled
        ) else "Write disabled"
        print(f"설정 검증 통과: {config.profile} ({ai_mode}, {write_mode})")
        return 0
    logger = setup_logging(config)
    try:
        ai_controller = (
            load_ai_controller(metadata, operating_voltage_limits(config),
                               deterministic=config.ai_deterministic)
            if metadata else None
        )
    except ModelCompatibilityError as exc:
        print(f"AI 모델 로드 실패: {exc}", file=sys.stderr)
        return 2
    client = ReadOnlyClient(config)
    emit(logger, dict(event="start", timestamp=timestamp(), profile=config.profile,
                      write_enabled=config.write_enabled))
    try:
        run(config, client, logger, ai_controller=ai_controller,
            max_cycles=args.max_cycles)
    except KeyboardInterrupt:
        pass
    finally:
        emit(logger, dict(event="stop", timestamp=timestamp()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
