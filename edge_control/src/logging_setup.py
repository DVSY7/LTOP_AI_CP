"""상세 JSONL 파일 로그와 사람이 읽는 콘솔 요약을 함께 기록한다."""

from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo


class ConsoleRecordFormatter(logging.Formatter):
    """JSONL 레코드에서 운전자가 빠르게 볼 핵심 정보만 한 줄로 만든다."""

    @staticmethod
    def _time(record):
        value = record.get("timestamp")
        if not value:
            return "--:--:--"
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(ZoneInfo("Asia/Seoul"))
            return parsed.strftime("%H:%M:%S")
        except (TypeError, ValueError):
            return "--:--:--"

    @staticmethod
    def _number(value, digits=2):
        return "-" if value is None else f"{float(value):.{digits}f}"

    def _format_cycle(self, record):
        values = record.get("measurements") or {}
        states = record.get("states") or {}
        action = record.get("control_action") or {}
        timestamp = self._time(record)
        mode = states.get("operation_mode", "UNKNOWN")
        voltage = self._number(values.get("rectifier_voltage"))
        current = self._number(values.get("rectifier_current"))
        potential = self._number(values.get("tb_potential"), 0)

        if not record.get("valid"):
            failed = (record.get("read_error") or {}).get("signal", "unknown")
            return f"[{timestamp}] #{record.get('cycle')} 통신/수집 오류: {failed}"

        summary = (
            f"[{timestamp}] #{record.get('cycle')} {mode} | "
            f"V={voltage} V I={current} A TB={potential} mV"
        )
        status = action.get("status")
        if status == "written":
            return f"{summary} | WRITE={self._number(action.get('actual_write_voltage'))} V"
        if status == "write_failed":
            return f"{summary} | WRITE 실패: {action.get('write_error', 'unknown')}"
        if status == "generated":
            target = self._number(action.get("target_set_voltage"))
            reasons = action.get("safety_reasons") or []
            limited = f" 제한={','.join(reasons)}" if reasons else ""
            return f"{summary} | 제안={target} V{limited}"
        if status == "waiting_for_interval":
            remaining = self._number(action.get("remaining_seconds"), 0)
            return f"{summary} | 다음 제어까지 {remaining}초"
        if status in {"blocked", "warming_up", "error"}:
            reasons = ",".join(action.get("reasons") or [])
            return f"{summary} | 제어 {status}: {reasons}"
        return summary

    def format(self, log_record):
        try:
            record = json.loads(log_record.getMessage())
        except (TypeError, json.JSONDecodeError):
            return log_record.getMessage()
        event = record.get("event")
        if event == "start":
            return (
                f"[{self._time(record)}] 시작 | profile={record.get('profile')} "
                f"| write={'ON' if record.get('write_enabled') else 'OFF'}"
            )
        if event == "stop":
            return f"[{self._time(record)}] 종료"
        if event == "cycle":
            return self._format_cycle(record)
        return f"[{event or 'log'}]"


def setup_logging(config):
    logger = logging.getLogger("edge_control.collector")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleRecordFormatter())
    file_handler = RotatingFileHandler(
        config.log_file, maxBytes=config.max_bytes,
        backupCount=config.backup_count, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def emit(logger, record):
    logger.info(json.dumps(record, ensure_ascii=False, allow_nan=False))
