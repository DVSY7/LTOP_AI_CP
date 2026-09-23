"""명시적 이중 활성화가 필요한 시험용 단일 Write 명령."""

import argparse
import json
import sys

from .config_loader import ConfigError, load_config
from .communication.test_writer import TestWriteError, perform_single_write


CONFIRMATION_TEXT = "I_UNDERSTAND_SINGLE_TEST_WRITE"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulator 대상 단일 Modbus Write 시험")
    parser.add_argument("--config", required=True)
    parser.add_argument("--signal", required=True)
    parser.add_argument("--value", required=True, type=float)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args(argv)

    if args.confirm != CONFIRMATION_TEXT:
        print(f"확인 문자열이 일치하지 않습니다: {CONFIRMATION_TEXT}", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        result = perform_single_write(config, args.signal, args.value)
    except (ConfigError, TestWriteError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
