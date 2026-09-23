#!/usr/bin/env bash
# systemd와 수동 실행이 공통으로 사용하는 edge 실행 래퍼다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${EDGE_CONTROL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${EDGE_CONTROL_PYTHON:-$ROOT_DIR/.venv/bin/python}"
CONFIG_FILE="${EDGE_CONTROL_CONFIG:-$ROOT_DIR/edge_control/configs/junction_test.yaml}"

exec "$PYTHON_BIN" -m edge_control.src.main --config "$CONFIG_FILE"
