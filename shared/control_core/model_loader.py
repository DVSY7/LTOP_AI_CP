"""모델 파일을 열기 전에 메타데이터와 런타임 규약을 검증한다."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import yaml

from .state import POLICY_STATE_FIELDS


class ModelCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class ModelMetadata:
    version: str
    algorithm: str
    model_path: Path
    sha256: str
    state_fields: tuple[str, ...]
    state_units: tuple[str, ...]
    state_bounds: tuple[tuple[float, float], ...]
    action_fields: tuple[str, ...]
    action_range: tuple[float, float]
    max_delta_voltage: float


def _finite_pair(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ModelCompatibilityError(f"{name}은 [min, max] 형식이어야 합니다")
    pair = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in pair) or pair[0] >= pair[1]:
        raise ModelCompatibilityError(f"{name} 범위가 잘못되었습니다")
    return pair


def load_model_metadata(path) -> ModelMetadata:
    metadata_path = Path(path).resolve()
    try:
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        state_items = data["state"]["fields"]
        action = data["action"]
        metadata = ModelMetadata(
            version=str(data["model_version"]),
            algorithm=str(data["algorithm"]),
            model_path=(metadata_path.parent / data["model_file"]).resolve(),
            sha256=str(data["sha256"]).lower(),
            state_fields=tuple(item["name"] for item in state_items),
            state_units=tuple(item["unit"] for item in state_items),
            state_bounds=tuple(
                (float(item["min"]), float(item["max"])) for item in state_items),
            action_fields=tuple(action["fields"]),
            action_range=_finite_pair(action["normalized_range"], "action.normalized_range"),
            max_delta_voltage=float(action["max_delta_voltage"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, ModelCompatibilityError):
            raise
        raise ModelCompatibilityError(f"모델 메타데이터가 잘못되었습니다: {exc}") from exc

    expected_fields = tuple(field.name for field in POLICY_STATE_FIELDS)
    expected_units = tuple(field.unit for field in POLICY_STATE_FIELDS)
    if metadata.algorithm != "SAC":
        raise ModelCompatibilityError("현재 엣지 런타임은 SAC 메타데이터만 지원합니다")
    if metadata.state_fields != expected_fields:
        raise ModelCompatibilityError(
            f"State 필드 순서 불일치: {metadata.state_fields} != {expected_fields}")
    if metadata.state_units != expected_units:
        raise ModelCompatibilityError("State 단위가 런타임 규약과 일치하지 않습니다")
    if any(not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1]
           for pair in metadata.state_bounds):
        raise ModelCompatibilityError("State 정규화 범위가 잘못되었습니다")
    if metadata.action_fields != ("delta_voltage",):
        raise ModelCompatibilityError("Action 필드는 delta_voltage 하나여야 합니다")
    if metadata.action_range != (-1.0, 1.0):
        raise ModelCompatibilityError("SAC Action 정규화 범위는 -1~1이어야 합니다")
    if not math.isfinite(metadata.max_delta_voltage) or metadata.max_delta_voltage <= 0:
        raise ModelCompatibilityError("Action 전압 변화량은 양수여야 합니다")
    return metadata


def verify_model_artifact(metadata: ModelMetadata) -> None:
    try:
        digest = hashlib.sha256(metadata.model_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ModelCompatibilityError(f"모델 파일을 읽을 수 없습니다: {exc}") from exc
    if digest != metadata.sha256:
        raise ModelCompatibilityError("모델 파일 SHA-256이 메타데이터와 일치하지 않습니다")
