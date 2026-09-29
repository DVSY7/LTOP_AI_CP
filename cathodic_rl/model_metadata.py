"""Create metadata for a newly trained SAC candidate without touching production metadata."""

import hashlib
from pathlib import Path

import yaml

from cathodic_rl.config.settings import SAC_MAX_DELTA_VOLTAGE

STATE_FIELDS = (
    ("rectifier_voltage", "V", 0.0, 60.0),
    ("rectifier_current", "A", 0.0, 30.0),
    ("tb_potential", "mV", -3000.0, 0.0),
    ("tb_trend", "mV/cycle", -10.0, 10.0),
)


def candidate_metadata_path(model_path: str | Path) -> Path:
    """Return the sidecar metadata name for a Stable-Baselines model path."""
    model = Path(model_path)
    if model.suffix == ".zip":
        model = model.with_suffix("")
    return model.with_suffix(".metadata.yaml")


def write_sac_metadata(model_path: str | Path, *, version: str = "sac-4state-dv050-step010-candidate") -> Path:
    """Write a metadata sidecar for an already saved candidate artifact."""
    model = Path(model_path)
    artifact = model if model.suffix == ".zip" else model.with_suffix(".zip")
    if not artifact.is_file():
        raise FileNotFoundError(f"Saved SAC artifact was not found: {artifact}")
    metadata_path = candidate_metadata_path(model)
    content = {
        "model_version": version,
        "algorithm": "SAC",
        "model_file": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "state": {"fields": [
            {"name": name, "unit": unit, "min": minimum, "max": maximum}
            for name, unit, minimum, maximum in STATE_FIELDS
        ], "normalized_range": [0.0, 1.0]},
        "action": {"fields": ["delta_voltage"], "normalized_range": [-1.0, 1.0],
                   "max_delta_voltage": SAC_MAX_DELTA_VOLTAGE},
    }
    metadata_path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return metadata_path
