"""모델 메타데이터와 파일 무결성 검증."""

from pathlib import Path
import tempfile
import unittest

import yaml

from shared.control_core import (
    ModelCompatibilityError,
    load_model_metadata,
    verify_model_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "cathodic_rl/models/trained/cathodic_sac.metadata.yaml"


class ModelLoaderTests(unittest.TestCase):
    def test_trained_model_matches_runtime_contract_and_checksum(self):
        metadata = load_model_metadata(METADATA)
        verify_model_artifact(metadata)
        self.assertEqual(metadata.state_fields, (
            "rectifier_voltage", "rectifier_current", "tb_potential", "tb_trend"))
        self.assertEqual(metadata.action_range, (-1.0, 1.0))
        self.assertEqual(metadata.max_delta_voltage, 0.05)

    def test_wrong_field_order_and_changed_model_are_rejected(self):
        source = yaml.safe_load(METADATA.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bad_metadata = directory / "metadata.yaml"
            source["state"]["fields"].reverse()
            bad_metadata.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaises(ModelCompatibilityError):
                load_model_metadata(bad_metadata)

            source = yaml.safe_load(METADATA.read_text(encoding="utf-8"))
            (directory / "model.zip").write_bytes(b"changed")
            source["model_file"] = "model.zip"
            bad_metadata.write_text(yaml.safe_dump(source), encoding="utf-8")
            metadata = load_model_metadata(bad_metadata)
            with self.assertRaises(ModelCompatibilityError):
                verify_model_artifact(metadata)


if __name__ == "__main__":
    unittest.main()
