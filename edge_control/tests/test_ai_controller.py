"""AI Shadow Mode 추론, 정규화, Safety Filter 및 Write 부재 검증."""

from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from edge_control.src.controller.ai_controller import AiController, build_ai_action
from edge_control.src.controller.control_guard import GuardResult
from shared.control_core import ModelMetadata, SafetyLimits


METADATA = ModelMetadata(
    version="test-4state",
    algorithm="SAC",
    model_path=Path("unused.zip"),
    sha256="0" * 64,
    state_fields=("rectifier_voltage", "rectifier_current", "tb_potential", "tb_trend"),
    state_units=("V", "A", "mV", "mV/cycle"),
    state_bounds=((0, 60), (0, 30), (-3000, 0), (-10, 10)),
    action_fields=("delta_voltage",),
    action_range=(-1, 1),
    max_delta_voltage=0.05,
)
LIMITS = SafetyLimits(0, 60, 3)
OPERATING_LIMITS = SafetyLimits(42, 45, 3)
ALLOWED = GuardResult(True, True, False, ("write_disabled",))
AI_STATES = {"operation_mode": "AI"}
PROCESSED = {
    "status": "valid",
    "values": {
        "rectifier_voltage": 30.0,
        "rectifier_current": 15.0,
        "tb_potential": -1500.0,
        "tb_trend": 0.0,
    },
}


class AiControllerTests(unittest.TestCase):
    def test_prediction_is_scaled_filtered_and_never_written(self):
        model = Mock()
        model.predict.return_value = (np.array([0.8], dtype=np.float32), None)
        action = AiController(model, METADATA, LIMITS).propose(PROCESSED, 59.99)

        observation = model.predict.call_args.args[0]
        np.testing.assert_allclose(observation, [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(action.original_delta_voltage, 0.04)
        self.assertAlmostEqual(action.filtered_delta_voltage, 0.01)
        self.assertEqual(action.target_set_voltage, 60.0)
        self.assertTrue(action.safety_intervened)
        self.assertIsNone(action.actual_write_voltage)

    def test_warmup_and_guard_block_do_not_call_model(self):
        controller = Mock()
        warming = build_ai_action(
            states=AI_STATES, values={"set_voltage": 45}, guard=ALLOWED,
            processed_state={"status": "warming_up"}, controller=controller)
        blocked = build_ai_action(
            states=AI_STATES, values={"set_voltage": 45},
            guard=GuardResult(True, False, False, ("not_remote", "write_disabled")),
            processed_state=PROCESSED, controller=controller)

        self.assertEqual(warming["status"], "warming_up")
        self.assertEqual(blocked["status"], "blocked")
        controller.propose.assert_not_called()

    def test_ai_record_separates_model_filter_and_write_values(self):
        model = Mock()
        model.predict.return_value = (np.array([-0.5], dtype=np.float32), None)
        record = build_ai_action(
            states=AI_STATES, values={"set_voltage": 45}, guard=ALLOWED,
            processed_state=PROCESSED,
            controller=AiController(model, METADATA, LIMITS),
        )
        self.assertEqual(record["status"], "generated")
        self.assertEqual(record["normalized_action"], -0.5)
        self.assertEqual(record["original_delta_voltage"], -0.03)
        self.assertEqual(record["filtered_delta_voltage"], -0.03)
        self.assertIsNone(record["actual_write_voltage"])

    def test_operating_range_blocks_ai_below_42v_and_recovers_outside_value(self):
        model = Mock()
        model.predict.return_value = (np.array([-1.0], dtype=np.float32), None)
        controller = AiController(model, METADATA, OPERATING_LIMITS)

        at_lower_limit = controller.propose(PROCESSED, 42.0)
        below_lower_limit = controller.propose(PROCESSED, 41.8)

        self.assertEqual(at_lower_limit.target_set_voltage, 42.0)
        self.assertEqual(at_lower_limit.filtered_delta_voltage, 0.0)
        self.assertIn("voltage_below_min", at_lower_limit.safety_reasons)
        self.assertEqual(below_lower_limit.target_set_voltage, 42.0)
        self.assertEqual(below_lower_limit.safety_reasons, ("current_voltage_below_min",))


if __name__ == "__main__":
    unittest.main()
