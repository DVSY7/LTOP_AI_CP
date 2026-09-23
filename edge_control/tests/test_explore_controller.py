"""EXPLORE Action 생성, Safety Filter 적용 및 차단 동작 검증."""

import unittest
from unittest.mock import Mock

from edge_control.src.controller.control_guard import GuardResult
from edge_control.src.controller.explore_controller import (
    ExploreController,
    PatternExploreController,
    build_explore_action,
)
from shared.control_core import SafetyLimits


LIMITS = SafetyLimits(0.0, 60.0, 3.0)
EXPLORE_STATES = {"operation_mode": "EXPLORE"}
ALLOWED = GuardResult(True, True, False, ("write_disabled",))


class ExploreControllerTests(unittest.TestCase):
    def test_continuous_action_uses_configured_range(self):
        random_source = Mock()
        random_source.uniform.return_value = 1.25
        action = ExploreController(LIMITS, random_source).propose(45.0)

        random_source.uniform.assert_called_once_with(-3.0, 3.0)
        self.assertEqual(action.original_delta_voltage, 1.25)
        self.assertEqual(action.filtered_delta_voltage, 1.25)
        self.assertEqual(action.target_set_voltage, 46.25)
        self.assertFalse(action.safety_intervened)
        self.assertIsNone(action.actual_write_voltage)

    def test_safety_filter_limits_action_at_voltage_boundary(self):
        random_source = Mock()
        random_source.uniform.return_value = 3.0
        action = ExploreController(LIMITS, random_source).propose(59.0)

        self.assertEqual(action.original_delta_voltage, 3.0)
        self.assertEqual(action.filtered_delta_voltage, 1.0)
        self.assertEqual(action.target_set_voltage, 60.0)
        self.assertTrue(action.safety_intervened)
        self.assertEqual(action.safety_reasons, ("voltage_above_max",))

    def test_guard_block_and_missing_set_voltage_do_not_generate(self):
        controller = Mock()
        blocked = build_explore_action(
            states=EXPLORE_STATES,
            values={"set_voltage": 45.0},
            guard=GuardResult(True, False, False, ("not_remote", "write_disabled")),
            controller=controller,
        )
        missing = build_explore_action(
            states=EXPLORE_STATES,
            values={},
            guard=ALLOWED,
            controller=controller,
        )

        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(missing["reasons"], ["set_voltage_missing"])
        controller.propose.assert_not_called()

    def test_only_explore_mode_generates_action_record(self):
        controller = Mock()
        for mode in ("AI", "MANUAL"):
            with self.subTest(mode=mode):
                self.assertIsNone(build_explore_action(
                    states={"operation_mode": mode},
                    values={"set_voltage": 45.0},
                    guard=ALLOWED,
                    controller=controller,
                ))
        controller.propose.assert_not_called()

    def test_generated_record_keeps_original_filtered_and_write_separate(self):
        random_source = Mock()
        random_source.uniform.return_value = -3.0
        record = build_explore_action(
            states=EXPLORE_STATES,
            values={"set_voltage": 1.0},
            guard=ALLOWED,
            controller=ExploreController(LIMITS, random_source),
        )

        self.assertEqual(record["status"], "generated")
        self.assertEqual(record["original_delta_voltage"], -3.0)
        self.assertEqual(record["filtered_delta_voltage"], -1.0)
        self.assertEqual(record["target_set_voltage"], 0.0)
        self.assertIsNone(record["actual_write_voltage"])

    def test_log_record_rounds_voltage_values_to_two_decimal_places(self):
        random_source = Mock()
        random_source.uniform.return_value = 1.234567
        action = ExploreController(LIMITS, random_source).propose(45.001)
        record = action.to_dict()

        # 내부 계산값은 유지하고 JSONL에 전달할 표현값만 반올림한다.
        self.assertEqual(action.original_delta_voltage, 1.234567)
        self.assertEqual(record["original_delta_voltage"], 1.23)
        self.assertEqual(record["filtered_delta_voltage"], 1.23)
        self.assertEqual(record["target_set_voltage"], 46.24)

    def test_up_down_pattern_reverses_at_both_voltage_limits(self):
        controller = PatternExploreController(
            SafetyLimits(42.0, 45.0, 0.2), pattern="up_down", step_voltage=0.2,
            hold_every_steps=4, hold_steps=3,
        )

        self.assertEqual(controller.propose(44.8).target_set_voltage, 45.0)
        self.assertEqual(controller.propose(45.0).target_set_voltage, 44.8)
        self.assertEqual(controller.propose(42.0).target_set_voltage, 42.2)

    def test_hold_pattern_inserts_configured_number_of_holds(self):
        controller = PatternExploreController(
            SafetyLimits(42.0, 45.0, 0.2), pattern="hold", step_voltage=0.2,
            hold_every_steps=2, hold_steps=3,
        )

        first = controller.propose(42.2)
        second = controller.propose(first.target_set_voltage)
        self.assertAlmostEqual(first.target_set_voltage, 42.4)
        self.assertAlmostEqual(second.target_set_voltage, 42.6)

        for _ in range(3):
            hold = controller.propose(42.6)
            self.assertEqual(hold.filtered_delta_voltage, 0.0)
            self.assertAlmostEqual(hold.target_set_voltage, 42.6)

        self.assertAlmostEqual(controller.propose(42.6).target_set_voltage, 42.8)


if __name__ == "__main__":
    unittest.main()
