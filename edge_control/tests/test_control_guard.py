"""Control Guard는 판정만 하며 장비 통신이나 Write를 수행하지 않는다."""

import unittest

from edge_control.src.controller.control_guard import evaluate_control_guard


NORMAL_EXPLORE = {
    "remote_mode": "REMOTE",
    "operation_status": "ON",
    "operation_mode": "EXPLORE",
    "tb_status": "NORMAL",
}


class ControlGuardTests(unittest.TestCase):
    def test_explore_state_allows_action_but_read_only_blocks_write(self):
        result = evaluate_control_guard(
            cycle_valid=True,
            states=NORMAL_EXPLORE,
            write_enabled=False,
        )
        self.assertTrue(result.state_usable)
        self.assertTrue(result.action_allowed)
        self.assertFalse(result.write_allowed)
        self.assertEqual(result.reasons, ("write_disabled",))

    def test_ai_and_explore_allow_action_in_normal_conditions(self):
        for mode in ("AI", "EXPLORE"):
            with self.subTest(mode=mode):
                result = evaluate_control_guard(
                    cycle_valid=True,
                    states={**NORMAL_EXPLORE, "operation_mode": mode},
                    write_enabled=False,
                )
                self.assertTrue(result.action_allowed)
                self.assertFalse(result.write_allowed)
                self.assertEqual(result.reasons, ("write_disabled",))

    def test_manual_local_off_and_tb_error_are_blocked(self):
        cases = (
            ({**NORMAL_EXPLORE, "operation_mode": "MANUAL"}, "manual_mode"),
            ({**NORMAL_EXPLORE, "remote_mode": "LOCAL"}, "not_remote"),
            ({**NORMAL_EXPLORE, "operation_status": "OFF"}, "operation_off"),
            ({**NORMAL_EXPLORE, "tb_status": "ERR_COM"}, "tb_status_err_com"),
        )
        for states, reason in cases:
            with self.subTest(reason=reason):
                result = evaluate_control_guard(
                    cycle_valid=True,
                    states=states,
                    write_enabled=False,
                )
                self.assertFalse(result.action_allowed)
                self.assertFalse(result.write_allowed)
                self.assertIn(reason, result.reasons)

    def test_invalid_or_incomplete_cycle_is_not_usable(self):
        invalid = evaluate_control_guard(
            cycle_valid=False,
            states=None,
            write_enabled=False,
        )
        self.assertFalse(invalid.state_usable)
        self.assertEqual(invalid.reasons, ("invalid_cycle",))

        incomplete = evaluate_control_guard(
            cycle_valid=True,
            states={"operation_mode": "EXPLORE"},
            write_enabled=False,
        )
        self.assertFalse(incomplete.state_usable)
        self.assertTrue(any(reason.startswith("missing_") for reason in incomplete.reasons))

    def test_write_requires_both_allowed_state_and_explicit_enable(self):
        enabled = evaluate_control_guard(
            cycle_valid=True,
            states=NORMAL_EXPLORE,
            write_enabled=True,
        )
        self.assertTrue(enabled.write_allowed)

        manual = evaluate_control_guard(
            cycle_valid=True,
            states={**NORMAL_EXPLORE, "operation_mode": "MANUAL"},
            write_enabled=True,
        )
        self.assertFalse(manual.write_allowed)


if __name__ == "__main__":
    unittest.main()
