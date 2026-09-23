"""엣지 State Processor가 무효 주기와 불완전 State를 차단하는지 검증."""

import unittest

from edge_control.src.state.state_processor import StateProcessor


class StateProcessorTests(unittest.TestCase):
    def test_second_consecutive_state_adds_tb_trend(self):
        processor = StateProcessor()
        first = processor.process(
            values={
                "rectifier_voltage": 45,
                "rectifier_current": 12,
                "tb_potential": -1500,
            },
            state_usable=True,
        )
        result = processor.process(
            values={
                "rectifier_voltage": 46,
                "rectifier_current": 12.5,
                "tb_potential": -1504,
            },
            state_usable=True,
        )
        self.assertEqual(first["status"], "warming_up")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["field_order"], [
            "rectifier_voltage", "rectifier_current", "tb_potential", "tb_trend"])
        self.assertEqual(result["vector"], [46.0, 12.5, -1504.0, -4.0])
        self.assertIsNone(result["normalized_vector"])

    def test_unusable_cycle_resets_trend_history(self):
        processor = StateProcessor()
        values = {"rectifier_voltage": 45, "rectifier_current": 12, "tb_potential": -1500}
        processor.process(values=values, state_usable=True)
        self.assertIsNone(processor.process(values=None, state_usable=False))
        result = processor.process(values={**values, "tb_potential": -1510}, state_usable=True)
        self.assertEqual(result["status"], "warming_up")

    def test_missing_state_is_not_fabricated(self):
        result = StateProcessor().process(
            values={"rectifier_voltage": 45, "rectifier_current": 12},
            state_usable=True,
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("tb_potential", result["error"])


if __name__ == "__main__":
    unittest.main()
