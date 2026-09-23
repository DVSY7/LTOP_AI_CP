"""기본 State 필드 순서와 입력 유효성 검증."""

import math
import unittest

from shared.control_core import (
    BASE_STATE_FIELDS,
    POLICY_STATE_FIELDS,
    StateBounds,
    build_base_state,
    build_policy_state,
    normalize_policy_state,
)


class BaseStateTests(unittest.TestCase):
    def test_field_order_units_and_vector_are_fixed(self):
        state = build_base_state({
            "tb_potential": -1500,
            "rectifier_current": 12,
            "rectifier_voltage": 45,
            "unrelated": 99,
        })
        record = state.to_dict()

        self.assertEqual(
            tuple(field.name for field in BASE_STATE_FIELDS),
            ("rectifier_voltage", "rectifier_current", "tb_potential"),
        )
        self.assertEqual(record["vector"], [45.0, 12.0, -1500.0])
        self.assertEqual(record["units"], {
            "rectifier_voltage": "V",
            "rectifier_current": "A",
            "tb_potential": "mV",
        })

    def test_missing_non_numeric_and_non_finite_values_are_rejected(self):
        valid = {"rectifier_voltage": 45, "rectifier_current": 12, "tb_potential": -1500}
        cases = (
            {key: value for key, value in valid.items() if key != "tb_potential"},
            {**valid, "rectifier_current": "unknown"},
            {**valid, "rectifier_voltage": True},
            {**valid, "tb_potential": math.nan},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_base_state(values)

    def test_policy_state_excludes_offset_and_normalizes_four_fields(self):
        state = build_policy_state({
            "rectifier_voltage": 30,
            "rectifier_current": 15,
            "tb_potential": -1500,
        }, tb_trend=0)
        self.assertEqual(
            tuple(field.name for field in POLICY_STATE_FIELDS),
            ("rectifier_voltage", "rectifier_current", "tb_potential", "tb_trend"),
        )
        normalized = normalize_policy_state(state, StateBounds(
            voltage=(0, 60), current=(0, 30), potential=(-3000, 0), tb_trend=(-10, 10)))
        self.assertEqual(normalized, (0.5, 0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
