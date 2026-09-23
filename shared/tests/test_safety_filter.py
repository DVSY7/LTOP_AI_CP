"""Safety Filter 입력·출력·제한 사유 검증."""

import math
import unittest

from shared.control_core.safety_filter import SafetyLimits, filter_voltage_request


class SafetyFilterTests(unittest.TestCase):
    def setUp(self):
        # 사용자에게 확인한 실제 안전 한계다.
        self.limits = SafetyLimits(
            min_set_voltage=0.0,
            max_set_voltage=60.0,
            max_delta_voltage=3.0,
        )

    def test_request_inside_all_limits_passes_through(self):
        result = filter_voltage_request(
            current_set_voltage=45.0,
            requested_delta_voltage=1.5,
            limits=self.limits,
        )
        self.assertEqual(result.requested_set_voltage, 46.5)
        self.assertEqual(result.filtered_set_voltage, 46.5)
        self.assertEqual(result.filtered_delta_voltage, 1.5)
        self.assertFalse(result.intervened)
        self.assertEqual(result.reasons, ())

    def test_positive_and_negative_delta_are_limited(self):
        positive = filter_voltage_request(
            current_set_voltage=45.0,
            requested_delta_voltage=5.0,
            limits=self.limits,
        )
        negative = filter_voltage_request(
            current_set_voltage=45.0,
            requested_delta_voltage=-5.0,
            limits=self.limits,
        )
        self.assertEqual(positive.filtered_delta_voltage, 3.0)
        self.assertEqual(positive.reasons, ("delta_above_max",))
        self.assertEqual(negative.filtered_delta_voltage, -3.0)
        self.assertEqual(negative.reasons, ("delta_below_min",))

    def test_absolute_voltage_limits_apply_after_delta_limit(self):
        upper = filter_voltage_request(
            current_set_voltage=59.0,
            requested_delta_voltage=2.0,
            limits=self.limits,
        )
        lower = filter_voltage_request(
            current_set_voltage=1.0,
            requested_delta_voltage=-2.0,
            limits=self.limits,
        )
        self.assertEqual(upper.filtered_set_voltage, 60.0)
        self.assertEqual(upper.filtered_delta_voltage, 1.0)
        self.assertEqual(upper.reasons, ("voltage_above_max",))
        self.assertEqual(lower.filtered_set_voltage, 0.0)
        self.assertEqual(lower.filtered_delta_voltage, -1.0)
        self.assertEqual(lower.reasons, ("voltage_below_min",))

    def test_multiple_interventions_are_recorded_in_order(self):
        result = filter_voltage_request(
            current_set_voltage=59.0,
            requested_delta_voltage=10.0,
            limits=self.limits,
        )
        self.assertEqual(result.requested_set_voltage, 69.0)
        self.assertEqual(result.filtered_set_voltage, 60.0)
        self.assertEqual(result.reasons, ("delta_above_max", "voltage_above_max"))

    def test_invalid_limits_and_inputs_are_rejected(self):
        invalid_limits = (
            dict(min_set_voltage=60, max_set_voltage=0, max_delta_voltage=1),
            dict(min_set_voltage=0, max_set_voltage=60, max_delta_voltage=0),
            dict(min_set_voltage=0, max_set_voltage=math.inf, max_delta_voltage=1),
        )
        for values in invalid_limits:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SafetyLimits(**values)

        for current, delta in ((-1, 0), (61, 0), (45, math.nan), (math.inf, 0)):
            with self.subTest(current=current, delta=delta), self.assertRaises(ValueError):
                filter_voltage_request(
                    current_set_voltage=current,
                    requested_delta_voltage=delta,
                    limits=self.limits,
                )


if __name__ == "__main__":
    unittest.main()
