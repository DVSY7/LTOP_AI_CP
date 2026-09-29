"""Control writer reports the read-back-verified quantized value."""

import unittest
from unittest.mock import patch

from edge_control.src.communication.control_writer import write_control_target


class ControlWriterTests(unittest.TestCase):
    def test_actual_write_voltage_uses_quantized_applied_value(self):
        config = type("Config", (), {
            "control": type("Control", (), {"enabled": True, "write_signal": "set_voltage"})()
        })()
        with patch(
            "edge_control.src.communication.control_writer.perform_single_write",
            return_value={"requested_value": 44.19, "applied_value": 44.2, "raw_value": 442},
        ):
            result = write_control_target(config, 44.19)
        self.assertEqual(result["actual_write_voltage"], 44.2)
        self.assertTrue(result["read_back_verified"])


if __name__ == "__main__":
    unittest.main()
