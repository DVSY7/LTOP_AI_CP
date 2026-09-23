"""단일 시험 Write의 다중 차단 조건과 Read-back을 검증한다."""

from dataclasses import replace
import unittest
from unittest.mock import Mock

from edge_control.src.config_loader import load_config
from edge_control.src.communication.test_writer import (
    TestWriteError,
    encode_single_register,
    perform_single_write,
)


class SingleWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("edge_control/configs/junction_test.yaml")
        cls.set_voltage = next(
            register for register in cls.config.registers if register.name == "set_voltage"
        )

    def test_engineering_value_encoding(self):
        self.assertEqual(encode_single_register(45.0, self.set_voltage), 4500)
        with self.assertRaises(TestWriteError):
            encode_single_register(45.001, self.set_voltage)

    def test_write_disabled_blocks_before_client_creation(self):
        factory = Mock()
        disabled = replace(self.config, write_enabled=False)
        with self.assertRaises(TestWriteError):
            perform_single_write(disabled, "set_voltage", 45.0, factory)
        factory.assert_not_called()

    def test_read_only_signal_is_rejected(self):
        factory = Mock()
        enabled = replace(self.config, write_enabled=True)
        with self.assertRaises(TestWriteError):
            perform_single_write(enabled, "rectifier_voltage", 45.0, factory)
        factory.assert_not_called()

    def test_exactly_one_write_and_read_back(self):
        enabled = replace(self.config, write_enabled=True)
        client = Mock()
        client.connect.return_value = True
        write_response = Mock()
        write_response.isError.return_value = False
        read_response = Mock(registers=[4450])
        read_response.isError.return_value = False
        client.write_register.return_value = write_response
        client.read_holding_registers.return_value = read_response

        result = perform_single_write(enabled, "set_voltage", 44.5, lambda *args, **kwargs: client)

        self.assertEqual(result["raw_value"], 4450)
        client.write_register.assert_called_once_with(20, 4450, device_id=1)
        client.read_holding_registers.assert_called_once_with(20, count=1, device_id=1)
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
