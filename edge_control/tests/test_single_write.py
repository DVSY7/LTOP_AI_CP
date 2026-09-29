"""단일 시험 Write의 다중 차단 조건과 Read-back을 검증한다."""

from dataclasses import replace
import unittest
from unittest.mock import Mock

from edge_control.src.config_loader import load_config
from edge_control.src.communication.test_writer import (
    TestWriteError,
    decode_single_register,
    encode_single_register,
    perform_single_write,
)


class SingleWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("edge_control/configs/rectifier.yaml")
        cls.set_voltage = next(
            register for register in cls.config.registers if register.name == "set_voltage"
        )

    def test_engineering_value_encoding(self):
        self.assertEqual(encode_single_register(45.0, self.set_voltage), 450)
        self.assertEqual(encode_single_register(45.001, self.set_voltage), 450)

    def test_half_up_register_resolution(self):
        register = replace(self.set_voltage, scale=0.1)
        self.assertEqual(encode_single_register(44.19, register), 442)
        self.assertEqual(decode_single_register(442, register), 44.2)
        self.assertEqual(encode_single_register(44.25, register), 443)
        self.assertEqual(decode_single_register(443, register), 44.3)

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
        read_response = Mock(registers=[445])
        read_response.isError.return_value = False
        client.write_register.return_value = write_response
        client.read_holding_registers.return_value = read_response

        result = perform_single_write(enabled, "set_voltage", 44.5, lambda *args, **kwargs: client)

        self.assertEqual(result["raw_value"], 445)
        client.write_register.assert_called_once_with(
            self.set_voltage.request_address, 445, device_id=1)
        client.read_holding_registers.assert_called_once_with(
            self.set_voltage.request_address, count=1, device_id=1)
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
