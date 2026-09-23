"""장비 접속 없이 설정, 변환, 장애 복구와 Write 부재를 검증한다."""

from dataclasses import replace
import io
import json
import logging
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml
from pymodbus.exceptions import ConnectionException

from edge_control.src.config_loader import ConfigError, load_config
from edge_control.src.communication.modbus_client import ReadOnlyClient, ReadError
from edge_control.src.communication.register_reader import decode, timestamp
from edge_control.src.logging_setup import ConsoleRecordFormatter, setup_logging, emit
from edge_control.src.main import (
    build_learning_measurements,
    main,
    operating_voltage_limits,
    run,
)


FIXTURE = Path(__file__).parent / "fixtures" / "confirmed_mock.yaml"
EDGE = Path(__file__).resolve().parents[1]


def response(words):
    result = Mock(registers=words)
    result.isError.return_value = False
    return result


class ReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(FIXTURE)
        self.transport = Mock()
        self.transport.connect.return_value = True
        self.transport.read_holding_registers.return_value = response([432])
        self.transport.read_input_registers.return_value = response([123])
        self.client = ReadOnlyClient(self.config, lambda *a, **k: self.transport)
        self.logger = Mock()

    def records(self):
        return [json.loads(call.args[0]) for call in self.logger.info.call_args_list]

    def test_unconfirmed_profiles_and_cli_never_connect(self):
        for name in ("rectifier",):
            with self.subTest(name=name), self.assertRaises(ConfigError):
                load_config(EDGE / "configs" / f"{name}.yaml")
        with patch("edge_control.src.main.ReadOnlyClient") as factory, patch("sys.stderr", io.StringIO()):
            self.assertEqual(main(["--config", str(EDGE / "configs/rectifier.yaml")]), 2)
            factory.assert_not_called()

    def test_invalid_configuration(self):
        mutations = [("address_base", None),
                     ("poll_interval_seconds", 0), ("connection.unit_id", True),
                     ("registers.rectifier_voltage.scale", float("nan")),
                     ("registers.rectifier_voltage.address", 0),
                     ("registers.rectifier_current.data_type", "unknown")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for key, value in mutations:
                data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
                target = data
                parts = key.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                path.write_text(yaml.safe_dump(data), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ConfigError):
                    load_config(path)

    def test_confirmed_safety_limits_and_invalid_values(self):
        config_path = EDGE / "configs/junction_test.yaml"
        config = load_config(config_path)
        self.assertEqual(config.safety_limits.min_set_voltage, 0.0)
        self.assertEqual(config.safety_limits.max_set_voltage, 60.0)
        self.assertEqual(config.safety_limits.max_delta_voltage, 3.0)
        self.assertTrue(config.control.enabled)
        self.assertGreater(config.control.write_interval_seconds, 0.0)
        self.assertEqual(config.control.explore.step_voltage, 0.2)
        self.assertEqual(config.control.explore.pattern, "hold")
        operating_limits = operating_voltage_limits(config)
        self.assertEqual(operating_limits.min_set_voltage, 42.0)
        self.assertEqual(operating_limits.max_set_voltage, 45.0)

        original = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        mutations = (
            ("min_set_voltage", 60.0),
            ("max_set_voltage", 0.0),
            ("max_delta_voltage", 0.0),
            ("max_delta_voltage", None),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for key, value in mutations:
                data = yaml.safe_load(yaml.safe_dump(original))
                data["safety"][key] = value
                path.write_text(yaml.safe_dump(data), encoding="utf-8")
                with self.subTest(key=key, value=value), self.assertRaises(ConfigError):
                    load_config(path)

    def test_yaml_mapping_change_and_both_read_functions(self):
        data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
        data["registers"]["rectifier_voltage"]["address"] += 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            config = load_config(path)
        run(config, self.client, self.logger, max_cycles=1)
        voltage, current = config.registers
        self.transport.read_holding_registers.assert_called_once_with(
            voltage.address - 1, count=voltage.count, device_id=config.unit_id)
        self.transport.read_input_registers.assert_called_once_with(
            current.address - 1, count=current.count, device_id=config.unit_id)
        self.assertEqual(self.records()[0]["measurements"],
                         {"rectifier_voltage": 43.2, "rectifier_current": 1.23})
        self.assertIn("timestamp", self.records()[0])
        self.assertNotIn("samples", self.records()[0])
        self.assertNotIn("control_guard", self.records()[0])
        self.assertNotIn("processed_state", self.records()[0])

    def test_integer_and_float_decoding_orders(self):
        register = self.config.registers[0]
        for dtype, fmt, number in [("uint16", "H", 1234), ("int16", "h", -1234),
                                    ("uint32", "I", 123456), ("int32", "i", -123456),
                                    ("float32", "f", 12.5)]:
            for byte_order in ("big", "little"):
                for word_order in ("big", "little"):
                    payload = struct.pack(">" + fmt, number)
                    words = [int.from_bytes(payload[i:i+2], byte_order)
                             for i in range(0, len(payload), 2)]
                    if word_order == "little":
                        words.reverse()
                    spec = replace(register, data_type=dtype, byte_order=byte_order,
                                   word_order=word_order, scale=2, offset=3)
                    with self.subTest(dtype=dtype, byte_order=byte_order, word_order=word_order):
                        self.assertEqual(decode(words, spec), number * 2 + 3)
        with self.assertRaises(ReadError):
            decode([0x7fc0, 0], replace(register, data_type="float32"))

    def test_operational_log_rounds_measurements_and_timestamp_to_seconds(self):
        measurements = build_learning_measurements(
            {
                "rectifier_voltage": 43.4567,
                "rectifier_current": 12.3456,
                "set_voltage": 43.4567,
                "tb_potential": -1500.5678,
            },
            {"values": {"tb_trend": 0.1267}},
        )
        self.assertEqual(measurements, {
            "rectifier_voltage": 43.46,
            "rectifier_current": 12.35,
            "set_voltage": 43.46,
            "tb_potential": -1500.57,
            "tb_trend": 0.13,
        })
        self.assertNotIn(".", timestamp().split("+")[0])
        self.assertRegex(timestamp(), r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_partial_failure_reconnect_and_no_stale_values(self):
        self.transport.read_input_registers.side_effect = [response([123]),
                                                           ConnectionException("disconnected"),
                                                           response([456])]
        sleeps = []
        run(self.config, self.client, self.logger, max_cycles=3, sleep=sleeps.append)
        records = self.records()
        self.assertEqual([r["valid"] for r in records], [True, False, True])
        self.assertIsNone(records[1]["measurements"])
        self.assertEqual(records[1]["read_error"]["signal"], "rectifier_current")
        self.assertAlmostEqual(records[2]["measurements"]["rectifier_current"], 4.56)
        self.assertEqual(self.transport.connect.call_count, 2)
        self.assertGreaterEqual(sleeps[1], self.config.reconnect_interval)
        self.assert_no_write()

    def test_normal_cycles_are_recorded_at_configured_interval(self):
        config = replace(self.config, record_interval=60.0)
        clock_values = iter((0.0, 0.0, 1.0, 1.0, 2.0, 2.0))

        run(
            config,
            self.client,
            self.logger,
            max_cycles=3,
            sleep=lambda _: None,
            clock=lambda: next(clock_values),
        )

        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.records()[0]["cycle"], 1)

    def test_initial_connection_failure_recovers(self):
        self.transport.connect.side_effect = [False, True]
        run(self.config, self.client, self.logger, max_cycles=2, sleep=lambda _: None)
        self.assertEqual([r["valid"] for r in self.records()], [False, True])
        self.assert_no_write()

    def test_error_response_and_malformed_data(self):
        error = Mock()
        error.isError.return_value = True
        for result in (None, error, response([]), response([1, 2]), response([-1])):
            self.transport.read_holding_registers.return_value = result
            with self.subTest(result=result), self.assertRaises(ReadError):
                self.client.read(self.config.registers[0])

    def test_keyboard_interrupt_closes_connection(self):
        self.transport.read_holding_registers.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run(self.config, self.client, self.logger, max_cycles=1)
        self.transport.close.assert_called_once()
        self.assert_no_write()

    def test_config_check_does_not_create_client(self):
        with patch("edge_control.src.main.ReadOnlyClient") as factory, patch("sys.stdout", io.StringIO()):
            self.assertEqual(main(["--config", str(FIXTURE), "--check-config"]), 0)
            factory.assert_not_called()

    def test_console_is_compact_while_file_keeps_full_json_record(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config, log_file=Path(directory) / "sample.jsonl")
            with patch("sys.stderr", io.StringIO()) as console:
                logger = setup_logging(config)
                emit(logger, {"event": "시험", "raw": [123], "value": 1.23})
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)
                file_record = json.loads(config.log_file.read_text(encoding="utf-8"))
                self.assertEqual(file_record, {"event": "시험", "raw": [123], "value": 1.23})
                self.assertEqual(console.getvalue(), "[시험]\n")

    def test_console_cycle_uses_compact_measurements(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=json.dumps({
                "event": "cycle", "timestamp": "2026-09-22T14:56:45",
                "cycle": 2, "valid": True,
                "measurements": {
                    "rectifier_voltage": 45.0,
                    "rectifier_current": 12.0,
                    "tb_potential": -1500.0,
                },
                "states": {"operation_mode": "AI"},
                "control_action": {"status": "generated", "target_set_voltage": 44.95},
            }),
            args=(), exc_info=None,
        )
        self.assertEqual(
            ConsoleRecordFormatter().format(record),
            "[14:56:45] #2 AI | V=45.00 V I=12.00 A TB=-1500 mV | 제안=44.95 V",
        )

    def assert_no_write(self):
        calls = {call[0] for call in self.transport.method_calls}
        self.assertFalse(any("write" in name.lower() for name in calls))
        self.assertTrue(calls <= {"connect", "read_holding_registers", "read_input_registers", "close"})

    def mode_config(self):
        data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
        modes = yaml.safe_load((EDGE / "configs/junction_test.yaml").read_text(encoding="utf-8"))
        for name in ("remote_mode", "operation_status", "operation_mode"):
            data["registers"][name] = modes["registers"][name]
        with patch("pathlib.Path.read_text", return_value=yaml.safe_dump(data)):
            return load_config(FIXTURE)

    def test_mode_changes_and_unknown_do_not_write_or_reuse_state(self):
        config = self.mode_config()
        # 전압, Local/Remote, ON/OFF, 자동/수동 순서. 전류는 Input 읽기.
        self.transport.read_holding_registers.side_effect = [response([value]) for value in
            (432, 0, 0, 0, 432, 1, 1, 1, 432, 1, 1, 99, 432, 1, 1, 0)]
        run(config, self.client, self.logger, max_cycles=4, sleep=lambda _: None)
        records = self.records()
        self.assertEqual(records[0]["states"], {
            "remote_mode": "LOCAL", "operation_status": "OFF", "operation_mode": "AI"})
        self.assertEqual(records[1]["states"], {
            "remote_mode": "REMOTE", "operation_status": "ON", "operation_mode": "MANUAL"})
        self.assertFalse(records[2]["valid"])
        self.assertIsNone(records[2]["states"])
        self.assertIsNone(records[2]["measurements"])
        self.assertEqual(records[2]["read_error"]["signal"], "operation_mode")
        self.assertEqual(records[3]["states"]["operation_mode"], "AI")
        self.assert_no_write()

    def test_failed_mode_read_invalidates_entire_cycle(self):
        config = self.mode_config()
        self.transport.read_holding_registers.side_effect = [
            response([432]), ConnectionException("mode read failed")]
        run(config, self.client, self.logger, max_cycles=1)
        self.assertFalse(self.records()[0]["valid"])
        self.assertIsNone(self.records()[0]["states"])
        self.assert_no_write()

    def test_missing_mode_and_invalid_code_mapping_rejected(self):
        data = yaml.safe_load((EDGE / "configs/junction_test.yaml").read_text(encoding="utf-8"))
        del data["registers"]["operation_mode"]
        with patch("pathlib.Path.read_text", return_value=yaml.safe_dump(data)):
            with self.assertRaises(ConfigError):
                load_config(FIXTURE)
        data = yaml.safe_load((EDGE / "configs/junction_test.yaml").read_text(encoding="utf-8"))
        data["registers"]["operation_status"]["codes"] = {0: False, 1: True}
        with patch("pathlib.Path.read_text", return_value=yaml.safe_dump(data)):
            with self.assertRaises(ConfigError):
                load_config(FIXTURE)


if __name__ == "__main__":
    unittest.main()
