"""Actual Modbus set-voltage resolution shared by training helpers."""

from decimal import Decimal, ROUND_HALF_UP


def quantize_voltage(value: float, step: float) -> float:
    """Round an absolute engineering voltage to a register step.

    Decimal and ROUND_HALF_UP intentionally match edge_control's Modbus writer.
    Python's built-in round uses banker's rounding and must not be used here.
    """
    if step <= 0:
        raise ValueError("voltage register step must be positive")
    decimal_step = Decimal(str(step))
    quantized = (
        Decimal(str(value)) / decimal_step
    ).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * decimal_step
    return float(quantized)


def quantized_target_delta(current_voltage: float, requested_delta: float, step: float) -> tuple[float, float]:
    """Return the quantized absolute target and its effective delta."""
    # Add decimal engineering values before quantizing. Adding binary floats
    # first can turn 42.3 + 0.05 into 42.349999..., changing a half-up result.
    target = quantize_voltage(
        Decimal(str(current_voltage)) + Decimal(str(requested_delta)), step)
    return target, float(Decimal(str(target)) - Decimal(str(current_voltage)))
