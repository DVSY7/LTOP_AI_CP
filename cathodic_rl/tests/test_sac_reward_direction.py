"""SAC 보상이 TB 구간별 올바른 Action 방향을 선호하는지 검증한다."""

import unittest

from cathodic_rl.reward.reward_function import calculate_sac_reward


def reward(previous_tb: float, current_tb: float, delta_v: float) -> float:
    return calculate_sac_reward(
        previous_pipe_potential=previous_tb,
        pipe_potential=current_tb,
        output_voltage=43.6,
        output_current=5.95,
        applied_delta_voltage=delta_v,
    )


class SacRewardDirectionTest(unittest.TestCase):
    def test_below_prefers_voltage_decrease(self):
        self.assertGreater(
            reward(-1640.0, -1640.0, -0.05),
            reward(-1640.0, -1640.0, +0.05),
        )

    def test_above_prefers_voltage_increase(self):
        self.assertGreater(
            reward(-1560.0, -1560.0, +0.05),
            reward(-1560.0, -1560.0, -0.05),
        )

    def test_target_prefers_hold(self):
        hold = reward(-1600.0, -1600.0, 0.0)
        self.assertGreater(hold, reward(-1600.0, -1600.0, -0.05))
        self.assertGreater(hold, reward(-1600.0, -1600.0, +0.05))


if __name__ == "__main__":
    unittest.main()
