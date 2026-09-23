import numpy as np

from cathodic_rl.env.cathodic_env import (
    CathodicProtectionEnv,
)


def test_real_data_step():

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )


    # --------------------------------------------------------
    # Seed를 고정해 같은 Reset 상태 사용
    # --------------------------------------------------------

    observation, info = (
        env.reset(
            seed=42
        )
    )


    print(
        "\n===== 실데이터 기반 Gym Step 테스트 ====="
    )


    print(
        "\n[초기 상태]"
    )

    print(
        f"Voltage : "
        f"{info['output_voltage']:.4f} V"
    )

    print(
        f"Current : "
        f"{info['output_current']:.4f} A"
    )

    print(
        f"TB      : "
        f"{info['pipe_potential']:.3f} mV"
    )

    print(
        f"History : "
        f"{env.environment_model.tb_history}"
    )


    # ========================================================
    # SAC Action 테스트
    #
    # 현재 SAC 변환식:
    #
    # normalized_action * 0.5V
    #
    # 따라서:
    #
    # +0.10 → +0.05V
    # ========================================================

    action = np.array(
        [0.1],
        dtype=np.float32,
    )


    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        action
    )


    print(
        "\n[1 Step 후]"
    )

    print(
        f"요청 Delta_V     : "
        f"{info['requested_delta_voltage']:+.4f} V"
    )

    print(
        f"Safety Delta_V   : "
        f"{info['safety_delta_voltage']:+.4f} V"
    )

    print(
        f"Effective Delta_V: "
        f"{info['effective_delta_voltage']:+.4f} V"
    )

    print(
        f"Model Limit Hit  : "
        f"{info['model_limit_hit']}"
    )


    print(
        f"\nVoltage : "
        f"{info['output_voltage']:.4f} V"
    )

    print(
        f"Current : "
        f"{info['output_current']:.4f} A"
    )

    print(
        f"TB      : "
        f"{info['pipe_potential']:.3f} mV"
    )


    print(
        "\n[Current Model]"
    )

    print(
        f"Delta_I Total    : "
        f"{info['delta_i_total']:+.4f} A"
    )

    print(
        f"Delta_I Baseline : "
        f"{info['delta_i_baseline']:+.4f} A"
    )

    print(
        f"Delta_I Action   : "
        f"{info['delta_i_action']:+.4f} A"
    )


    print(
        "\n[TB Model]"
    )

    print(
        f"Natural TB       : "
        f"{info['tb_natural_next']:.3f} mV"
    )

    print(
        f"Control Effect   : "
        f"{info['delta_tb_control']:+.3f} mV"
    )


    print(
        "\n[Gym]"
    )

    print(
        f"Observation : "
        f"{observation}"
    )

    print(
        f"Reward      : "
        f"{reward:.4f}"
    )

    print(
        f"Current Step: "
        f"{info['current_step']}"
    )


    print(
        "\nTB History :"
    )

    print(
        env.environment_model.tb_history
    )


    # --------------------------------------------------------
    # 기본 검사
    # --------------------------------------------------------

    assert (
        observation.shape
        ==
        (3,)
    )

    assert (
        observation.dtype
        ==
        np.float32
    )

    assert (
        env.observation_space.contains(
            observation
        )
    )

    assert (
        info["current_step"]
        ==
        1
    )

    assert (
        len(
            env.environment_model.tb_history
        )
        ==
        7
    )

    assert np.isclose(
        env.environment_model.tb_history[-1],
        info["pipe_potential"],
    )

    assert (
        terminated
        is False
    )

    assert (
        truncated
        is False
    )


    env.close()


    print(
        "\n실데이터 기반 Gym Step 테스트 통과"
    )


if __name__ == "__main__":

    test_real_data_step()