import numpy as np

from env.cathodic_env import (
    CathodicProtectionEnv,
)


def test_real_data_reset():

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )


    print(
        "\n===== 실제 데이터 기반 Gym Reset 테스트 ====="
    )


    for reset_number in range(
        1,
        6,
    ):

        observation, info = (
            env.reset()
        )


        print(
            f"\n===== Reset {reset_number} ====="
        )

        print(
            f"DateTime : "
            f"{info['reset_datetime']}"
        )

        print(
            f"Segment  : "
            f"{info['reset_segment']}"
        )

        print(
            f"Voltage  : "
            f"{info['output_voltage']:.4f} V"
        )

        print(
            f"Current  : "
            f"{info['output_current']:.4f} A"
        )

        print(
            f"TB       : "
            f"{info['pipe_potential']:.1f} mV"
        )

        print(
            "TB History :"
        )

        print(
            env.environment_model.tb_history
        )

        print(
            "SAC Observation :"
        )

        print(
            observation
        )


        # ----------------------------------------------------
        # 기본 검사
        # ----------------------------------------------------

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
            info["current_step"]
            ==
            0
        )

        assert (
            env.observation_space.contains(
                observation
            )
        )


    env.close()


    print(
        "\n실제 데이터 기반 Gym Reset 테스트 통과"
    )


if __name__ == "__main__":

    test_real_data_reset()