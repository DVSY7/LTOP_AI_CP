import numpy as np

from env.cathodic_env import CathodicProtectionEnv


def run_case(
    case_name: str,
    action_value: float,
    seed: int = 42,
    max_steps: int = 50,
):

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )

    observation, info = env.reset(
        seed=seed
    )

    initial_voltage = info["output_voltage"]
    initial_current = info["output_current"]
    initial_tb = info["pipe_potential"]

    voltages = [
        initial_voltage
    ]

    currents = [
        initial_current
    ]

    tb_values = [
        initial_tb
    ]

    model_limit_hit_count = 0


    # --------------------------------------------------------
    # SAC normalized action
    #
    # -1.0 → 최대 전압 감소
    #  0.0 → HOLD
    # +1.0 → 최대 전압 증가
    # --------------------------------------------------------

    action = np.array(
        [action_value],
        dtype=np.float32,
    )


    for step in range(
        1,
        max_steps + 1,
    ):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        voltages.append(
            info["output_voltage"]
        )

        currents.append(
            info["output_current"]
        )

        tb_values.append(
            info["pipe_potential"]
        )


        if info["model_limit_hit"]:
            model_limit_hit_count += 1


        if (
            terminated
            or
            truncated
        ):
            break


    result = {
        "case_name":
            case_name,

        "action_value":
            action_value,

        "initial_voltage":
            initial_voltage,

        "final_voltage":
            voltages[-1],

        "min_voltage":
            min(voltages),

        "max_voltage":
            max(voltages),

        "initial_current":
            initial_current,

        "final_current":
            currents[-1],

        "initial_tb":
            initial_tb,

        "final_tb":
            tb_values[-1],

        "min_tb":
            min(tb_values),

        "max_tb":
            max(tb_values),

        "tb_change":
            tb_values[-1]
            - initial_tb,

        "model_limit_hit_count":
            model_limit_hit_count,

        "steps":
            len(tb_values) - 1,
    }


    env.close()

    return result


def test_environment_controllability():

    print(
        "\n"
        "===== Environment Controllability Test ====="
    )


    # --------------------------------------------------------
    # 같은 Seed 사용
    #
    # 세 Case가 정확히 같은 실제 초기 상태에서 시작하도록 한다.
    # --------------------------------------------------------

    seed = 42


    cases = [
        (
            "MAX DECREASE",
            -1.0,
        ),

        (
            "HOLD",
            0.0,
        ),

        (
            "MAX INCREASE",
            +1.0,
        ),
    ]


    results = []


    for (
        case_name,
        action_value,
    ) in cases:

        result = run_case(
            case_name=case_name,
            action_value=action_value,
            seed=seed,
            max_steps=50,
        )

        results.append(
            result
        )


    # ========================================================
    # 초기 상태
    # ========================================================

    first = results[0]

    print(
        "\n[동일 초기 상태]"
    )

    print(
        f"Voltage : "
        f"{first['initial_voltage']:.4f} V"
    )

    print(
        f"Current : "
        f"{first['initial_current']:.4f} A"
    )

    print(
        f"TB      : "
        f"{first['initial_tb']:.3f} mV"
    )


    # ========================================================
    # Case별 결과
    # ========================================================

    for result in results:

        print(
            "\n"
            f"===== {result['case_name']} ====="
        )

        print(
            f"SAC Action : "
            f"{result['action_value']:+.1f}"
        )

        print(
            f"Steps      : "
            f"{result['steps']}"
        )

        print(
            f"Voltage    : "
            f"{result['initial_voltage']:.4f}"
            f" → "
            f"{result['final_voltage']:.4f} V"
        )

        print(
            f"V Range    : "
            f"{result['min_voltage']:.4f}"
            f" ~ "
            f"{result['max_voltage']:.4f} V"
        )

        print(
            f"Current    : "
            f"{result['initial_current']:.4f}"
            f" → "
            f"{result['final_current']:.4f} A"
        )

        print(
            f"TB         : "
            f"{result['initial_tb']:.3f}"
            f" → "
            f"{result['final_tb']:.3f} mV"
        )

        print(
            f"TB Change  : "
            f"{result['tb_change']:+.3f} mV"
        )

        print(
            f"TB Range   : "
            f"{result['min_tb']:.3f}"
            f" ~ "
            f"{result['max_tb']:.3f} mV"
        )

        print(
            f"Limit Hit  : "
            f"{result['model_limit_hit_count']}회"
        )


    # ========================================================
    # Case 간 최종 TB 비교
    # ========================================================

    decrease_result = results[0]
    hold_result = results[1]
    increase_result = results[2]


    print(
        "\n"
        "===== 최종 TB 비교 ====="
    )

    print(
        f"MAX DECREASE : "
        f"{decrease_result['final_tb']:.3f} mV"
    )

    print(
        f"HOLD         : "
        f"{hold_result['final_tb']:.3f} mV"
    )

    print(
        f"MAX INCREASE : "
        f"{increase_result['final_tb']:.3f} mV"
    )


    print(
        "\n"
        "===== HOLD 대비 차이 ====="
    )

    print(
        f"MAX DECREASE - HOLD : "
        f"{(
            decrease_result['final_tb']
            - hold_result['final_tb']
        ):+.3f} mV"
    )

    print(
        f"MAX INCREASE - HOLD : "
        f"{(
            increase_result['final_tb']
            - hold_result['final_tb']
        ):+.3f} mV"
    )


    # --------------------------------------------------------
    # 최소한 실행 자체가 정상인지 확인
    # --------------------------------------------------------

    assert (
        decrease_result["steps"]
        ==
        50
    )

    assert (
        hold_result["steps"]
        ==
        50
    )

    assert (
        increase_result["steps"]
        ==
        50
    )


    print(
        "\nEnvironment Controllability Test 완료"
    )


if __name__ == "__main__":

    test_environment_controllability()