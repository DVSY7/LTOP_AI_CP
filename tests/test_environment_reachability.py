import numpy as np

from env.cathodic_env import CathodicProtectionEnv


# ============================================================
# 설정
# ============================================================

NUM_SAMPLES = 100
MAX_STEPS = 50

TARGET_MIN = -1610.0
TARGET_MAX = -1590.0


# ============================================================
# 한 방향으로 끝까지 밀어보기
# ============================================================

def run_case(seed, action_value):
    """
    동일한 seed에서 시작하여
    지정된 SAC Action을 50 Step 동안 계속 적용한다.

    action_value
    -1.0 : 최대 전압 감소
     0.0 : HOLD
    +1.0 : 최대 전압 증가
    """

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )

    obs, info = env.reset(
        seed=seed
    )

    initial_voltage = env.output_voltage
    initial_current = env.output_current
    initial_tb = env.pipe_potential

    tb_values = [
        initial_tb
    ]

    voltage_values = [
        initial_voltage
    ]

    target_reached = (
        TARGET_MIN
        <= initial_tb
        <= TARGET_MAX
    )

    limit_hit_count = 0

    for _ in range(MAX_STEPS):

        action = np.array(
            [action_value],
            dtype=np.float32,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        tb_values.append(
            env.pipe_potential
        )

        voltage_values.append(
            env.output_voltage
        )

        if info.get(
            "model_limit_hit",
            False,
        ):
            limit_hit_count += 1

        if (
            TARGET_MIN
            <= env.pipe_potential
            <= TARGET_MAX
        ):
            target_reached = True

        if terminated or truncated:
            break

    result = {
        "initial_voltage":
            initial_voltage,

        "initial_current":
            initial_current,

        "initial_tb":
            initial_tb,

        "final_voltage":
            env.output_voltage,

        "final_tb":
            env.pipe_potential,

        "min_tb":
            min(tb_values),

        "max_tb":
            max(tb_values),

        "min_voltage":
            min(voltage_values),

        "max_voltage":
            max(voltage_values),

        "target_reached":
            target_reached,

        "limit_hit_count":
            limit_hit_count,
    }

    env.close()

    return result


# ============================================================
# Main Test
# ============================================================

def main():

    reachable_count = 0

    decrease_only_count = 0
    increase_only_count = 0
    both_count = 0
    impossible_count = 0

    results = []

    print(
        "\n===== Environment Reachability Test ====="
    )

    for seed in range(
        NUM_SAMPLES
    ):

        decrease = run_case(
            seed=seed,
            action_value=-1.0,
        )

        hold = run_case(
            seed=seed,
            action_value=0.0,
        )

        increase = run_case(
            seed=seed,
            action_value=+1.0,
        )

        decrease_reachable = (
            decrease[
                "target_reached"
            ]
        )

        increase_reachable = (
            increase[
                "target_reached"
            ]
        )

        hold_reachable = (
            hold[
                "target_reached"
            ]
        )

        # ----------------------------------------------------
        # 전체적으로 목표 도달 가능한지 판단
        # ----------------------------------------------------

        reachable = (
            decrease_reachable
            or hold_reachable
            or increase_reachable
        )

        if reachable:
            reachable_count += 1

        # ----------------------------------------------------
        # 방향별 분류
        # ----------------------------------------------------

        if (
            decrease_reachable
            and increase_reachable
        ):
            both_count += 1

        elif decrease_reachable:
            decrease_only_count += 1

        elif increase_reachable:
            increase_only_count += 1

        else:
            impossible_count += 1

        results.append(
            {
                "seed": seed,

                "initial_v":
                    decrease[
                        "initial_voltage"
                    ],

                "initial_tb":
                    decrease[
                        "initial_tb"
                    ],

                "decrease_final_tb":
                    decrease[
                        "final_tb"
                    ],

                "hold_final_tb":
                    hold[
                        "final_tb"
                    ],

                "increase_final_tb":
                    increase[
                        "final_tb"
                    ],

                "decrease_reached":
                    decrease_reachable,

                "hold_reached":
                    hold_reachable,

                "increase_reached":
                    increase_reachable,

                "decrease_limit":
                    decrease[
                        "limit_hit_count"
                    ],

                "increase_limit":
                    increase[
                        "limit_hit_count"
                    ],
            }
        )

    # ========================================================
    # 결과 요약
    # ========================================================

    print()

    print(
        f"테스트 Reset 수 : "
        f"{NUM_SAMPLES}"
    )

    print(
        f"목표 도달 가능   : "
        f"{reachable_count}"
    )

    print(
        f"목표 도달 불가능 : "
        f"{impossible_count}"
    )

    print()

    print(
        f"도달 가능 비율   : "
        f"{reachable_count / NUM_SAMPLES * 100:.1f}%"
    )

    print()

    print(
        f"감소 방향만 가능 : "
        f"{decrease_only_count}"
    )

    print(
        f"증가 방향만 가능 : "
        f"{increase_only_count}"
    )

    print(
        f"양쪽 모두 가능   : "
        f"{both_count}"
    )

    print()

    # ========================================================
    # 도달 불가능 사례 일부 출력
    # ========================================================

    impossible_results = [
        r
        for r in results
        if not (
            r[
                "decrease_reached"
            ]
            or r[
                "hold_reached"
            ]
            or r[
                "increase_reached"
            ]
        )
    ]

    print(
        "===== 도달 불가능 사례 일부 ====="
    )

    for r in impossible_results[
        :10
    ]:

        print(
            f"Seed {r['seed']:3d} | "
            f"초기 V={r['initial_v']:.3f} | "
            f"초기 TB={r['initial_tb']:.1f} | "
            f"DEC={r['decrease_final_tb']:.1f} | "
            f"HOLD={r['hold_final_tb']:.1f} | "
            f"INC={r['increase_final_tb']:.1f}"
        )

    print()

    print(
        "Environment Reachability Test 완료"
    )


if __name__ == "__main__":
    main()