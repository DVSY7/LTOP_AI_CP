import numpy as np

from cathodic_rl.env.cathodic_env import CathodicProtectionEnv


# ============================================================
# V1 목표 범위
# ============================================================

TARGET_MIN = -1610.0
TARGET_MAX = -1590.0

# 각 그룹에서 검사할 Reset 개수
SAMPLE_SIZE = 100

# 한 Reset에서 최대 제어 Step
MAX_STEPS = 50

# 결과 재현용 난수
RANDOM_SEED = 42


# ============================================================
# 현재 Env를 특정 Reset Candidate로 직접 초기화
# ============================================================

def reset_to_candidate(env, row):
    """
    env.reset()의 랜덤 선택을 사용하지 않고
    지정한 실제 Reset Candidate로 직접 초기화한다.
    """

    # 현재 V / I / TB
    env.output_voltage = float(
        row["Rectifier_Voltage"]
    )

    env.output_current = float(
        row["Rectifier_Current"]
    )

    env.pipe_potential = float(
        row["TB1-Volt"]
    )

    # TB History
    initial_tb_history = [
        float(row["TB_Lag6"]),
        float(row["TB_Lag5"]),
        float(row["TB_Lag4"]),
        float(row["TB_Lag3"]),
        float(row["TB_Lag2"]),
        float(row["TB_Lag1"]),
        float(row["TB1-Volt"]),
    ]

    env.environment_model.reset_history(
        initial_tb_history
    )

    # Reset 정보
    env.reset_datetime = row["DateTime"]
    env.reset_segment = int(
        row["Segment"]
    )

    env.current_step = 0


# ============================================================
# 하나의 Reset Candidate에서 Reachability 검사
# ============================================================

def check_reachability(
    env,
    row,
    action_value,
):

    reset_to_candidate(
        env,
        row,
    )

    initial_v = env.output_voltage
    initial_tb = env.pipe_potential

    reached = (
        TARGET_MIN
        <= initial_tb
        <= TARGET_MAX
    )

    reached_step = 0 if reached else None

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        action = np.array(
            [action_value],
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        current_tb = (
            env.pipe_potential
        )

        if (
            TARGET_MIN
            <= current_tb
            <= TARGET_MAX
        ):
            reached = True
            reached_step = step
            break

        if terminated or truncated:
            break

    return {
        "initial_v": initial_v,
        "initial_tb": initial_tb,
        "final_v": env.output_voltage,
        "final_tb": env.pipe_potential,
        "reached": reached,
        "reached_step": reached_step,
    }


# ============================================================
# 메인
# ============================================================

def main():

    print()
    print(
        "===== 경량 Reset Reachability 분석 ====="
    )

    print()
    print(
        "환경 초기화 중..."
    )

    # SAC용 연속 행동 환경
    #
    # Env는 한 번만 생성하고 계속 재사용한다.
    env = CathodicProtectionEnv(
        action_mode="continuous"
    )

    reset_df = (
        env.reset_candidates
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"전체 Reset 후보 : "
        f"{len(reset_df)}"
    )


    # ========================================================
    # 1. 목표 기준 그룹 분리
    # ========================================================

    below_df = reset_df[
        reset_df["TB1-Volt"]
        < TARGET_MIN
    ].copy()

    target_df = reset_df[
        (
            reset_df["TB1-Volt"]
            >= TARGET_MIN
        )
        &
        (
            reset_df["TB1-Volt"]
            <= TARGET_MAX
        )
    ].copy()

    above_df = reset_df[
        reset_df["TB1-Volt"]
        > TARGET_MAX
    ].copy()


    print()
    print(
        "===== 전체 Reset 분포 ====="
    )

    print(
        f"너무 음수 : "
        f"{len(below_df)}"
    )

    print(
        f"목표 범위 : "
        f"{len(target_df)}"
    )

    print(
        f"덜 음수   : "
        f"{len(above_df)}"
    )


    # ========================================================
    # 2. 각 그룹에서 100개 랜덤 Sampling
    # ========================================================

    below_sample = (
        below_df.sample(
            n=min(
                SAMPLE_SIZE,
                len(below_df),
            ),
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    above_sample = (
        above_df.sample(
            n=min(
                SAMPLE_SIZE,
                len(above_df),
            ),
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # 3. 너무 음수 그룹
    #
    # 목표보다 너무 음수이므로
    # 전압 감소(-1.0) 방향만 검사
    # ========================================================

    print()
    print(
        "===== 너무 음수 그룹 검사 ====="
    )

    below_reachable = 0
    below_results = []

    for i, (_, row) in enumerate(
        below_sample.iterrows(),
        start=1,
    ):

        result = check_reachability(
            env=env,
            row=row,
            action_value=-1.0,
        )

        below_results.append(
            result
        )

        if result["reached"]:
            below_reachable += 1

        if i % 20 == 0:
            print(
                f"진행 : "
                f"{i}/"
                f"{len(below_sample)}"
            )


    # ========================================================
    # 4. 덜 음수 그룹
    #
    # 목표보다 덜 음수이므로
    # 전압 증가(+1.0) 방향만 검사
    # ========================================================

    print()
    print(
        "===== 덜 음수 그룹 검사 ====="
    )

    above_reachable = 0
    above_results = []

    for i, (_, row) in enumerate(
        above_sample.iterrows(),
        start=1,
    ):

        result = check_reachability(
            env=env,
            row=row,
            action_value=+1.0,
        )

        above_results.append(
            result
        )

        if result["reached"]:
            above_reachable += 1

        if i % 20 == 0:
            print(
                f"진행 : "
                f"{i}/"
                f"{len(above_sample)}"
            )


    # ========================================================
    # 5. 결과 출력
    # ========================================================

    below_ratio = (
        below_reachable
        / len(below_sample)
        * 100
    )

    above_ratio = (
        above_reachable
        / len(above_sample)
        * 100
    )


    print()
    print(
        "===== Reachability 결과 ====="
    )

    print()

    print(
        "[목표보다 너무 음수]"
    )

    print(
        f"검사 수    : "
        f"{len(below_sample)}"
    )

    print(
        f"도달 가능  : "
        f"{below_reachable}"
    )

    print(
        f"도달 불가능: "
        f"{len(below_sample) - below_reachable}"
    )

    print(
        f"도달 비율  : "
        f"{below_ratio:.1f}%"
    )


    print()

    print(
        "[목표 범위]"
    )

    print(
        f"전체 후보  : "
        f"{len(target_df)}"
    )

    print(
        "시작부터 목표 범위이므로 "
        "Reachable = 100%"
    )


    print()

    print(
        "[목표보다 덜 음수]"
    )

    print(
        f"검사 수    : "
        f"{len(above_sample)}"
    )

    print(
        f"도달 가능  : "
        f"{above_reachable}"
    )

    print(
        f"도달 불가능: "
        f"{len(above_sample) - above_reachable}"
    )

    print(
        f"도달 비율  : "
        f"{above_ratio:.1f}%"
    )


    # ========================================================
    # 6. 실패 사례 일부 확인
    # ========================================================

    print()
    print(
        "===== 너무 음수 그룹 실패 예시 ====="
    )

    failed_below = [
        result
        for result in below_results
        if not result["reached"]
    ]

    for result in failed_below[:5]:

        print(
            f"초기 V="
            f"{result['initial_v']:.3f}"
            f" | 초기 TB="
            f"{result['initial_tb']:.1f}"
            f" | 최종 V="
            f"{result['final_v']:.3f}"
            f" | 최종 TB="
            f"{result['final_tb']:.1f}"
        )


    print()
    print(
        "===== 덜 음수 그룹 실패 예시 ====="
    )

    failed_above = [
        result
        for result in above_results
        if not result["reached"]
    ]

    for result in failed_above[:5]:

        print(
            f"초기 V="
            f"{result['initial_v']:.3f}"
            f" | 초기 TB="
            f"{result['initial_tb']:.1f}"
            f" | 최종 V="
            f"{result['final_v']:.3f}"
            f" | 최종 TB="
            f"{result['final_tb']:.1f}"
        )


    env.close()

    print()
    print(
        "경량 Reset Reachability 분석 완료"
    )


if __name__ == "__main__":
    main()