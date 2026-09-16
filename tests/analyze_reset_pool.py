from env_model.preprocessing import (
    preprocess_environment_data,
    create_reset_candidates,
)

from env_model.config.settings import DATA_PATH


# ============================================================
# V1 목표 범위
# ============================================================

TARGET_MIN = -1610.0
TARGET_MAX = -1590.0


# ============================================================
# Reset Candidate 분석
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. 실데이터 전처리
    # --------------------------------------------------------

    df, _, _ = preprocess_environment_data(
        DATA_PATH
    )

    reset_df = create_reset_candidates(
        df
    )

    print(
        "\n===== Reset Candidate 분석 ====="
    )

    print(
        f"전체 Reset 후보 : "
        f"{len(reset_df)}"
    )


    # --------------------------------------------------------
    # 2. 목표 기준으로 3개 그룹 분리
    # --------------------------------------------------------

    # 너무 음수
    # 예: -1650mV
    below_target = reset_df[
        reset_df["TB1-Volt"]
        < TARGET_MIN
    ].copy()

    # 목표 범위
    target_zone = reset_df[
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

    # 덜 음수
    # 예: -1570mV
    above_target = reset_df[
        reset_df["TB1-Volt"]
        > TARGET_MAX
    ].copy()


    # --------------------------------------------------------
    # 3. 그룹별 기본 통계
    # --------------------------------------------------------

    print()

    print(
        "===== 목표 기준 Reset 분포 ====="
    )

    print(
        f"목표보다 너무 음수 "
        f"(TB < {TARGET_MIN:.0f}) : "
        f"{len(below_target)}"
    )

    print(
        f"목표 범위 "
        f"({TARGET_MIN:.0f} ~ {TARGET_MAX:.0f}) : "
        f"{len(target_zone)}"
    )

    print(
        f"목표보다 덜 음수 "
        f"(TB > {TARGET_MAX:.0f}) : "
        f"{len(above_target)}"
    )


    # --------------------------------------------------------
    # 4. 비율 출력
    # --------------------------------------------------------

    total = len(reset_df)

    print()

    print(
        "===== 비율 ====="
    )

    print(
        f"너무 음수 : "
        f"{len(below_target) / total * 100:.1f}%"
    )

    print(
        f"목표 범위 : "
        f"{len(target_zone) / total * 100:.1f}%"
    )

    print(
        f"덜 음수   : "
        f"{len(above_target) / total * 100:.1f}%"
    )


    # --------------------------------------------------------
    # 5. 각 그룹의 TB / Voltage 범위 확인
    # --------------------------------------------------------

    def print_group_stats(
        name,
        group,
    ):

        print()
        print(
            f"===== {name} ====="
        )

        if len(group) == 0:

            print(
                "해당 Reset 후보 없음"
            )

            return

        print(
            f"개수 : {len(group)}"
        )

        print(
            f"TB 범위 : "
            f"{group['TB1-Volt'].min():.1f} "
            f"~ "
            f"{group['TB1-Volt'].max():.1f} mV"
        )

        print(
            f"TB 평균 : "
            f"{group['TB1-Volt'].mean():.1f} mV"
        )

        print(
            f"Voltage 범위 : "
            f"{group['Rectifier_Voltage'].min():.3f} "
            f"~ "
            f"{group['Rectifier_Voltage'].max():.3f} V"
        )

        print(
            f"Voltage 평균 : "
            f"{group['Rectifier_Voltage'].mean():.3f} V"
        )


    print_group_stats(
        "목표보다 너무 음수",
        below_target,
    )

    print_group_stats(
        "목표 범위",
        target_zone,
    )

    print_group_stats(
        "목표보다 덜 음수",
        above_target,
    )


    # --------------------------------------------------------
    # 6. 목표보다 덜 음수인 데이터 일부 확인
    #
    # 이전 Reachability 결과에서 이쪽 데이터가
    # 상대적으로 적어 보였기 때문에 실제 후보를 확인한다.
    # --------------------------------------------------------

    print()

    print(
        "===== 목표보다 덜 음수인 Reset 예시 ====="
    )

    if len(above_target) > 0:

        example = above_target[
            [
                "DateTime",
                "Rectifier_Voltage",
                "Rectifier_Current",
                "TB1-Volt",
            ]
        ].head(10)

        print(
            example.to_string(
                index=False
            )
        )


    print()

    print(
        "Reset Candidate 분석 완료"
    )


if __name__ == "__main__":
    main()