import numpy as np
import pandas as pd

from env_model.preprocessing import (
    preprocess_environment_data,
)

from env_model.current_model import (
    CurrentModel,
)

from env_model.config.settings import (
    DATA_PATH,
)


# ============================================================
# 설정
# ============================================================

TEST_ACTIONS = [
    -0.20,
    -0.10,
    -0.05,
    0.05,
    0.10,
    0.20,
]


# ============================================================
# 데이터 불러오기
# ============================================================

_, current_df, _ = (
    preprocess_environment_data(
        DATA_PATH
    )
)

current_model = CurrentModel()


print(
    "\n===== Current Control Response 분석 ====="
)

print(
    f"Transition 수 : {len(current_df)}"
)


# ============================================================
# 1. 실제 데이터의 단순 Delta_V → Delta_I 관계
# ============================================================

delta_v = (
    current_df["Delta_V"]
    .to_numpy()
)

delta_i = (
    current_df["Delta_I"]
    .to_numpy()
)


# ------------------------------------------------------------
# Delta_I = slope * Delta_V + intercept
# ------------------------------------------------------------

slope, intercept = np.polyfit(
    delta_v,
    delta_i,
    1,
)


print()
print(
    "===== 실제 Transition 단순 선형관계 ====="
)

print(
    f"Slope     : {slope:+.6f} A/V"
)

print(
    f"Intercept : {intercept:+.6f} A"
)


# ============================================================
# 2. Current Model Action 반응을 Batch로 계산
# ============================================================

results = []


# 현재 상태 배열
V_values = (
    current_df[
        "Rectifier_Voltage"
    ]
    .to_numpy()
)

I_values = (
    current_df[
        "Rectifier_Current"
    ]
    .to_numpy()
)


# ============================================================
# Baseline 예측
#
# 모든 상태에 대해 Delta_V = 0을 한 번에 예측
# ============================================================

baseline_input = pd.DataFrame(
    {
        "Rectifier_Current": I_values,
        "Rectifier_Voltage": V_values,
        "Delta_V": np.zeros(
            len(current_df)
        ),
    }
)


baseline_predictions = (
    current_model.model.predict(
        baseline_input
    )
)


# ============================================================
# 각 Action별 반응 계산
# ============================================================

for action in TEST_ACTIONS:

    print(
        f"Action {action:+.2f} V 분석 중..."
    )


    # --------------------------------------------------------
    # 전체 4165개 상태에 같은 Action 적용
    # --------------------------------------------------------

    action_input = pd.DataFrame(
        {
            "Rectifier_Current": I_values,
            "Rectifier_Voltage": V_values,
            "Delta_V": np.full(
                len(current_df),
                action,
            ),
        }
    )


    # --------------------------------------------------------
    # 전체 Delta_I 예측
    # --------------------------------------------------------

    action_predictions = (
        current_model.model.predict(
            action_input
        )
    )


    # --------------------------------------------------------
    # Action에 의한 순수 효과
    #
    # Delta_I_action
    # =
    # 예측 Delta_I(action)
    # -
    # 예측 Delta_I(0)
    # --------------------------------------------------------

    delta_i_action = (
        action_predictions
        -
        baseline_predictions
    )


    # --------------------------------------------------------
    # 1V당 전류 변화량
    # --------------------------------------------------------

    action_slope = (
        delta_i_action
        / action
    )


    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    action_df = pd.DataFrame(
        {
            "V_t": V_values,
            "I_t": I_values,
            "Delta_V": action,
            "Delta_I_Action":
                delta_i_action,
            "Action_Slope":
                action_slope,
        }
    )


    results.append(
        action_df
    )


# ============================================================
# 모든 Action 결과 합치기
# ============================================================

response_df = pd.concat(
    results,
    ignore_index=True,
)


# ============================================================
# 3. Action별 평균 / 중앙값
# ============================================================

summary = (
    response_df
    .groupby("Delta_V")
    .agg(
        Mean_Delta_I=(
            "Delta_I_Action",
            "mean",
        ),

        Median_Delta_I=(
            "Delta_I_Action",
            "median",
        ),

        Mean_Slope=(
            "Action_Slope",
            "mean",
        ),

        Median_Slope=(
            "Action_Slope",
            "median",
        ),
    )
)


print()
print(
    "===== Action별 CurrentModel 반응 ====="
)

print(
    summary.to_string()
)


# ============================================================
# 4. Action Slope 전체 분포
# ============================================================

slopes = (
    response_df[
        "Action_Slope"
    ]
    .to_numpy()
)


print()
print(
    "===== Action Slope 전체 분포 ====="
)

print(
    f"Mean   : "
    f"{np.mean(slopes):+.6f} A/V"
)

print(
    f"Median : "
    f"{np.median(slopes):+.6f} A/V"
)

print(
    f"10%    : "
    f"{np.percentile(slopes, 10):+.6f} A/V"
)

print(
    f"25%    : "
    f"{np.percentile(slopes, 25):+.6f} A/V"
)

print(
    f"75%    : "
    f"{np.percentile(slopes, 75):+.6f} A/V"
)

print(
    f"90%    : "
    f"{np.percentile(slopes, 90):+.6f} A/V"
)


# ============================================================
# 5. 제어 방향성 확인
# ============================================================

same_direction = (
    np.sign(
        response_df["Delta_V"]
    )
    ==
    np.sign(
        response_df[
            "Delta_I_Action"
        ]
    )
)


direction_ratio = (
    same_direction.mean()
    * 100
)


print()
print(
    "===== 제어 방향성 ====="
)

print(
    f"기대 방향 일치율 : "
    f"{direction_ratio:.2f} %"
)


# ============================================================
# 6. V1 수식 후보
# ============================================================

candidate_k = float(
    np.median(slopes)
)


print()
print(
    "===== V1 후보 ====="
)

print(
    f"k_I 후보 : "
    f"{candidate_k:.6f} A/V"
)

print()
print(
    "수식 후보:"
)

print(
    f"Delta_I_action = "
    f"{candidate_k:.6f} * Delta_V"
)   

# ============================================================
# 7. 실제 데이터의 상승 / 하강 비대칭 분석
# ============================================================

print()
print(
    "===== 실제 데이터 상승 / 하강 비대칭 분석 ====="
)


# ------------------------------------------------------------
# 전압 상승 / 하강 데이터 분리
# ------------------------------------------------------------

up_df = current_df[
    current_df["Delta_V"] > 0
].copy()

down_df = current_df[
    current_df["Delta_V"] < 0
].copy()


print(
    f"전압 상승 Transition : {len(up_df)}"
)

print(
    f"전압 하강 Transition : {len(down_df)}"
)


# ============================================================
# 상승 구간 회귀
#
# Delta_I = k_up * Delta_V + intercept
# ============================================================

up_slope, up_intercept = np.polyfit(
    up_df["Delta_V"],
    up_df["Delta_I"],
    1,
)


# ============================================================
# 하강 구간 회귀
# ============================================================

down_slope, down_intercept = np.polyfit(
    down_df["Delta_V"],
    down_df["Delta_I"],
    1,
)


print()
print(
    "===== 상승 구간 ====="
)

print(
    f"Slope     : {up_slope:+.6f} A/V"
)

print(
    f"Intercept : {up_intercept:+.6f} A"
)


print()
print(
    "===== 하강 구간 ====="
)

print(
    f"Slope     : {down_slope:+.6f} A/V"
)

print(
    f"Intercept : {down_intercept:+.6f} A"
)


# ============================================================
# 실제 데이터의 방향 일치율
#
# 상승 : Delta_V > 0 이면서 Delta_I > 0
# 하강 : Delta_V < 0 이면서 Delta_I < 0
# ============================================================

up_direction_ratio = (
    (up_df["Delta_I"] > 0)
    .mean()
    * 100
)

down_direction_ratio = (
    (down_df["Delta_I"] < 0)
    .mean()
    * 100
)


print()
print(
    "===== 실제 데이터 방향 일치율 ====="
)

print(
    f"전압 상승 → 전류 상승 : "
    f"{up_direction_ratio:.2f} %"
)

print(
    f"전압 하강 → 전류 하강 : "
    f"{down_direction_ratio:.2f} %"
)


# ============================================================
# Delta_V 크기가 너무 작은 데이터 제외 후 재분석
#
# 아주 작은 전압 변화는 노이즈 영향을 크게 받을 수 있으므로
# |Delta_V| >= 0.05 V 데이터만 따로 확인한다.
# ============================================================

filtered_up_df = up_df[
    up_df["Delta_V"] >= 0.05
].copy()

filtered_down_df = down_df[
    down_df["Delta_V"] <= -0.05
].copy()


filtered_up_slope, filtered_up_intercept = np.polyfit(
    filtered_up_df["Delta_V"],
    filtered_up_df["Delta_I"],
    1,
)

filtered_down_slope, filtered_down_intercept = np.polyfit(
    filtered_down_df["Delta_V"],
    filtered_down_df["Delta_I"],
    1,
)


filtered_up_ratio = (
    (filtered_up_df["Delta_I"] > 0)
    .mean()
    * 100
)

filtered_down_ratio = (
    (filtered_down_df["Delta_I"] < 0)
    .mean()
    * 100
)


print()
print(
    "===== |Delta_V| >= 0.05 V만 사용 ====="
)

print(
    f"상승 데이터 수 : "
    f"{len(filtered_up_df)}"
)

print(
    f"하강 데이터 수 : "
    f"{len(filtered_down_df)}"
)


print()
print(
    "--- 상승 ---"
)

print(
    f"Slope     : "
    f"{filtered_up_slope:+.6f} A/V"
)

print(
    f"Intercept : "
    f"{filtered_up_intercept:+.6f} A"
)

print(
    f"방향 일치 : "
    f"{filtered_up_ratio:.2f} %"
)


print()
print(
    "--- 하강 ---"
)

print(
    f"Slope     : "
    f"{filtered_down_slope:+.6f} A/V"
)

print(
    f"Intercept : "
    f"{filtered_down_intercept:+.6f} A"
)

print(
    f"방향 일치 : "
    f"{filtered_down_ratio:.2f} %"
)


# ============================================================
# 상승 / 하강 평균 Delta_I 확인
# ============================================================

print()
print(
    "===== 실제 평균 변화량 ====="
)

print(
    f"상승 평균 Delta_V : "
    f"{up_df['Delta_V'].mean():+.6f} V"
)

print(
    f"상승 평균 Delta_I : "
    f"{up_df['Delta_I'].mean():+.6f} A"
)

print()

print(
    f"하강 평균 Delta_V : "
    f"{down_df['Delta_V'].mean():+.6f} V"
)

print(
    f"하강 평균 Delta_I : "
    f"{down_df['Delta_I'].mean():+.6f} A"
)

# ============================================================
# 8. Delta_V 구간별 실제 Delta_I 평균 분석
# ============================================================
#
# 목적:
# 개별 Transition은 자연변동 / 노이즈가 크기 때문에
# Delta_V 크기가 비슷한 데이터끼리 묶어서
# 평균적인 Delta_I 반응을 확인한다.
#
# 예)
# Delta_V 약 +0.10V인 데이터들을 모았을 때
# 평균 Delta_I가 실제로 + 방향인지 확인
# ============================================================

print()
print(
    "===== Delta_V 구간별 실제 Delta_I 평균 ====="
)


# ------------------------------------------------------------
# Delta_V 구간 설정
#
# -0.225 ~ +0.225 V를
# 0.05V 간격으로 나눈다.
#
# 예:
# -0.125 ~ -0.075 → 중심값 약 -0.10V
# +0.075 ~ +0.125 → 중심값 약 +0.10V
# ------------------------------------------------------------

bin_edges = np.arange(
    -0.225,
    0.226,
    0.05,
)


current_df["Delta_V_Bin"] = pd.cut(
    current_df["Delta_V"],
    bins=bin_edges,
    include_lowest=True,
)


# ============================================================
# 각 구간별 통계
# ============================================================

bin_summary = (
    current_df
    .groupby(
        "Delta_V_Bin",
        observed=True,
    )
    .agg(
        Count=(
            "Delta_I",
            "count",
        ),

        Mean_Delta_V=(
            "Delta_V",
            "mean",
        ),

        Mean_Delta_I=(
            "Delta_I",
            "mean",
        ),

        Median_Delta_I=(
            "Delta_I",
            "median",
        ),

        Std_Delta_I=(
            "Delta_I",
            "std",
        ),
    )
    .reset_index()
)


# ============================================================
# 각 구간의 평균 Delta_V를 기준으로
# 실제 평균 변화 비율 계산
#
# k_bin = Mean Delta_I / Mean Delta_V
#
# Delta_V가 거의 0인 구간은 제외
# ============================================================

bin_summary["K_Bin"] = np.where(
    np.abs(
        bin_summary["Mean_Delta_V"]
    ) >= 0.01,

    bin_summary["Mean_Delta_I"]
    /
    bin_summary["Mean_Delta_V"],

    np.nan,
)


print()

print(
    bin_summary[
        [
            "Delta_V_Bin",
            "Count",
            "Mean_Delta_V",
            "Mean_Delta_I",
            "Median_Delta_I",
            "Std_Delta_I",
            "K_Bin",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 9. 충분한 데이터가 있는 구간만 따로 확인
# ============================================================
#
# 데이터 수가 너무 적은 구간은 평균값이
# 몇 개의 이상치에 크게 영향을 받을 수 있다.
#
# 여기서는 임시로 100개 이상인 구간만 확인한다.
# ============================================================

reliable_bins = bin_summary[
    bin_summary["Count"] >= 100
].copy()


print()
print(
    "===== 데이터 100개 이상 구간 ====="
)

print(
    reliable_bins[
        [
            "Count",
            "Mean_Delta_V",
            "Mean_Delta_I",
            "Median_Delta_I",
            "K_Bin",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 10. 구간 평균값을 이용한 선형 회귀
# ============================================================
#
# 개별 4165개 Transition에 바로 회귀하는 대신,
# Delta_V 구간별 평균점을 이용해서
#
# Delta_I = k * Delta_V + intercept
#
# 관계를 다시 확인한다.
# ============================================================

regression_bins = reliable_bins[
    np.abs(
        reliable_bins["Mean_Delta_V"]
    ) >= 0.01
].copy()


if len(regression_bins) >= 2:

    bin_slope, bin_intercept = np.polyfit(
        regression_bins["Mean_Delta_V"],
        regression_bins["Mean_Delta_I"],
        1,
    )


    print()
    print(
        "===== 구간 평균 기반 선형관계 ====="
    )

    print(
        f"Slope     : "
        f"{bin_slope:+.6f} A/V"
    )

    print(
        f"Intercept : "
        f"{bin_intercept:+.6f} A"
    )

    print()

    print(
        "수식 형태:"
    )

    print(
        f"Delta_I ≈ "
        f"{bin_slope:.6f} * Delta_V "
        f"{bin_intercept:+.6f}"
    )