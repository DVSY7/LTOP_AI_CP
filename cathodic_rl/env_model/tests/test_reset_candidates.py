"""수동 진단: Gymnasium reset에 쓸 7개 TB History 후보를 확인한다."""

from cathodic_rl.env_model.config.settings import DATA_PATH

from cathodic_rl.env_model.preprocessing import (
    preprocess_environment_data,
    create_reset_candidates,
)


# ============================================================
# 기본 전처리
# ============================================================

df, _, _ = (
    preprocess_environment_data(
        DATA_PATH
    )
)


# ============================================================
# Reset 후보 데이터 생성
# ============================================================

reset_df = (
    create_reset_candidates(
        df
    )
)


print(
    "\n===== Gym Reset 후보 데이터 ====="
)

print(
    f"전체 후보 수 : {len(reset_df)}"
)


# ============================================================
# 앞부분 확인
# ============================================================

columns_to_show = [
    "DateTime",
    "Segment",
    "Rectifier_Voltage",
    "Rectifier_Current",
    "TB_Lag6",
    "TB_Lag5",
    "TB_Lag4",
    "TB_Lag3",
    "TB_Lag2",
    "TB_Lag1",
    "TB1-Volt",
]


print()

print(
    reset_df[
        columns_to_show
    ].head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# 기본 검사
# ============================================================

assert len(reset_df) > 0

assert reset_df[
    columns_to_show
].isna().sum().sum() == 0


# ============================================================
# TB History 예시 확인
# ============================================================

sample = reset_df.iloc[0]

tb_history = [
    float(sample["TB_Lag6"]),
    float(sample["TB_Lag5"]),
    float(sample["TB_Lag4"]),
    float(sample["TB_Lag3"]),
    float(sample["TB_Lag2"]),
    float(sample["TB_Lag1"]),
    float(sample["TB1-Volt"]),
]


print(
    "\n===== 첫 번째 Reset 후보 ====="
)

print(
    f"DateTime : "
    f"{sample['DateTime']}"
)

print(
    f"Segment  : "
    f"{sample['Segment']}"
)

print(
    f"Voltage  : "
    f"{sample['Rectifier_Voltage']:.4f} V"
)

print(
    f"Current  : "
    f"{sample['Rectifier_Current']:.4f} A"
)

print(
    "TB History :"
)

print(
    tb_history
)


assert len(tb_history) == 7

assert (
    tb_history[-1]
    ==
    float(sample["TB1-Volt"])
)


print(
    "\nReset 후보 데이터 테스트 통과"
)
