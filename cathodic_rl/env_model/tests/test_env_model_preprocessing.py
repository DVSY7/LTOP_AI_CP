"""수동 진단: 원본 데이터가 Current/TB 모델 학습용 표로 변환되는지 확인한다."""

from pathlib import Path

from cathodic_rl.env_model.tests import _path_setup

from cathodic_rl.env_model.config.settings import(
    DATA_PATH
)
from cathodic_rl.env_model.preprocessing import (
    preprocess_environment_data
)



df, current_df, tb_df = (
    preprocess_environment_data(
        DATA_PATH
    )
)


print("\n===== 기본 데이터 =====")
print(df.shape)

print("\n===== Current Model 데이터 =====")
print(current_df.shape)

print(
    current_df[
        [
            "DateTime",
            "Rectifier_Voltage",
            "Rectifier_Current",
            "Delta_V",
            "Delta_I"
        ]
    ].head()
)


print("\n===== TB History Model 데이터 =====")
print(tb_df.shape)

print(
    tb_df[
        [
            "DateTime",
            "TB_Lag6",
            "TB_Lag3",
            "TB_Lag2",
            "TB_Lag1",
            "TB1-Volt",
            "Next_TB"
        ]
    ].head()
)
