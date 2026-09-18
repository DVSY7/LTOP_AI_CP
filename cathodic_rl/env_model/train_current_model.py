from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from env_model.preprocessing import (
    preprocess_environment_data,
)


# ============================================================
# 경로 설정
# ============================================================

# 현재 파일:
# cathodic_rl/env_model/train_current_model.py

from env_model.config.settings import(
    DATA_PATH,
    SAVE_DIR,
    CURRENT_MODEL_PATH
)



# ============================================================
# Current Model 3 설정
# ============================================================

# 입력값
#
# Rectifier_Current
#     현재 시점의 추정 정류기 총 전류
#
# Rectifier_Voltage
#     현재 시점의 추정 정류기 전압
#
# Delta_V
#     현재 시점에서 다음 시점까지의 전압 변화량
#
CURRENT_FEATURES = [
    "Rectifier_Current",
    "Rectifier_Voltage",
    "Delta_V",
]

# 예측 대상
#
# 다음 시점까지의 총 전류 변화량
CURRENT_TARGET = "Delta_I"


# ============================================================
# 데이터 시간순 분할
# ============================================================

def split_chronologically(
    df,
    train_ratio=0.8,
):
    """
    시계열 데이터이므로 데이터를 섞지 않고
    시간순으로 Train / Test 데이터를 나눈다.
    """

    split_index = int(
        len(df) * train_ratio
    )

    train_df = (
        df
        .iloc[:split_index]
        .copy()
    )

    test_df = (
        df
        .iloc[split_index:]
        .copy()
    )

    return train_df, test_df


# ============================================================
# Current Model 생성
# ============================================================

def create_current_model():
    """
    Current Model 3에서 사용했던
    RandomForest 모델 설정을 그대로 사용한다.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    return model


# ============================================================
# 모델 학습 및 평가
# ============================================================

def train_current_model():
    """
    Current Model 3을 학습하고
    Test 데이터에서 성능을 확인한 뒤
    학습된 모델을 저장한다.
    """

    # --------------------------------------------------------
    # 1. 전처리 데이터 생성
    # --------------------------------------------------------

    _, current_df, _ = (
        preprocess_environment_data(
            DATA_PATH
        )
    )

    print(
        "\n===== Current Model 데이터 ====="
    )

    print(
        f"전체 데이터 수 : "
        f"{len(current_df)}"
    )


    # --------------------------------------------------------
    # 2. 시간순 Train / Test 분리
    # --------------------------------------------------------

    train_df, test_df = (
        split_chronologically(
            current_df,
            train_ratio=0.8,
        )
    )

    print(
        f"Train 데이터 수 : "
        f"{len(train_df)}"
    )

    print(
        f"Test 데이터 수  : "
        f"{len(test_df)}"
    )


    # --------------------------------------------------------
    # 3. 학습용 X, y 생성
    # --------------------------------------------------------

    X_train = train_df[
        CURRENT_FEATURES
    ]

    y_train = train_df[
        CURRENT_TARGET
    ]

    X_test = test_df[
        CURRENT_FEATURES
    ]

    y_test = test_df[
        CURRENT_TARGET
    ]


    # --------------------------------------------------------
    # 4. 모델 생성
    # --------------------------------------------------------

    model = (
        create_current_model()
    )


    # --------------------------------------------------------
    # 5. 모델 학습
    # --------------------------------------------------------

    print(
        "\n===== Current Model 학습 시작 ====="
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "학습 완료"
    )


    # --------------------------------------------------------
    # 6. Test 데이터 예측
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # 7. 성능 평가
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )

    print(
        "\n===== Current Model 성능 ====="
    )

    print(
        f"MAE  : {mae:.4f} A"
    )

    print(
        f"RMSE : {rmse:.4f} A"
    )


    # --------------------------------------------------------
    # 8. 모델 저장 폴더 생성
    # --------------------------------------------------------

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # 9. 학습된 모델 저장
    # --------------------------------------------------------

    joblib.dump(
        model,
        CURRENT_MODEL_PATH
    )

    print(
        "\n===== 모델 저장 완료 ====="
    )

    print(
        CURRENT_MODEL_PATH
    )


    # --------------------------------------------------------
    # 10. 나중에 다른 코드에서도 사용할 수 있도록
    #     학습된 모델 반환
    # --------------------------------------------------------

    return model


# ============================================================
# 직접 실행했을 때만 학습 수행
# ============================================================

if __name__ == "__main__":
    train_current_model()