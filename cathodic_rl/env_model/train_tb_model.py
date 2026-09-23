import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from cathodic_rl.env_model.preprocessing import (
    preprocess_environment_data,
)

from cathodic_rl.env_model.config.settings import (
    DATA_PATH,
    SAVE_DIR,
    TB_MODEL_PATH,
)


# ============================================================
# TB History Model 설정
# ============================================================

# TB 자연 변화 예측에 사용할 입력값
TB_FEATURES = [
    "TB1-Volt",
    "TB_Lag1",
    "TB_Lag2",
    "TB_Lag3",
    "TB_Lag6",
    "TB_Change_10m",
    "TB_Change_30m",
]

# 다음 시점(10분 뒤)의 TB 전위
TB_TARGET = "Next_TB"


# ============================================================
# 데이터 시간순 분할
# ============================================================

def split_chronologically(
    df,
    train_ratio=0.8,
):
    """
    시계열 데이터이므로 랜덤하게 섞지 않고
    시간 순서대로 Train / Test를 나눈다.
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
# TB History Model 생성
# ============================================================

def create_tb_model():
    """
    이전 노트북에서 사용했던
    TB History RandomForest 모델을 생성한다.
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
# Persistence 성능 계산
# ============================================================

def evaluate_persistence(
    test_df,
):
    """
    가장 단순한 기준 모델.

    현재 TB가 다음 10분에도 그대로 유지된다고
    가정했을 때의 예측 성능을 계산한다.

    TB_(t+1) = TB_t
    """

    y_true = test_df[
        TB_TARGET
    ]

    y_persistence = test_df[
        "TB1-Volt"
    ]

    mae = mean_absolute_error(
        y_true,
        y_persistence,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_persistence,
        )
    )

    return mae, rmse


# ============================================================
# TB History Model 학습
# ============================================================

def train_tb_model():

    # --------------------------------------------------------
    # 1. 전처리
    # --------------------------------------------------------

    _, _, tb_df = (
        preprocess_environment_data(
            DATA_PATH
        )
    )

    print(
        "\n===== TB History Model 데이터 ====="
    )

    print(
        f"전체 데이터 수 : "
        f"{len(tb_df)}"
    )


    # --------------------------------------------------------
    # 2. 시간순 Train / Test 분리
    # --------------------------------------------------------

    train_df, test_df = (
        split_chronologically(
            tb_df,
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
    # 3. 입력 X / 정답 y 생성
    # --------------------------------------------------------

    X_train = train_df[
        TB_FEATURES
    ]

    y_train = train_df[
        TB_TARGET
    ]

    X_test = test_df[
        TB_FEATURES
    ]

    y_test = test_df[
        TB_TARGET
    ]


    # --------------------------------------------------------
    # 4. Persistence 기준 성능
    # --------------------------------------------------------

    persistence_mae, persistence_rmse = (
        evaluate_persistence(
            test_df
        )
    )

    print(
        "\n===== Persistence 기준 성능 ====="
    )

    print(
        f"MAE  : "
        f"{persistence_mae:.4f} mV"
    )

    print(
        f"RMSE : "
        f"{persistence_rmse:.4f} mV"
    )


    # --------------------------------------------------------
    # 5. TB History Model 생성
    # --------------------------------------------------------

    model = (
        create_tb_model()
    )


    # --------------------------------------------------------
    # 6. 모델 학습
    # --------------------------------------------------------

    print(
        "\n===== TB History Model 학습 시작 ====="
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "학습 완료"
    )


    # --------------------------------------------------------
    # 7. Test 데이터 예측
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # 8. 모델 성능 평가
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
        "\n===== TB History Model 성능 ====="
    )

    print(
        f"MAE  : {mae:.4f} mV"
    )

    print(
        f"RMSE : {rmse:.4f} mV"
    )


    # --------------------------------------------------------
    # 9. Persistence 대비 개선율
    # --------------------------------------------------------

    improvement = (
        (
            persistence_mae
            - mae
        )
        / persistence_mae
        * 100
    )

    print(
        f"Persistence 대비 MAE 개선율 : "
        f"{improvement:.2f} %"
    )


    # --------------------------------------------------------
    # 10. 모델 저장 폴더 생성
    # --------------------------------------------------------

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # 11. 모델 저장
    # --------------------------------------------------------

    joblib.dump(
        model,
        TB_MODEL_PATH,
    )

    print(
        "\n===== TB History Model 저장 완료 ====="
    )

    print(
        TB_MODEL_PATH
    )


    return model


# ============================================================
# 직접 실행
# ============================================================

if __name__ == "__main__":
    train_tb_model()
