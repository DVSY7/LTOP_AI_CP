import pandas as pd


# ============================================================
# 환경모델 데이터 전처리
#
# 목적
# ------------------------------------------------------------
# 원본 JB 데이터에서
#
# 1. 가상 정류기 전압 생성
# 2. 가상 정류기 전류 생성
# 3. 연속된 10분 데이터 구간 생성
# 4. Current Model 3 학습 데이터 생성
# 5. TB History Model 학습 데이터 생성
#
# 분석/그래프/모델 비교 코드는 이 파일에 포함하지 않는다.
# ============================================================


# ------------------------------------------------------------
# 1. 원본 데이터 불러오기
# ------------------------------------------------------------

def load_raw_data(file_path):
    """
    환경모델 학습용 Excel 데이터를 불러온다.
    """

    df = pd.read_excel(file_path)

    # DateTime 형식 변환
    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        errors="coerce"
    )

    # DateTime 없는 행 제거
    df = df.dropna(
        subset=["DateTime"]
    ).copy()

    # 시간 순으로 정렬
    df = df.sort_values(
        "DateTime"
    ).reset_index(drop=True)

    return df


# ------------------------------------------------------------
# 2. 가상 정류기 전압 / 전류 생성
# ------------------------------------------------------------

def create_rectifier_features(df):
    """
    12개 JB 채널을 이용해
    가상 정류기 전압/전류를 생성한다.

    Rectifier_Voltage
        = 12개 OutVolt 평균

    Rectifier_Current
        = 12개 OutCurr 합계
    """

    df = df.copy()

    voltage_columns = [
        f"{i}-OutVolt"
        for i in range(1, 13)
    ]

    current_columns = [
        f"{i}-OutCurr"
        for i in range(1, 13)
    ]

    # 정류기 출력전압 Proxy
    df["Rectifier_Voltage"] = (
        df[voltage_columns].mean(axis=1)
    )

    # 정류기 총 출력전류 Proxy
    df["Rectifier_Current"] = (
        df[current_columns].sum(axis=1)
    )

    return df


# ------------------------------------------------------------
# 3. 연속 10분 Segment 생성
# ------------------------------------------------------------

def create_segments(df):
    """
    연속적인 약 10분 간격 데이터끼리
    같은 Segment 번호를 부여한다.

    정상 범위:
        9분 30초 ~ 10분 30초

    데이터 누락 구간을 넘어
    Lag / Next 값이 연결되는 것을 방지한다.
    """

    df = df.copy()

    df["Prev_Time_Diff"] = (
        df["DateTime"]
        - df["DateTime"].shift(1)
    )

    df["Is_Continuous"] = (
        (
            df["Prev_Time_Diff"]
            >= pd.Timedelta(
                minutes=9,
                seconds=30
            )
        )
        &
        (
            df["Prev_Time_Diff"]
            <= pd.Timedelta(
                minutes=10,
                seconds=30
            )
        )
    )

    # 연속성이 끊길 때마다 새로운 Segment
    df["Segment"] = (
        ~df["Is_Continuous"]
    ).cumsum()

    return df


# ------------------------------------------------------------
# 4. Current Model 3 학습 데이터 생성
# ------------------------------------------------------------

def create_current_model_data(df):
    """
    Current Model 3 학습용 데이터 생성

    입력:
        Rectifier_Current(t)
        Rectifier_Voltage(t)
        Delta_V

    Target:
        Delta_I
    """

    current_df = df.copy()

    # 같은 Segment 안에서만 다음 값 생성
    current_df["Next_Voltage"] = (
        current_df
        .groupby("Segment")[
            "Rectifier_Voltage"
        ]
        .shift(-1)
    )

    current_df["Next_Current"] = (
        current_df
        .groupby("Segment")[
            "Rectifier_Current"
        ]
        .shift(-1)
    )

    # Action 역할
    current_df["Delta_V"] = (
        current_df["Next_Voltage"]
        - current_df["Rectifier_Voltage"]
    )

    # Target
    current_df["Delta_I"] = (
        current_df["Next_Current"]
        - current_df["Rectifier_Current"]
    )

    current_df = current_df.dropna(
        subset=[
            "Next_Voltage",
            "Next_Current",
            "Delta_V",
            "Delta_I"
        ]
    ).copy()

    return current_df


# ------------------------------------------------------------
# 5. TB History Model 학습 데이터 생성
# ------------------------------------------------------------

def create_tb_history_data(df):
    """
    TB History Model 학습용 데이터 생성

    현재 TB와 과거 TB History를 이용해
    다음 TB를 예측하기 위한 데이터 생성
    """

    tb_df = df.copy()

    grouped_tb = (
        tb_df
        .groupby("Segment")[
            "TB1-Volt"
        ]
    )

    # 과거 TB
    tb_df["TB_Lag1"] = (
        grouped_tb.shift(1)
    )

    tb_df["TB_Lag2"] = (
        grouped_tb.shift(2)
    )

    tb_df["TB_Lag3"] = (
        grouped_tb.shift(3)
    )

    tb_df["TB_Lag6"] = (
        grouped_tb.shift(6)
    )

    # 다음 TB
    tb_df["Next_TB"] = (
        grouped_tb.shift(-1)
    )

    # 변화량
    tb_df["TB_Change_10m"] = (
        tb_df["TB1-Volt"]
        - tb_df["TB_Lag1"]
    )

    tb_df["TB_Change_30m"] = (
        tb_df["TB1-Volt"]
        - tb_df["TB_Lag3"]
    )

    tb_df = tb_df.dropna(
        subset=[
            "TB1-Volt",
            "TB_Lag1",
            "TB_Lag2",
            "TB_Lag3",
            "TB_Lag6",
            "TB_Change_10m",
            "TB_Change_30m",
            "Next_TB"
        ]
    ).copy()

    return tb_df

# ------------------------------------------------------------
# 6. Gym Reset 후보 데이터 생성
# ------------------------------------------------------------

def create_reset_candidates(df):
    """
    Gymnasium Environment의 reset()에서 사용할
    실제 데이터 기반 초기 상태 후보를 생성한다.

    하나의 Reset 후보에는 다음 값이 모두 포함된다.

        현재 시점:
            Rectifier_Voltage
            Rectifier_Current
            TB1-Volt

        이전 60분 TB History:
            TB_Lag1
            TB_Lag2
            TB_Lag3
            TB_Lag4
            TB_Lag5
            TB_Lag6

    Lag 값은 반드시 같은 Segment 안에서만 생성한다.

    즉 데이터 누락 구간을 넘어
    서로 다른 시점의 TB가 연결되는 것을 방지한다.
    """

    reset_df = df.copy()

    # --------------------------------------------------------
    # 같은 연속 Segment 안에서만 TB History 생성
    # --------------------------------------------------------

    grouped_tb = (
        reset_df
        .groupby("Segment")["TB1-Volt"]
    )

    reset_df["TB_Lag1"] = (
        grouped_tb.shift(1)
    )

    reset_df["TB_Lag2"] = (
        grouped_tb.shift(2)
    )

    reset_df["TB_Lag3"] = (
        grouped_tb.shift(3)
    )

    reset_df["TB_Lag4"] = (
        grouped_tb.shift(4)
    )

    reset_df["TB_Lag5"] = (
        grouped_tb.shift(5)
    )

    reset_df["TB_Lag6"] = (
        grouped_tb.shift(6)
    )


    # --------------------------------------------------------
    # Reset에 필요한 값이 모두 존재하는 행만 사용
    # --------------------------------------------------------

    reset_df = reset_df.dropna(
        subset=[
            "DateTime",
            "Segment",
            "Rectifier_Voltage",
            "Rectifier_Current",
            "TB1-Volt",
            "TB_Lag1",
            "TB_Lag2",
            "TB_Lag3",
            "TB_Lag4",
            "TB_Lag5",
            "TB_Lag6",
        ]
    ).copy()


    # index를 다시 0부터 정리
    reset_df = (
        reset_df
        .reset_index(drop=True)
    )


    return reset_df

# ------------------------------------------------------------
# 7. 전체 전처리 실행
# ------------------------------------------------------------

def preprocess_environment_data(file_path):
    """
    환경모델에 필요한 전체 전처리를 한 번에 수행한다.

    반환:
        base_df
        current_df
        tb_df
    """

    # 원본 로드
    df = load_raw_data(file_path)

    # 가상 정류기 V / I 생성
    df = create_rectifier_features(df)

    # 연속 Segment 생성
    df = create_segments(df)

    # Current Model 학습 데이터
    current_df = (
        create_current_model_data(df)
    )

    # TB History Model 학습 데이터
    tb_df = (
        create_tb_history_data(df)
    )

    return (
        df,
        current_df,
        tb_df
    )