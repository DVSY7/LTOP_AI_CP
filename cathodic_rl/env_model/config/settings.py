from pathlib import Path

# 데이터가 있는 경로를 현재 경로를 기준으로 찾게하기 위함
ENV_MODEL_DIR = Path(__file__).resolve().parent.parent

# 환경모델 학습에 필요한 데이터 경로
DATA_PATH = (
    ENV_MODEL_DIR
    /"data"
    /"환경모델 학습용 데이터.xlsx"
)

# 모델 저장 폴더
SAVE_DIR = (
    ENV_MODEL_DIR
    / "saved_models"
)

# 전류모델 저장 경로
CURRENT_MODEL_PATH = (
    SAVE_DIR
    / "current_model_v1.joblib"
)

# TB모델 저장 결로
TB_MODEL_PATH = (
    SAVE_DIR
    / "tb_history_model_v1.joblib"
)

# ============================================================
# Current Model V1.5 설정
# ============================================================

# Current Model이 학습한 Rectifier Voltage 범위
MODEL_V_MIN = 43.3133
MODEL_V_MAX = 43.9525

# 실데이터 분석에서 얻은 평균적인
# 전압 변화량 → 전류 변화량 계수
#
# Delta_I_action = CURRENT_CONTROL_GAIN * Delta_V
CURRENT_CONTROL_GAIN = 0.17

# ============================================================
# Environment Model V1 유효 범위
# ============================================================

# 실제 학습 데이터 전압 범위:
# 약 43.31 ~ 43.95 V
#
# V1에서는 약간의 Margin을 허용한다.
MODEL_VALID_V_MIN = 42.10
MODEL_VALID_V_MAX = 45.10