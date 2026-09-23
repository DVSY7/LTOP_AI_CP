# cathodic_rl/config/settings.py - 파일경로

"""전기방식 강화학습 환경에서 사용하는 공통 설정값."""

# ============================================================
# 정류기 출력 설정
# ============================================================

# 정류기 출력전압의 최소값과 최대값 [V]
MIN_OUTPUT_VOLTAGE: float = 0.0
MAX_OUTPUT_VOLTAGE: float = 60.0

# 강화학습 에이전트가 한 번의 행동으로 변경하는 전압 [V]
VOLTAGE_STEP: float = 0.5

# 환경 초기화 시 사용할 출력전압 [V]
INITIAL_OUTPUT_VOLTAGE: float = 10.0


# ============================================================
# 출력전류 설정
# ============================================================

# 정류기 출력전류 범위 [A]
MIN_OUTPUT_CURRENT: float = 0.0
MAX_OUTPUT_CURRENT: float = 30.0

# 환경 초기화 시 사용할 출력전류 [A]
INITIAL_OUTPUT_CURRENT: float = 1.0

# 임시 시뮬레이션용 전류 계산 계수 [A/V]
# 실제 물리모델이 아닌 프로그램 흐름 검증용 값
TEMP_CURRENT_PER_VOLT: float = 0.1


# ============================================================
# 방식전위 설정
# ============================================================

# 시뮬레이션에서 허용하는 방식전위 범위 [mV]
MIN_PIPE_POTENTIAL: float = -3000.0
MAX_PIPE_POTENTIAL: float = 0.0

# 임시 시뮬레이션용 방식전위 반응 계수 [mV/V]
# 출력전압이 증가하면 방식전위가 음의 방향으로 이동한다고 가정
TEMP_POTENTIAL_CHANGE_PER_VOLT: float = -20.0

# ============================================================
# V1 SAC 학습용 목표 방식전위
# ============================================================
#
# 현재 환경모델이 학습한 실제 데이터 범위 안에서
# 강화학습 제어 흐름을 검증하기 위한 임시 목표값이다.
#
# 최종 현장 전기방식 목표값이 아님.
#
TARGET_POTENTIAL_MIN: float = -1610.0
TARGET_POTENTIAL_MAX: float = -1590.0

# 환경 초기화 시 사용할 방식전위 [mV]
INITIAL_PIPE_POTENTIAL: float = -700.0


# ============================================================
# 에피소드 설정
# ============================================================

# 한 에피소드에서 실행할 최대 제어 횟수
MAX_EPISODE_STEPS: int = 100


# ============================================================
# 행동 설정
# ============================================================

# DQN이 선택할 수 있는 이산 행동
ACTION_DECREASE: int = 0
ACTION_HOLD: int = 1
ACTION_INCREASE: int = 2

# 행동 번호와 실제 전압 변화량의 연결
ACTION_TO_DELTA_VOLTAGE: dict[int, float] = {
    ACTION_DECREASE: -VOLTAGE_STEP,
    ACTION_HOLD: 0.0,
    ACTION_INCREASE: VOLTAGE_STEP,
}

# ============================================================
# SAC 연속 Action 설정
# ============================================================
#
# SAC의 normalized action:
#
#   -1.0 ~ +1.0
#
# 을 실제 정류기 전압 변화량으로 변환할 때 사용하는
# V1 최대 변화량.
#
#   -1.0 → -0.20 V
#    0.0 →  0.00 V
#   +1.0 → +0.20 V
#
# EnvironmentModel의 Model Validity Guard가
# 최종 전압 범위를 다시 제한한다.
#
SAC_MAX_DELTA_VOLTAGE: float = 0.05

# ============================================================
# 재현성 설정
# ============================================================

# 난수 결과를 일정하게 재현하기 위한 기본 시드
DEFAULT_RANDOM_SEED: int = 42

# ============================================================
# 학습 설정
# ============================================================

# 1차 프로토타입 학습 횟수
TOTAL_TRAINING_STEPS: int = 10_000
SAC_BUFFER_SIZE: int =10_000
SAC_LEARNING_STARTS: int = 2_000
SAC_ENT_COEF: str = "auto"

# 학습된 모델 저장 경로. 현재 작업 디렉터리가 아니라 cathodic_rl 기준이다.
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
DQN_MODEL_PATH: str = str(_PROJECT_DIR / "models" / "trained" / "cathodic_dqn")
SAC_MODEL_PATH: str = str(_PROJECT_DIR / "models" / "trained" / "cathodic_sac")
SAC_CANDIDATE_MODEL_PATH: str = str(
    _PROJECT_DIR / "models" / "trained" / "cathodic_sac_4state_candidate"
)

# 사용할 학습모델(알고리즘) 선택
RL_ALGORITHM: str = "sac"
