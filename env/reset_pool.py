"""
SAC 학습용 Reachable Reset Pool 관리 모듈

역할
----
1. 실데이터에서 Reset 후보를 생성한다.
2. 현재 EnvironmentModel에서 목표 전위까지
   도달 가능한 후보인지 검사한다.
3. Reachable 후보를 CSV 캐시로 저장한다.
4. 이후 SAC 학습에서는 CSV를 불러와 사용한다.

주의
----
Reachability 계산은 Gym reset()마다 수행하지 않는다.

환경모델 또는 원본 학습 데이터가 변경되었을 때만
이 파일을 직접 실행하여 Reset Pool을 다시 생성한다.
"""

from pathlib import Path

import pandas as pd

from env_model.config.settings import DATA_PATH
from env_model.preprocessing import (
    preprocess_environment_data,
    create_reset_candidates,
)
from env_model.environment_model import EnvironmentModel


# ============================================================
# 경로
# ============================================================

ENV_DIR = Path(__file__).resolve().parent

CACHE_DIR = ENV_DIR / "cache"

RESET_POOL_CACHE_PATH = (
    CACHE_DIR
    / "reachable_reset_pool_v1.csv"
)


# ============================================================
# V1 설정
# ============================================================

TARGET_POTENTIAL_MIN = -1610.0
TARGET_POTENTIAL_MAX = -1590.0

# Reachability 검사 최대 Step
REACHABILITY_MAX_STEPS = 50

# SAC 최대 Action을 실제 ΔV로 변환했을 때의 값
# 현재 SAC_MAX_DELTA_VOLTAGE = 0.20 V
MAX_DELTA_VOLTAGE = 0.20


# ============================================================
# Reset 그룹 분류
# ============================================================

def classify_reset_group(
    pipe_potential: float,
) -> str:
    """
    초기 TB 전위를 기준으로 Reset 그룹을 분류한다.

    BELOW
        목표보다 너무 음수
        → 전압 감소 필요

    TARGET
        이미 목표 범위

    ABOVE
        목표보다 덜 음수
        → 전압 증가 필요
    """

    if pipe_potential < TARGET_POTENTIAL_MIN:
        return "BELOW"

    if pipe_potential > TARGET_POTENTIAL_MAX:
        return "ABOVE"

    return "TARGET"


# ============================================================
# 캐시 관련 함수
# ============================================================

def reset_pool_cache_exists() -> bool:
    """Reachable Reset Pool 캐시 존재 여부."""

    return RESET_POOL_CACHE_PATH.exists()


def load_reachable_reset_pool() -> pd.DataFrame:
    """
    이미 생성된 Reachable Reset Pool을 불러온다.
    """

    if not reset_pool_cache_exists():
        raise FileNotFoundError(
            "Reachable Reset Pool이 없습니다.\n"
            f"경로: {RESET_POOL_CACHE_PATH}\n"
            "먼저 python -m env.reset_pool 을 실행하세요."
        )

    return pd.read_csv(
        RESET_POOL_CACHE_PATH
    )


# ============================================================
# EnvironmentModel 초기화
# ============================================================

def initialize_environment_model(
    environment_model: EnvironmentModel,
    row: pd.Series,
) -> tuple[float, float, float]:
    """
    하나의 Reset 후보를 기준으로
    EnvironmentModel의 내부 상태를 초기화한다.

    반환
    ----
    voltage
    current
    pipe_potential
    """

    voltage = float(
        row["Rectifier_Voltage"]
    )

    current = float(
        row["Rectifier_Current"]
    )

    pipe_potential = float(
        row["TB1-Volt"]
    )

    # TB 모델이 사용할 실제 과거 이력
    tb_history = [
        float(row["TB_Lag6"]),
        float(row["TB_Lag5"]),
        float(row["TB_Lag4"]),
        float(row["TB_Lag3"]),
        float(row["TB_Lag2"]),
        float(row["TB_Lag1"]),
        float(row["TB1-Volt"]),
    ]

    environment_model.reset_history(
        tb_history
    )

    return (
        voltage,
        current,
        pipe_potential,
    )


# ============================================================
# 단일 후보 Reachability 검사
# ============================================================

def is_reachable(
    environment_model: EnvironmentModel,
    row: pd.Series,
) -> bool:
    """
    하나의 Reset 후보가 현재 V1 환경에서
    목표 전위까지 도달 가능한지 검사한다.

    BELOW
        최대 전압 감소(-0.20V)를 반복

    TARGET
        시작부터 Reachable

    ABOVE
        최대 전압 증가(+0.20V)를 반복
    """

    (
        voltage,
        current,
        pipe_potential,
    ) = initialize_environment_model(
        environment_model,
        row,
    )

    reset_group = classify_reset_group(
        pipe_potential
    )

    # --------------------------------------------------------
    # 이미 목표 범위이면 Reachable
    # --------------------------------------------------------

    if reset_group == "TARGET":
        return True

    # --------------------------------------------------------
    # 목표 방향에 따라 최대 Action 결정
    # --------------------------------------------------------

    if reset_group == "BELOW":
        # 너무 음수
        # → 전압 감소
        delta_v = -MAX_DELTA_VOLTAGE

    else:
        # 덜 음수
        # → 전압 증가
        delta_v = +MAX_DELTA_VOLTAGE

    # --------------------------------------------------------
    # 최대 50 Step 동안 목표 도달 여부 검사
    # --------------------------------------------------------

    for _ in range(
        REACHABILITY_MAX_STEPS
    ):

        result = (
            environment_model.predict_next_state(
                V_t=voltage,
                I_t=current,
                delta_v=delta_v,
            )
        )

        # EnvironmentModel이 예측한 다음 상태
        voltage = float(
            result["next_voltage"]
        )

        current = float(
            result["next_current"]
        )

        pipe_potential = float(
            result["next_tb"]
        )

        # 목표 범위 진입 여부 확인
        if (
            TARGET_POTENTIAL_MIN
            <= pipe_potential
            <= TARGET_POTENTIAL_MAX
        ):
            return True

    return False

def generate_reachable_reset_pool() -> pd.DataFrame:
    """
    전체 실데이터 Reset 후보를 검사해서
    Reachable 후보만 남기고 CSV로 저장한다.
    """

    print()
    print(
        "===== Reachable Reset Pool 생성 ====="
    )

    # ----------------------------------------
    # 원본 데이터 전처리
    # ----------------------------------------

    df, _, _ = preprocess_environment_data(
        DATA_PATH
    )

    reset_candidates = (
        create_reset_candidates(df)
        .reset_index(drop=True)
    )

    print(
        f"전체 Reset 후보 : "
        f"{len(reset_candidates)}"
    )

    # ----------------------------------------
    # EnvironmentModel 생성
    # ----------------------------------------

    environment_model = EnvironmentModel()

    reachable_rows = []

    group_total = {
        "BELOW": 0,
        "TARGET": 0,
        "ABOVE": 0,
    }

    group_reachable = {
        "BELOW": 0,
        "TARGET": 0,
        "ABOVE": 0,
    }

    # ----------------------------------------
    # 모든 후보 검사
    # ----------------------------------------

    for index, row in (
        reset_candidates.iterrows()
    ):

        pipe_potential = float(
            row["TB1-Volt"]
        )

        reset_group = (
            classify_reset_group(
                pipe_potential
            )
        )

        group_total[reset_group] += 1

        reachable = is_reachable(
            environment_model,
            row,
        )

        if reachable:

            saved_row = row.copy()

            # 나중에 Balanced Sampling할 때 사용
            saved_row["Reset_Group"] = (
                reset_group
            )

            reachable_rows.append(
                saved_row
            )

            group_reachable[
                reset_group
            ] += 1

        # 진행상황 출력
        processed = index + 1

        if processed % 100 == 0:
            print(
                f"진행 : "
                f"{processed}"
                f"/{len(reset_candidates)}"
            )

    # ----------------------------------------
    # 최종 DataFrame
    # ----------------------------------------

    reachable_pool = pd.DataFrame(
        reachable_rows
    )

    # ----------------------------------------
    # cache 폴더 자동 생성
    # ----------------------------------------

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # CSV 저장
    # ----------------------------------------

    reachable_pool.to_csv(
        RESET_POOL_CACHE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------
    # 결과 출력
    # ----------------------------------------

    print()
    print(
        "===== Reachable Reset Pool 결과 ====="
    )

    print(
        f"BELOW  : "
        f"{group_reachable['BELOW']}"
        f"/{group_total['BELOW']}"
    )

    print(
        f"TARGET : "
        f"{group_reachable['TARGET']}"
        f"/{group_total['TARGET']}"
    )

    print(
        f"ABOVE  : "
        f"{group_reachable['ABOVE']}"
        f"/{group_total['ABOVE']}"
    )

    print()

    print(
        f"최종 Reachable 후보 : "
        f"{len(reachable_pool)}"
    )

    print(
        f"저장 경로 : "
        f"{RESET_POOL_CACHE_PATH}"
    )

    return reachable_pool


# ============================================================
# 직접 실행
# ============================================================

if __name__ == "__main__":

    generate_reachable_reset_pool()