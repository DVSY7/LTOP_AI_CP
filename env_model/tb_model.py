import joblib
import numpy as np
import pandas as pd

from env_model.config.settings import (
    TB_MODEL_PATH,
)


# ============================================================
# TB 제어 효과 설정
# ============================================================

# Action에 의해 발생한 전류 변화량 1A당
# TB 전위를 몇 mV 변화시킬지에 대한 V1 가정값
TB_CONTROL_WEIGHT = 10.0

# 제어 효과가 지나치게 커지지 않도록 제한
TB_CONTROL_LIMIT = 3.0


# ============================================================
# TB Model
# ============================================================

class TBModel:
    """
    TB History Model을 이용하여
    다음 시점의 자연적인 TB 전위를 예측하고,

    Current Model에서 계산된 delta_i_action을 이용하여
    제어에 의한 TB 변화량을 추가한다.
    """

    def __init__(self, model_path=TB_MODEL_PATH):

        self.model = joblib.load(
            model_path
        )

        # ----------------------------------------------------
        # Gym / RL 학습의 결정론적 실행 보장
        # ----------------------------------------------------

        if hasattr(
            self.model,
            "n_jobs",
        ):
            self.model.n_jobs = 1


    # ========================================================
    # 자연적인 TB 변화 예측
    # ========================================================

    def predict_natural_tb(
        self,
        TB_t,
        TB_lag1,
        TB_lag2,
        TB_lag3,
        TB_lag6,
    ):
        """
        최근 TB History를 이용하여
        제어 효과가 없는 자연적인 다음 TB를 예측한다.
        """

        # 최근 TB 변화량 계산
        tb_change_10m = (
            TB_t
            - TB_lag1
        )

        tb_change_30m = (
            TB_t
            - TB_lag3
        )


        # 모델 입력 구성
        model_input = pd.DataFrame(
            {
                "TB1-Volt": [TB_t],
                "TB_Lag1": [TB_lag1],
                "TB_Lag2": [TB_lag2],
                "TB_Lag3": [TB_lag3],
                "TB_Lag6": [TB_lag6],
                "TB_Change_10m": [tb_change_10m],
                "TB_Change_30m": [tb_change_30m],
            }
        )


        # 자연적인 다음 TB 예측
        tb_natural_next = (
            self.model.predict(
                model_input
            )[0]
        )

        return float(
            tb_natural_next
        )


    # ========================================================
    # 제어에 의한 TB 보정치 계산
    # ========================================================

    def calculate_control_effect(
        self,
        delta_i_action,
    ):
        """
        SAC의 전압 제어로 인해 발생한
        전류 변화량(delta_i_action)을 이용하여

        TB 전위 제어 효과를 계산한다.

        delta_i_action > 0
            → TB를 더 음(-)의 방향으로

        delta_i_action < 0
            → TB를 덜 음(-)의 방향으로
        """

        delta_tb_control = (
            -TB_CONTROL_WEIGHT
            * delta_i_action
        )


        # 지나치게 큰 보정 방지
        delta_tb_control = np.clip(
            delta_tb_control,
            -TB_CONTROL_LIMIT,
            TB_CONTROL_LIMIT,
        )

        return float(
            delta_tb_control
        )


    # ========================================================
    # 최종 TB 예측
    # ========================================================

    def predict(
        self,
        TB_t,
        TB_lag1,
        TB_lag2,
        TB_lag3,
        TB_lag6,
        delta_i_action,
    ):
        """
        자연적인 TB 변화와
        제어에 의한 TB 변화량을 합쳐

        최종 다음 TB를 계산한다.
        """

        # ----------------------------------------------------
        # 1. 자연적인 다음 TB 예측
        # ----------------------------------------------------

        tb_natural_next = (
            self.predict_natural_tb(
                TB_t=TB_t,
                TB_lag1=TB_lag1,
                TB_lag2=TB_lag2,
                TB_lag3=TB_lag3,
                TB_lag6=TB_lag6,
            )
        )


        # ----------------------------------------------------
        # 2. 제어에 의한 TB 변화량 계산
        # ----------------------------------------------------

        delta_tb_control = (
            self.calculate_control_effect(
                delta_i_action
            )
        )


        # ----------------------------------------------------
        # 3. 최종 다음 TB
        # ----------------------------------------------------

        tb_next = (
            tb_natural_next
            + delta_tb_control
        )


        return {
            "tb_natural_next": tb_natural_next,
            "delta_tb_control": delta_tb_control,
            "tb_next": tb_next,
        }