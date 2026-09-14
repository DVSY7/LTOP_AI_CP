import numpy as np

from env_model.current_model import CurrentModel
from env_model.tb_model import TBModel
from env_model.config.settings import (
    MODEL_VALID_V_MIN,
    MODEL_VALID_V_MAX,
)


# ============================================================
# Environment Model
# ============================================================

class EnvironmentModel:
    """
    Current Model과 TB Model을 통합하여
    현재 상태와 SAC Action(Delta_V)을 입력받고
    다음 상태를 계산한다.

    TB History는 EnvironmentModel 내부에서 관리한다.
    """

    def __init__(self):

        # 전류 예측 모델
        self.current_model = CurrentModel()

        # TB 예측 모델
        self.tb_model = TBModel()

        # TB 과거 이력 저장용
        self.tb_history = []


    # ========================================================
    # TB History 초기화
    # ========================================================

    def reset_history(
        self,
        initial_tb_history,
    ):
        """
        Episode 시작 시 실제 연속 TB 데이터를 이용하여
        TB History를 초기화한다.

        최소 7개의 TB 값이 필요하다.

        예:
        [
            -1602,
            -1600,
            -1599,
            -1598,
            -1595,
            -1597,
            -1593,
        ]

        마지막 값이 현재 TB_t가 된다.
        """

        if len(initial_tb_history) < 7:

            raise ValueError(
                "TB History는 최소 7개의 값이 필요합니다."
            )

        self.tb_history = list(
            initial_tb_history
        )


    # ========================================================
    # 현재 TB History에서 Lag 값 추출
    # ========================================================

    def _get_tb_lags(self):
        """
        내부 TB History에서
        현재 TB와 필요한 Lag 값을 가져온다.
        """

        if len(self.tb_history) < 7:

            raise RuntimeError(
                "TB History가 초기화되지 않았습니다. "
                "reset_history()를 먼저 호출하세요."
            )

        TB_t = self.tb_history[-1]

        TB_lag1 = self.tb_history[-2]
        TB_lag2 = self.tb_history[-3]
        TB_lag3 = self.tb_history[-4]
        TB_lag6 = self.tb_history[-7]

        return {
            "TB_t": TB_t,
            "TB_lag1": TB_lag1,
            "TB_lag2": TB_lag2,
            "TB_lag3": TB_lag3,
            "TB_lag6": TB_lag6,
        }


    # ========================================================
    # 다음 상태 예측
    # ========================================================

    def predict_next_state(
        self,
        V_t,
        I_t,
        delta_v,
    ):
        """
        현재 전압, 현재 전류, Delta_V를 이용하여
        다음 상태를 계산한다.

        TB_t와 TB Lag들은
        내부 tb_history에서 자동으로 가져온다.
        """

        # ----------------------------------------------------
        # 1. TB History에서 현재 TB / Lag 추출
        # ----------------------------------------------------

        tb_lags = self._get_tb_lags()

        TB_t = tb_lags["TB_t"]
        TB_lag1 = tb_lags["TB_lag1"]
        TB_lag2 = tb_lags["TB_lag2"]
        TB_lag3 = tb_lags["TB_lag3"]
        TB_lag6 = tb_lags["TB_lag6"]


        # ----------------------------------------------------
        # 2. 다음 전압 계산
        # ----------------------------------------------------

            # ========================================================
            # 1. SAC가 요청한 Action
            # ========================================================

        requested_delta_v = float(delta_v)

        requested_next_v = (
            V_t
            +
            requested_delta_v
        )


            # ========================================================
            # 2. Environment Model 유효범위 적용
            # ========================================================

        V_next = float(
            np.clip(
                requested_next_v,
                MODEL_VALID_V_MIN,
                MODEL_VALID_V_MAX,
            )
        )


        # 실제 Environment에 적용되는 Action
        effective_delta_v = (
            V_next
            -
            V_t
        )


        # 요청 Action이 제한되었는지 확인
        model_limit_hit = not np.isclose(
            requested_delta_v,
            effective_delta_v,
        )


        # ----------------------------------------------------
        # 3. Current Model 예측
        # ----------------------------------------------------

        current_result = (
            self.current_model.predict(
                V_t=V_t,
                I_t=I_t,
                delta_v=effective_delta_v,
            )
        )

        delta_i_total = (
            current_result[
                "delta_i_total"
            ]
        )

        delta_i_baseline = (
            current_result[
                "delta_i_baseline"
            ]
        )

        delta_i_action = (
            current_result[
                "delta_i_action"
            ]
        )

        I_next = (
            current_result[
                "next_current"
            ]
        )


        # ----------------------------------------------------
        # 4. TB Model 예측
        # ----------------------------------------------------

        tb_result = (
            self.tb_model.predict(
                TB_t=TB_t,
                TB_lag1=TB_lag1,
                TB_lag2=TB_lag2,
                TB_lag3=TB_lag3,
                TB_lag6=TB_lag6,
                delta_i_action=delta_i_action,
            )
        )

        tb_natural_next = (
            tb_result[
                "tb_natural_next"
            ]
        )

        delta_tb_control = (
            tb_result[
                "delta_tb_control"
            ]
        )

        TB_next = (
            tb_result[
                "tb_next"
            ]
        )


        # ----------------------------------------------------
        # 5. 새로운 TB를 History에 추가
        # ----------------------------------------------------

        self.tb_history.append(
            TB_next
        )


        # ----------------------------------------------------
        # 6. History 길이 제한
        # ----------------------------------------------------
        # 실제로 필요한 것은 최근 7개뿐이므로
        # 너무 길어지지 않도록 유지한다.

        if len(self.tb_history) > 7:

            self.tb_history = (
                self.tb_history[-7:]
            )


        # ----------------------------------------------------
        # 7. 다음 상태 및 진단값 반환
        # ----------------------------------------------------

        return {

            # 현재 TB
            "current_tb": TB_t,

            # 다음 상태
            "next_voltage": V_next,
            "next_current": I_next,
            "next_tb": TB_next,

            # Current Model 진단값
            "delta_i_total": delta_i_total,
            "delta_i_baseline": delta_i_baseline,
            "delta_i_action": delta_i_action,

            # TB Model 진단값
            "tb_natural_next": tb_natural_next,
            "delta_tb_control": delta_tb_control,

            # 추가(제한된 요청 정보)
            "requested_delta_v": requested_delta_v,
            "effective_delta_v": effective_delta_v,
            "model_limit_hit": model_limit_hit,
        }