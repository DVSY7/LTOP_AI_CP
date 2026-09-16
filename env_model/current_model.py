import joblib
import numpy as np
import pandas as pd

from env_model.config.settings import (
    CURRENT_MODEL_PATH,
    MODEL_V_MIN,
    MODEL_V_MAX,
    CURRENT_CONTROL_GAIN,
)


class CurrentModel:

    def __init__(self, model_path=CURRENT_MODEL_PATH):

        self.model = joblib.load(
            model_path
        )

        # ----------------------------------------------------
        # Gym / RL 학습의 결정론적 실행 보장
        #
        # RandomForest를 여러 CPU에서 병렬 예측하면
        # 트리 결과 합산 순서 때문에 약 1e-18 수준의
        # 부동소수점 차이가 발생할 수 있다.
        #
        # 환경모델 inference에서는 재현성이 중요하므로
        # 단일 스레드로 고정한다.
        # ----------------------------------------------------

        if hasattr(
            self.model,
            "n_jobs",
        ):
            self.model.n_jobs = 1


    # ========================================================
    # 자연 전류 변화 예측
    # ========================================================

    def _predict_natural_delta_i(
        self,
        V_t,
        I_t,
    ):
        """
        제어 Action이 없다고 가정했을 때
        다음 Step에서 발생하는 자연적인 전류 변화를 예측한다.

        RandomForest는 약 43.31 ~ 43.95V 범위에서
        학습되었으므로, 해당 범위를 벗어난 V는
        가장 가까운 학습 경계값으로 제한해서 사용한다.
        """

        # ----------------------------------------------------
        # RandomForest 입력용 전압만 학습 범위로 제한
        # ----------------------------------------------------

        model_voltage = float(
            np.clip(
                V_t,
                MODEL_V_MIN,
                MODEL_V_MAX,
            )
        )


        model_input = pd.DataFrame(
            {
                "Rectifier_Current": [
                    I_t
                ],

                "Rectifier_Voltage": [
                    model_voltage
                ],

                # 자연변동을 보기 때문에
                # Action은 0으로 고정
                "Delta_V": [
                    0.0
                ],
            }
        )


        delta_i_natural = float(
            self.model.predict(
                model_input
            )[0]
        )


        return (
            delta_i_natural,
            model_voltage,
        )


    # ========================================================
    # Action에 의한 전류 변화
    # ========================================================

    def _calculate_action_delta_i(
        self,
        delta_v,
    ):
        """
        실데이터 분석으로 얻은 평균적인
        Delta_V → Delta_I 관계를 사용한다.

        V1.5:

        Delta_I_action
        =
        0.17 * Delta_V
        """

        delta_i_action = (
            CURRENT_CONTROL_GAIN
            *
            delta_v
        )


        return float(
            delta_i_action
        )


    # ========================================================
    # 최종 Current Prediction
    # ========================================================

    def predict(
        self,
        V_t,
        I_t,
        delta_v,
    ):

        # ----------------------------------------------------
        # 1. 자연적인 전류 변화
        # ----------------------------------------------------

        (
            delta_i_natural,
            model_voltage,
        ) = self._predict_natural_delta_i(
            V_t=V_t,
            I_t=I_t,
        )


        # ----------------------------------------------------
        # 2. 제어 Action에 의한 전류 변화
        # ----------------------------------------------------

        delta_i_action = (
            self._calculate_action_delta_i(
                delta_v=delta_v,
            )
        )


        # ----------------------------------------------------
        # 3. 전체 전류 변화
        # ----------------------------------------------------

        delta_i_total = (
            delta_i_natural
            +
            delta_i_action
        )


        # ----------------------------------------------------
        # 4. 다음 전류
        # ----------------------------------------------------

        next_current = (
            I_t
            +
            delta_i_total
        )


        return {

            # 전체 전류 변화
            "delta_i_total":
                delta_i_total,

            # 자연 전류 변화
            "delta_i_natural":
                delta_i_natural,

            # 이전 코드 호환성을 위해
            # baseline 이름도 같은 값으로 유지
            "delta_i_baseline":
                delta_i_natural,

            # Action에 의한 전류 변화
            "delta_i_action":
                delta_i_action,

            # 다음 전류
            "next_current":
                next_current,

            # RF가 실제로 사용한 전압
            "model_voltage":
                model_voltage,
        }