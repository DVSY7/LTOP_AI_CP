# cathodic_rl/env/cathodic_env.py - 파일경로

"""전기방식 강화학습용 Gymnasium 환경."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_model.environment_model import EnvironmentModel

from env.reset_pool import (
    load_reachable_reset_pool,
)

from config.settings import (
    ACTION_TO_DELTA_VOLTAGE,
    INITIAL_OUTPUT_CURRENT,
    INITIAL_OUTPUT_VOLTAGE,
    INITIAL_PIPE_POTENTIAL,
    MAX_EPISODE_STEPS,
    MAX_OUTPUT_CURRENT,
    MAX_OUTPUT_VOLTAGE,
    MAX_PIPE_POTENTIAL,
    MIN_OUTPUT_CURRENT,
    MIN_OUTPUT_VOLTAGE,
    MIN_PIPE_POTENTIAL,
    TEMP_CURRENT_PER_VOLT,
    TEMP_POTENTIAL_CHANGE_PER_VOLT,
    SAC_MAX_DELTA_VOLTAGE
)

from reward.reward_function import (
    calculate_dqn_reward,
    calculate_sac_reward
)

from safety.safety_filter import filter_voltage_action


class CathodicProtectionEnv(gym.Env):
    """전기방식 시스템을 표현하는 강화학습 환경."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 1,
    }

    def __init__(self, render_mode: str | None = None, action_mode: str = "discrete") -> None:
        """환경의 상태 공간과 행동 공간을 정의한다."""

        super().__init__()

        if render_mode not in {None, "human"}:
            raise ValueError(
                f"지원하지 않는 render_mode입니다: {render_mode}"
            )

        self.render_mode = render_mode

        # ----------------------------------------------------
        # 행동 공간
        # ----------------------------------------------------
        # 이산 행동(DQN) action_mode == "discrete"일 경우
        # 행동 0: 출력전압 감소
        # 행동 1: 출력전압 유지
        # 행동 2: 출력전압 증가
        # -----------------------------------------------------
        # 연속 행동(SAC) action_mode == "continuous"일 경우
        # -1.0 ~ 1.0의 연속 값
        if action_mode == "discrete":
            self.action_space = spaces.Discrete(
               len(ACTION_TO_DELTA_VOLTAGE)
            )
        elif action_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError(
                f"지원하지 않는 action_mode입니다: {action_mode}"
            )

        self.action_mode = action_mode

        # ----------------------------------------------------
        # 상태 공간
        # ----------------------------------------------------
        # 상태 순서:
        # [
        #     출력전압 [V],
        #     출력전류 [A],
        #     방식전위 [mV],
        # ]
        if self.action_mode == "discrete":
            #기존 DQN observation 공간 유지
            observation_low = np.array(
                [
                    MIN_OUTPUT_VOLTAGE,
                    MIN_OUTPUT_CURRENT,
                    MIN_PIPE_POTENTIAL,
                ],
                dtype=np.float32,
            )
    
            observation_high = np.array(
                [
                    MAX_OUTPUT_VOLTAGE,
                    MAX_OUTPUT_CURRENT,
                    MAX_PIPE_POTENTIAL,
                ],
                dtype=np.float32,
            )
        elif self.action_mode == "continuous":
            # SAC는 0~1로 정규화된 Observation 사용
            observation_low = np.zeros(
                3,
                dtype=np.float32
            )

            observation_high = np.ones(
                3,
                dtype=np.float32
            )
    
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # 환경 내부 상태값
        # ----------------------------------------------------
        self.output_voltage: float = INITIAL_OUTPUT_VOLTAGE
        self.output_current: float = INITIAL_OUTPUT_CURRENT
        self.pipe_potential: float = INITIAL_PIPE_POTENTIAL

        self.current_step: int = 0

        # ----------------------------------------------------
        # 실데이터 기반 Environment Model
        # ----------------------------------------------------

        self.environment_model = EnvironmentModel()


        # ----------------------------------------------------
        # Gym Reset용 Reachable Pool 준비
        #
        # 현재 EnvironmentModel에서 목표 범위까지
        # 도달 가능한 실제 초기 상태만 사용한다.
        #
        # Reachability 계산은 여기서 하지 않는다.
        # env/cache/reachable_reset_pool_v1.csv에
        # 미리 생성된 결과를 불러오기만 한다.
        # ----------------------------------------------------

        self.reset_candidates = (
            load_reachable_reset_pool()
        )


        # ----------------------------------------------------
        # Balanced Reset을 위한 그룹 분리
        #
        # BELOW
        #   목표보다 너무 음수
        #   → 전압 감소 방향 학습
        #
        # TARGET
        #   목표 범위
        #   → 유지 학습
        #
        # ABOVE
        #   목표보다 덜 음수
        #   → 전압 증가 방향 학습
        # ----------------------------------------------------

        self.reset_groups = {
            "BELOW": (
                self.reset_candidates[
                    self.reset_candidates[
                        "Reset_Group"
                    ] == "BELOW"
                ].reset_index(drop=True)
            ),

            "TARGET": (
                self.reset_candidates[
                    self.reset_candidates[
                        "Reset_Group"
                    ] == "TARGET"
                ].reset_index(drop=True)
            ),

            "ABOVE": (
                self.reset_candidates[
                    self.reset_candidates[
                        "Reset_Group"
                    ] == "ABOVE"
                ].reset_index(drop=True)
            ),
        }


        # Reset에 사용된 실제 데이터 정보
        self.reset_datetime = None
        self.reset_segment = None
        self.reset_group = None

    def _get_observation(self) -> np.ndarray:
        """현재 내부 상태를 강화학습 관측값으로 변환한다."""

        if self.action_mode == "discrete":
            # 기존 DQN 방식 그대로 유지
            return np.array(
                [
                    self.output_voltage,
                    self.output_current,
                    self.pipe_potential,
                ],
                dtype=np.float32,
            )

        if self.action_mode == "continuous":
            # SAC용 Observation 정규화
            normalized_voltage = (
                self.output_voltage - MIN_OUTPUT_VOLTAGE
            ) / (
                MAX_OUTPUT_VOLTAGE - MIN_OUTPUT_VOLTAGE
            )

            normalized_current = (
                self.output_current - MIN_OUTPUT_CURRENT
            ) / (
                MAX_OUTPUT_CURRENT - MIN_OUTPUT_CURRENT
            )

            normalized_potential = (
                self.pipe_potential - MIN_PIPE_POTENTIAL
            ) / (
                MAX_PIPE_POTENTIAL - MIN_PIPE_POTENTIAL
            )

            return np.array(
                [
                    normalized_voltage,
                    normalized_current,
                    normalized_potential
                ],
                dtype=np.float32,
            )
        raise ValueError(
            f"지원하지 않는 action_mode입니다: {self.action_mode}"
        )

    def _get_info(self) -> dict[str, Any]:
        """환경 상태에 관한 추가 정보를 반환한다."""

        return {
            "current_step": self.current_step,
            "output_voltage": self.output_voltage,
            "output_current": self.output_current,
            "pipe_potential": self.pipe_potential,

            # Episode 시작에 사용된 실제 데이터 정보
            "reset_datetime": self.reset_datetime,
            "reset_segment": self.reset_segment,
            "reset_group": self.reset_group,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        실제 데이터의 연속 구간에서
        Episode 초기 상태를 랜덤하게 선택한다.
        """

        # Gymnasium 난수 초기화
        super().reset(seed=seed)


        # --------------------------------------------------------
        # 1. Reset Group을 균등 확률로 선택
        #
        # BELOW / TARGET / ABOVE
        #
        # 각 그룹을 1/3 확률로 선택한다.
        # 데이터 개수 차이로 인한 학습 편향을 방지한다.
        # --------------------------------------------------------

        reset_group_names = [
            "BELOW",
            "TARGET",
            "ABOVE",
        ]

        group_index = int(
            self.np_random.integers(
                0,
                len(reset_group_names),
            )
        )

        selected_group = (
            reset_group_names[
                group_index
            ]
        )


        # --------------------------------------------------------
        # 2. 선택된 그룹 내부에서
        #    실제 Reset 후보 하나 랜덤 선택
        # --------------------------------------------------------

        group_candidates = (
            self.reset_groups[
                selected_group
            ]
        )

        random_index = int(
            self.np_random.integers(
                0,
                len(group_candidates),
            )
        )

        row = (
            group_candidates
            .iloc[random_index]
        )

        self.reset_group = selected_group


        # --------------------------------------------------------
        # 3. 같은 실제 시점의 V / I / TB로 상태 초기화
        # --------------------------------------------------------

        self.output_voltage = float(
            row["Rectifier_Voltage"]
        )

        self.output_current = float(
            row["Rectifier_Current"]
        )

        self.pipe_potential = float(
            row["TB1-Volt"]
        )


        # --------------------------------------------------------
        # 4. EnvironmentModel TB History 초기화
        #
        # 순서:
        # 가장 오래된 값 → 현재 값
        # --------------------------------------------------------

        initial_tb_history = [
            float(row["TB_Lag6"]),
            float(row["TB_Lag5"]),
            float(row["TB_Lag4"]),
            float(row["TB_Lag3"]),
            float(row["TB_Lag2"]),
            float(row["TB_Lag1"]),
            float(row["TB1-Volt"]),
        ]


        self.environment_model.reset_history(
            initial_tb_history
        )


        # --------------------------------------------------------
        # 5. Reset에 사용된 실제 데이터 정보 저장
        # --------------------------------------------------------

        self.reset_datetime = (
            row["DateTime"]
        )

        self.reset_segment = int(
            row["Segment"]
        )


        # --------------------------------------------------------
        # 6. Episode Step 초기화
        # --------------------------------------------------------

        self.current_step = 0


        # --------------------------------------------------------
        # 7. Gymnasium 반환값 생성
        # --------------------------------------------------------

        observation = (
            self._get_observation()
        )

        info = (
            self._get_info()
        )


        return (
            observation,
            info,
        )

    def _validate_action(self, action: Any) -> None:
        """Action이 행동 공간에 포함되는지 검사한다."""

        if not self.action_space.contains(action):
            raise ValueError(
                f"유효하지 않은 Action입니다: {action}"
            )

    def _convert_action(self, action: int) -> float:
        """Agent의 Action을 실제 출력전압 변화량으로 변환한다."""

        if self.action_mode == "discrete":
            return float(
                ACTION_TO_DELTA_VOLTAGE[int(action)]
            )

        if self.action_mode == "continuous":
            normalized_action = float(
                np.asarray(action).item()
            )

            # SAC Action:
            #
            # -1.0 ~ +1.0
            #
            # ↓
            #
            # -0.20V ~ +0.20V
            #
            return (
                normalized_action
                * SAC_MAX_DELTA_VOLTAGE
            )

        raise ValueError(
            f"지원하지 않는 Action_mode입니다: {self.action_mode}"
        )


        

    def _simulate_state_change(
        self,
        applied_delta_voltage: float,
    ) -> None:
        """임시 계산식으로 출력전류와 방식전위를 변경한다.

        실제 전기방식 물리모델이 아니며,
        Gymnasium 환경의 동작 흐름을 검증하기 위한 계산식이다.
        """

        self.output_current = float(
            np.clip(
                self.output_voltage * TEMP_CURRENT_PER_VOLT,
                MIN_OUTPUT_CURRENT,
                MAX_OUTPUT_CURRENT,
            )
        )

        self.pipe_potential = float(
            np.clip(
                self.pipe_potential
                + applied_delta_voltage
                * TEMP_POTENTIAL_CHANGE_PER_VOLT,
                MIN_PIPE_POTENTIAL,
                MAX_PIPE_POTENTIAL,
            )
        )

    def step(
    self,
    action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Agent의 Action을 실데이터 기반 EnvironmentModel에 적용한다."""

        # --------------------------------------------------------
        # 1. Action 유효성 검사
        # --------------------------------------------------------

        self._validate_action(
            action
        )


        # --------------------------------------------------------
        # 2. Agent Action → 실제 Delta_V 변환
        #
        # DQN:
        #   0 / 1 / 2 → -0.5 / 0 / +0.5V
        #
        # SAC:
        #   -1 ~ +1 → -0.20 ~ +0.20V
        #
        # SAC Action 범위는 이후 V1에 맞게 별도로 조정할 예정
        # --------------------------------------------------------

        requested_delta_voltage = (
            self._convert_action(
                action
            )
        )


        # --------------------------------------------------------
        # 3. 물리적 Safety Filter 적용
        #
        # 현재 Safety Filter:
        #   0 ~ 60V
        #
        # Environment Model의 Validity Guard와는
        # 서로 다른 역할이다.
        # --------------------------------------------------------

        safety_delta_voltage = (
            filter_voltage_action(
                current_voltage=self.output_voltage,
                requested_delta_voltage=(
                    requested_delta_voltage
                ),
            )
        )


        # --------------------------------------------------------
        # 4. 현재 TB 저장
        #
        # Reward 계산에서 이전 TB와
        # 다음 TB를 비교하기 위해 사용
        # --------------------------------------------------------

        previous_pipe_potential = (
            self.pipe_potential
        )


        # --------------------------------------------------------
        # 5. 실데이터 기반 EnvironmentModel로
        #    다음 상태 예측
        #
        # EnvironmentModel 내부에서:
        #
        #   Model Validity Guard
        #   CurrentModel
        #   TBModel
        #
        # 을 순서대로 처리한다.
        # --------------------------------------------------------

        model_result = (
            self.environment_model
            .predict_next_state(
                V_t=self.output_voltage,
                I_t=self.output_current,
                delta_v=safety_delta_voltage,
            )
        )


        # --------------------------------------------------------
        # 6. EnvironmentModel 결과를
        #    Gym 내부 상태에 반영
        # --------------------------------------------------------

        self.output_voltage = float(
            model_result[
                "next_voltage"
            ]
        )

        self.output_current = float(
            model_result[
                "next_current"
            ]
        )

        self.pipe_potential = float(
            model_result[
                "next_tb"
            ]
        )


        # --------------------------------------------------------
        # 7. 실제 EnvironmentModel에 적용된 Delta_V
        #
        # Safety Filter를 통과했더라도
        # Model Validity Guard에서 추가 제한될 수 있다.
        # --------------------------------------------------------

        effective_delta_voltage = float(
            model_result[
                "effective_delta_v"
            ]
        )


        # --------------------------------------------------------
        # 8. Step 증가
        # --------------------------------------------------------

        self.current_step += 1


        # --------------------------------------------------------
        # 9. Reward 계산
        #
        # 주의:
        # 현재 Reward 목표(-950 ~ -850mV)는
        # 기존 프로토타입 설정이다.
        #
        # 이번 단계에서는 연결 확인만 하고,
        # 이후 V1 환경에 맞는 Reward로 수정한다.
        # --------------------------------------------------------

        if self.action_mode == "discrete":

            reward = (
                calculate_dqn_reward(
                    pipe_potential=(
                        self.pipe_potential
                    ),
                    output_voltage=(
                        self.output_voltage
                    ),
                    output_current=(
                        self.output_current
                    ),
                    applied_delta_voltage=(
                        effective_delta_voltage
                    ),
                )
            )

        elif self.action_mode == "continuous":

            reward = (
                calculate_sac_reward(
                    previous_pipe_potential=(
                        previous_pipe_potential
                    ),
                    pipe_potential=(
                        self.pipe_potential
                    ),
                    output_voltage=(
                        self.output_voltage
                    ),
                    output_current=(
                        self.output_current
                    ),
                    applied_delta_voltage=(
                        effective_delta_voltage
                    ),
                )
            )

        else:

            raise ValueError(
                f"지원하지 않는 "
                f"action_mode입니다: "
                f"{self.action_mode}"
            )


        # --------------------------------------------------------
        # 10. Episode 종료 여부
        # --------------------------------------------------------

        terminated = False

        truncated = (
            self.current_step
            >= MAX_EPISODE_STEPS
        )


        # --------------------------------------------------------
        # 11. Observation 생성
        # --------------------------------------------------------

        observation = (
            self._get_observation()
        )


        # --------------------------------------------------------
        # 12. 기본 Info 생성
        # --------------------------------------------------------

        info = (
            self._get_info()
        )


        # --------------------------------------------------------
        # 13. EnvironmentModel 진단값 추가
        # --------------------------------------------------------

        info.update(
            {
                # Agent가 요청한 Delta_V
                "requested_delta_voltage":
                    requested_delta_voltage,

                # Safety Filter 통과 후 Delta_V
                "safety_delta_voltage":
                    safety_delta_voltage,

                # Model Validity Guard까지 적용된
                # 실제 EnvironmentModel Delta_V
                "effective_delta_voltage":
                    effective_delta_voltage,

                # Model 유효범위 제한 여부
                "model_limit_hit":
                    model_result[
                        "model_limit_hit"
                    ],

                # CurrentModel 결과
                "delta_i_total":
                    model_result[
                        "delta_i_total"
                    ],

                "delta_i_baseline":
                    model_result[
                        "delta_i_baseline"
                    ],

                "delta_i_action":
                    model_result[
                        "delta_i_action"
                    ],

                # TBModel 결과
                "tb_natural_next":
                    model_result[
                        "tb_natural_next"
                    ],

                "delta_tb_control":
                    model_result[
                        "delta_tb_control"
                    ],
            }
        )


        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        """현재 환경 상태를 화면에 출력한다."""

        if self.render_mode == "human":
            print(
                f"Step={self.current_step} | "
                f"전압={self.output_voltage:.1f} V | "
                f"전류={self.output_current:.1f} A | "
                f"전위={self.pipe_potential:.1f} mV"
            )

    def close(self) -> None:
        """환경 종료 시 사용하는 정리 메서드."""

        pass