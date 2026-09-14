# cathodic_rl/env/cathodic_env.py - 파일경로

"""전기방식 강화학습용 Gymnasium 환경."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
    VOLTAGE_STEP
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
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """환경을 초기 상태로 되돌린다."""

        super().reset(seed=seed)

        self.output_voltage = INITIAL_OUTPUT_VOLTAGE
        self.output_current = INITIAL_OUTPUT_CURRENT
        self.pipe_potential = INITIAL_PIPE_POTENTIAL

        self.current_step = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

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

            return normalized_action * VOLTAGE_STEP

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
        """행동을 환경에 적용한다."""

        # 1. Action 유효성 검사
        self._validate_action(action)

        # 2. Action을 출력전압 변화량으로 변환
        delta_voltage = self._convert_action(action)

        # 3. Safety Filter를 통해 안전한 변화량 계산
        applied_delta_voltage = filter_voltage_action(
            current_voltage=self.output_voltage,
            requested_delta_voltage=delta_voltage,
        )

        # 4. 안전처리된 Action을 출력전압에 적용
        self.output_voltage = float(
            self.output_voltage + applied_delta_voltage
        )

        previous_pipe_potential = self.pipe_potential

        # 5. 임시 계산식으로 출력전류와 방식전위 변경
        self._simulate_state_change(applied_delta_voltage)

        # 6. 현재 Step 증가
        self.current_step += 1
        
        # 7. 변경된 상태에 대한 Reward 계산
        if self.action_mode == "discrete":
          reward = calculate_dqn_reward(
              pipe_potential=self.pipe_potential,
              output_voltage=self.output_voltage,
              output_current=self.output_current,
              applied_delta_voltage=applied_delta_voltage,
          )

        elif self.action_mode == "continuous":
          reward = calculate_sac_reward(
              previous_pipe_potential=previous_pipe_potential,
              pipe_potential=self.pipe_potential,
              output_voltage=self.output_voltage,
              output_current=self.output_current,
              applied_delta_voltage=applied_delta_voltage,
          )

        else:
          raise ValueError(
              f"지원하지 않는 action_mode입니다: {self.action_mode}"
          )

        # 아직 별도의 환경 종료조건은 적용하지 않음
        terminated = False

        # 최대 실행 횟수 도달 여부
        truncated = self.current_step >= MAX_EPISODE_STEPS

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

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