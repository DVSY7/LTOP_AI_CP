"""학습된 강화학습 모델 실행 및 동작 평가."""

import numpy as np
import time
from stable_baselines3 import DQN, SAC

from config.settings import (
    DQN_MODEL_PATH,
    SAC_MODEL_PATH,
    RL_ALGORITHM,
    TARGET_POTENTIAL_MAX,
    TARGET_POTENTIAL_MIN,
    VOLTAGE_STEP,
)
from env.cathodic_env import CathodicProtectionEnv


ACTION_NAMES = {
    0: "전압 감소",
    1: "전압 유지",
    2: "전압 증가",
}


def evaluate() -> None:
    """학습된 강화학습 모델을 불러와 가상환경에서 실행한다."""

    if RL_ALGORITHM == "dqn":
        env = CathodicProtectionEnv(
            action_mode="discrete"
        )

        model = DQN.load(
            DQN_MODEL_PATH,
            env=env,
        )

    elif RL_ALGORITHM == "sac":
        env = CathodicProtectionEnv(
            action_mode="continuous"
        )

        model = SAC.load(
            SAC_MODEL_PATH,
            env=env,
        )

    else:
        raise ValueError(
            f"지원하지 않는 RL 알고리즘입니다: {RL_ALGORITHM}"
        )

    observation, info = env.reset()

    total_reward = 0.0
    evaluation_steps = 500

    print(
        f"\n[학습된 {RL_ALGORITHM.upper()} 모델 실행]"
    )

    print(
        f"초기 상태 | "
        f"전압={info['output_voltage']:.1f} V | "
        f"전류={info['output_current']:.2f} A | "
        f"전위={info['pipe_potential']:.1f} mV"
    )

    for step_number in range(1, evaluation_steps + 1):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        # ------------------------------------------------
        # DQN / SAC Action 처리
        # ------------------------------------------------
        if RL_ALGORITHM == "dqn":

            action_for_env = int(
                np.asarray(action).item()
            )

            action_text = (
                f"{action_for_env}"
                f"({ACTION_NAMES[action_for_env]})"
            )

        else:

            action_for_env = np.asarray(
                action,
                dtype=np.float32,
            )

            normalized_action = float(
                action_for_env.item()
            )

            delta_voltage = (
                normalized_action * VOLTAGE_STEP
            )

            action_text = (
                f"{normalized_action:+.4f} "
                f"(ΔV={delta_voltage:+.4f} V)"
            )

        observation, reward, terminated, truncated, info = env.step(
            action_for_env
        )

        total_reward += reward

        target_reached = (
            TARGET_POTENTIAL_MIN
            <= info["pipe_potential"]
            <= TARGET_POTENTIAL_MAX
        )

        print(
            f"Step {step_number:03d} | "
            f"Action={action_text} | "
            f"전압={info['output_voltage']:.2f} V | "
            f"전류={info['output_current']:.2f} A | "
            f"전위={info['pipe_potential']:.1f} mV | "
            f"Reward={reward:.5f} | "
            f"목표={'도달' if target_reached else '미도달'}"
        )

        time.sleep(0.3)

        if terminated or truncated:
            print("에피소드가 종료되었습니다.")
            break

    final_potential = info["pipe_potential"]

    print(
        f"\n[{RL_ALGORITHM.upper()} 실행 결과]\n"
        f"실행 Step: {info['current_step']}\n"
        f"최종 방식전위: {final_potential:.1f} mV\n"
        f"누적 Reward: {total_reward:.5f}\n"
        f"최종 목표 상태: "
        f"{'도달' if TARGET_POTENTIAL_MIN <= final_potential <= TARGET_POTENTIAL_MAX else '미도달'}"
    )

    env.close()


if __name__ == "__main__":
    evaluate()