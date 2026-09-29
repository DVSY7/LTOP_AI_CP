"""학습된 강화학습 모델 실행 및 동작 평가."""

import argparse
import numpy as np
import time
from pathlib import Path
import sys
from stable_baselines3 import DQN, SAC

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cathodic_rl.config.settings import (
    DQN_MODEL_PATH,
    SAC_CANDIDATE_MODEL_PATH,
    RL_ALGORITHM,
    TARGET_POTENTIAL_MAX,
    TARGET_POTENTIAL_MIN,
    SAC_MAX_DELTA_VOLTAGE
)
from cathodic_rl.env.cathodic_env import CathodicProtectionEnv


ACTION_NAMES = {
    0: "전압 감소",
    1: "전압 유지",
    2: "전압 증가",
}


def evaluation_model_path(model_path: str | None = None) -> str:
    """SAC evaluation defaults to the new candidate, not the active model."""
    return model_path or (SAC_CANDIDATE_MODEL_PATH if RL_ALGORITHM == "sac" else DQN_MODEL_PATH)


def evaluate(*, model_path: str | None = None, evaluation_steps: int = 500) -> None:
    """학습된 강화학습 모델을 불러와 가상환경에서 실행한다."""

    selected_model_path = evaluation_model_path(model_path)
    if RL_ALGORITHM == "dqn":
        env = CathodicProtectionEnv(
            action_mode="discrete"
        )

        model = DQN.load(
            selected_model_path,
            env=env,
        )

    elif RL_ALGORITHM == "sac":
        env = CathodicProtectionEnv(
            action_mode="continuous"
        )

        model = SAC.load(
            selected_model_path,
            env=env,
        )

    else:
        raise ValueError(
            f"지원하지 않는 RL 알고리즘입니다: {RL_ALGORITHM}"
        )

    print(f"Model file: {Path(selected_model_path).with_suffix('.zip').resolve()}")
    observation, info = env.reset()

    total_reward = 0.0
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
                normalized_action * SAC_MAX_DELTA_VOLTAGE
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
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC/DQN model")
    parser.add_argument("--model", help="Model path with or without .zip")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    evaluate(model_path=args.model, evaluation_steps=args.steps)
