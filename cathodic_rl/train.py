"""전기방식 제어용 DQN 모델 학습."""

import argparse
from pathlib import Path
import sys

# `python cathodic_rl/train.py` 직접 실행도 패키지 실행과 같은 import를 사용한다.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cathodic_rl.config.settings import (
    DQN_MODEL_PATH,
    SAC_MODEL_PATH,
    SAC_CANDIDATE_MODEL_PATH,
    DEFAULT_RANDOM_SEED,
    RL_ALGORITHM,
    TOTAL_TRAINING_STEPS,
)
from cathodic_rl.env.cathodic_env import CathodicProtectionEnv
from cathodic_rl.models.dqn_model import create_dqn_model
from cathodic_rl.models.sac_model import create_sac_model


def train(*, steps=TOTAL_TRAINING_STEPS, seed=DEFAULT_RANDOM_SEED, output=None) -> Path:
    """DQN 모델을 생성하고 학습한 뒤 저장한다."""

    if RL_ALGORITHM == "dqn":
        env = CathodicProtectionEnv(
            action_mode="discrete"
        )
        model = create_dqn_model(env)
        model_path = DQN_MODEL_PATH

    elif RL_ALGORITHM == "sac":
        env = CathodicProtectionEnv(
            action_mode="continuous"
        )
        model = create_sac_model(env, seed=seed)
        model_path = output or SAC_CANDIDATE_MODEL_PATH

    else:
        raise ValueError(
            f"지원하지 않는 RL 알고리즘입니다: {RL_ALGORITHM}"
        )
    
    model_path = Path(model_path)

    print(
        f"\n[{RL_ALGORITHM.upper()} 학습 시작]\n"
        f"총 학습 Step: {steps}\n"
        f"Seed: {seed}"
    )

    model.learn(
        total_timesteps=steps,
    )

    # 모델 저장 폴더가 없으면 생성
    # model_path = Path(DQN_MODEL_PATH)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(model_path)

    print(
        f"\n[{RL_ALGORITHM.upper()} 학습 완료]\n"
        f"모델 저장 위치: {model_path}.zip"
    )

    env.close()
    return model_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="전기방식 SAC/DQN 후보 모델 학습")
    parser.add_argument("--steps", type=int, default=TOTAL_TRAINING_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output", help="확장자를 제외한 모델 출력 경로")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps는 양수여야 합니다")
    train(steps=args.steps, seed=args.seed, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
