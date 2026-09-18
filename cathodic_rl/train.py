"""전기방식 제어용 DQN 모델 학습."""

from pathlib import Path

from config.settings import (
    DQN_MODEL_PATH,
    SAC_MODEL_PATH,
    RL_ALGORITHM,
    TOTAL_TRAINING_STEPS,
)
from env.cathodic_env import CathodicProtectionEnv
from models.dqn_model import create_dqn_model
from models.sac_model import create_sac_model


def train() -> None:
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
        model = create_sac_model(env)
        model_path = SAC_MODEL_PATH

    else:
        raise ValueError(
            f"지원하지 않는 RL 알고리즘입니다: {RL_ALGORITHM}"
        )
    
    model_path = Path(model_path)

    print(
        f"\n[{RL_ALGORITHM.upper()} 학습 시작]\n"
        f"총 학습 Step: {TOTAL_TRAINING_STEPS}"
    )

    model.learn(
        total_timesteps=TOTAL_TRAINING_STEPS,
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


if __name__ == "__main__":
    train()