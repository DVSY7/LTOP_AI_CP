"""DQN 모델 생성 테스트."""
import _path_setup

from stable_baselines3 import DQN

from env.cathodic_env import CathodicProtectionEnv
from models.dqn_model import create_dqn_model


def test_dqn_creation() -> None:
    """DQN과 환경이 정상적으로 연결되는지 확인한다."""

    env = CathodicProtectionEnv()
    model = create_dqn_model(env)

    assert isinstance(model, DQN)
    assert model.get_env() is not None

    observation, _ = env.reset()

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    assert env.action_space.contains(action)

    print(
        "\n[DQN 모델 생성 테스트]\n"
        f"Observation: {observation}\n"
        f"DQN 선택 Action: {action}\n"
        f"Action 유효성: 통과\n"
        f"결과: 통과"
    )

    env.close()


if __name__ == "__main__":
    test_dqn_creation()

    print("\nDQN 모델 생성 테스트 통과")