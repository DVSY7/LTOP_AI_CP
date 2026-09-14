"""SAC 모델 생성 테스트."""
import _path_setup

from stable_baselines3 import SAC

from env.cathodic_env import CathodicProtectionEnv
from models.sac_model import create_sac_model


def test_sac_creation() -> None:
    """SAC와 연속 행동 환경이 정상적으로 연결되는지 확인한다."""

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )

    model = create_sac_model(env)

    assert isinstance(model, SAC)
    assert model.get_env() is not None

    observation, _ = env.reset()

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    assert env.action_space.contains(action)

    print(
        "\n[SAC 모델 생성 테스트]\n"
        f"Observation: {observation}\n"
        f"SAC 선택 Action: {action}\n"
        f"Action 유효성: 통과\n"
        f"결과: 통과"
    )

    env.close()


if __name__ == "__main__":
    test_sac_creation()

    print("\nSAC 모델 생성 테스트 통과")