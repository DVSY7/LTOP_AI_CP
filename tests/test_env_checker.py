"""Gymnasium 환경 전체 규격 테스트."""
import _path_setup

from gymnasium.utils.env_checker import check_env

from env.cathodic_env import CathodicProtectionEnv


def test_environment_format() -> None:
    """환경이 Gymnasium 인터페이스 규칙을 따르는지 확인한다."""

    env = CathodicProtectionEnv()

    print("\n[Gymnasium 환경 규격 테스트]")

    check_env(
        env,
        skip_render_check=True,
    )

    env.close()

    print(
        "관측공간, 행동공간, reset(), step() 규격\n"
        "결과: 통과"
    )


if __name__ == "__main__":
    test_environment_format()

    print("\nGymnasium 환경 전체 검사 통과")