# SAC Replay Buffer 진단 스크립트

import numpy as np

from cathodic_rl.env.cathodic_env import CathodicProtectionEnv
from cathodic_rl.models.sac_model import create_sac_model

from cathodic_rl.config.settings import (
    MIN_OUTPUT_VOLTAGE,
    MAX_OUTPUT_VOLTAGE,
    MIN_OUTPUT_CURRENT,
    MAX_OUTPUT_CURRENT,
    MIN_PIPE_POTENTIAL,
    MAX_PIPE_POTENTIAL,
    TARGET_POTENTIAL_MIN,
    TARGET_POTENTIAL_MAX,
)


# ============================================================
# 진단용 짧은 학습
# ============================================================

DIAGNOSTIC_TRAINING_STEPS = 10_000


def denormalize_tb(
    normalized_tb: float,
) -> float:
    """
    SAC Observation의 정규화된 TB를
    실제 mV 값으로 되돌린다.
    """

    return (
        normalized_tb
        * (
            MAX_PIPE_POTENTIAL
            - MIN_PIPE_POTENTIAL
        )
        + MIN_PIPE_POTENTIAL
    )


def classify_tb(
    tb: float,
) -> str:
    """
    현재 TB가 어느 제어 영역인지 분류한다.
    """

    if tb < TARGET_POTENTIAL_MIN:
        return "BELOW"

    elif tb > TARGET_POTENTIAL_MAX:
        return "ABOVE"

    else:
        return "TARGET"


def classify_action(
    action: float,
) -> str:
    """
    SAC normalized Action을
    감소 / 유지 / 증가 방향으로 분류한다.

    너무 작은 Action은 HOLD로 처리한다.
    """

    if action < -0.1:
        return "DECREASE"

    elif action > 0.1:
        return "INCREASE"

    else:
        return "HOLD"


def run_replay_buffer_analysis():

    print()
    print("=" * 80)
    print("SAC Replay Buffer 진단")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 새로운 환경 / SAC 생성
    # --------------------------------------------------------

    env = CathodicProtectionEnv(
        action_mode="continuous"
    )

    model = create_sac_model(
        env
    )


    # --------------------------------------------------------
    # 2. 진단용 짧은 학습
    # --------------------------------------------------------

    print()
    print(
        f"진단용 SAC 학습 시작: "
        f"{DIAGNOSTIC_TRAINING_STEPS} steps"
    )

    model.learn(
        total_timesteps=(
            DIAGNOSTIC_TRAINING_STEPS
        )
    )

    print()
    print("학습 완료")


    # --------------------------------------------------------
    # 3. Replay Buffer 가져오기
    # --------------------------------------------------------

    replay_buffer = (
        model.replay_buffer
    )

    buffer_size = replay_buffer.size()

    print()
    print(
        f"Replay Buffer 저장 Transition 수: "
        f"{buffer_size}"
    )


    # --------------------------------------------------------
    # 4. Replay Buffer 데이터 추출
    # --------------------------------------------------------

    observations = (
        replay_buffer
        .observations[:buffer_size]
    )

    actions = (
        replay_buffer
        .actions[:buffer_size]
    )

    rewards = (
        replay_buffer
        .rewards[:buffer_size]
    )


    # --------------------------------------------------------
    # 5. 통계 저장 구조
    # --------------------------------------------------------

    stats = {}

    groups = [
        "BELOW",
        "TARGET",
        "ABOVE",
    ]

    action_groups = [
        "DECREASE",
        "HOLD",
        "INCREASE",
    ]

    for group in groups:

        stats[group] = {}

        for action_group in action_groups:

            stats[group][action_group] = {
                "count": 0,
                "rewards": [],
                "actions": [],
            }


    # --------------------------------------------------------
    # 6. Transition 하나씩 분석
    # --------------------------------------------------------

    for i in range(buffer_size):

        # VecEnv 구조 때문에
        # observations shape이 보통
        # [buffer, env, obs] 형태이다.
        obs = observations[i, 0]

        action = float(
            actions[i, 0, 0]
        )

        reward = float(
            rewards[i, 0]
        )


        # ----------------------------------------------------
        # Observation:
        #
        # [0] Voltage
        # [1] Current
        # [2] TB
        #
        # TB만 실제 mV로 복원
        # ----------------------------------------------------

        normalized_tb = float(
            obs[2]
        )

        tb = denormalize_tb(
            normalized_tb
        )


        # ----------------------------------------------------
        # TB / Action 그룹 분류
        # ----------------------------------------------------

        tb_group = classify_tb(
            tb
        )

        action_group = classify_action(
            action
        )


        # ----------------------------------------------------
        # 통계 저장
        # ----------------------------------------------------

        stats[
            tb_group
        ][
            action_group
        ][
            "count"
        ] += 1

        stats[
            tb_group
        ][
            action_group
        ][
            "rewards"
        ].append(
            reward
        )

        stats[
            tb_group
        ][
            action_group
        ][
            "actions"
        ].append(
            action
        )


    # --------------------------------------------------------
    # 7. 결과 출력
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Replay Buffer 분석 결과")
    print("=" * 80)

    for group in groups:

        print()
        print(
            f"===== {group} ====="
        )

        for action_group in action_groups:

            data = (
                stats[
                    group
                ][
                    action_group
                ]
            )

            count = data["count"]

            if count == 0:

                print(
                    f"{action_group:<10} | "
                    f"Count=0"
                )

                continue


            mean_reward = float(
                np.mean(
                    data["rewards"]
                )
            )

            median_reward = float(
                np.median(
                    data["rewards"]
                )
            )

            mean_action = float(
                np.mean(
                    data["actions"]
                )
            )


            print(
                f"{action_group:<10} | "
                f"Count={count:5d} | "
                f"Mean Action={mean_action:+.3f} | "
                f"Mean Reward={mean_reward:+8.3f} | "
                f"Median Reward={median_reward:+8.3f}"
            )


    # --------------------------------------------------------
    # 8. Critic도 같은 학습 직후 확인
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("진단 완료")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    run_replay_buffer_analysis()
