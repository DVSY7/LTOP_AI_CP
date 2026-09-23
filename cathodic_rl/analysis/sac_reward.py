"""
현재 SAC Reward 동작 확인

목적
----
목표 전위 -1600mV를 기준으로

1. 목표 방향으로 이동
2. 반대 방향으로 이동
3. 목표 범위 유지

상황에서 Reward가 얼마나 차이나는지 확인한다.
"""

from cathodic_rl.reward.reward_function import calculate_sac_reward


def check_reward(
    name,
    previous_tb,
    current_tb,
    delta_v,
):
    reward = calculate_sac_reward(
        previous_pipe_potential=previous_tb,
        pipe_potential=current_tb,

        # 현재 V1의 대표적인 운전값
        output_voltage=43.6,
        output_current=5.95,

        applied_delta_voltage=delta_v,
    )

    print(
        f"{name:25s} | "
        f"{previous_tb:7.1f} → {current_tb:7.1f} mV | "
        f"ΔV={delta_v:+.2f} V | "
        f"Reward={reward:8.3f}"
    )


print()
print("===== BELOW : 너무 음수 =====")

# -1640에서는 -1600 방향으로 올라가는 것이 좋은 변화
check_reward(
    "목표 방향",
    -1640,
    -1635,
    -0.20,
)

check_reward(
    "변화 없음",
    -1640,
    -1640,
    0.00,
)

check_reward(
    "반대 방향",
    -1640,
    -1645,
    +0.20,
)


print()
print("===== ABOVE : 덜 음수 =====")

# -1560에서는 -1600 방향으로 내려가는 것이 좋은 변화
check_reward(
    "목표 방향",
    -1560,
    -1565,
    +0.20,
)

check_reward(
    "변화 없음",
    -1560,
    -1560,
    0.00,
)

check_reward(
    "반대 방향",
    -1560,
    -1555,
    -0.20,
)


print()
print("===== TARGET =====")

check_reward(
    "목표 중심 유지",
    -1600,
    -1600,
    0.00,
)

check_reward(
    "목표 내 이동",
    -1600,
    -1605,
    -0.10,
)

check_reward(
    "목표 이탈",
    -1600,
    -1585,
    +0.20,
)