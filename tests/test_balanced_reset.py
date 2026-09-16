"""
Reachable + Balanced Reset 동작 확인

목적
----
CathodicProtectionEnv의 reset()을 반복 호출하여

BELOW / TARGET / ABOVE

세 그룹이 실제로 거의 동일한 확률로
선택되는지 확인한다.
"""

from collections import Counter

from env.cathodic_env import CathodicProtectionEnv


# ============================================================
# 테스트 설정
# ============================================================

NUM_RESETS = 300


# ============================================================
# 환경 생성
# ============================================================

env = CathodicProtectionEnv(
    action_mode="continuous"
)


# ============================================================
# Reset 반복
# ============================================================

group_counts = Counter()

for i in range(NUM_RESETS):

    observation, info = env.reset()

    reset_group = info[
        "reset_group"
    ]

    group_counts[
        reset_group
    ] += 1


# ============================================================
# 결과 출력
# ============================================================

print()
print(
    "===== Balanced Reset 테스트 ====="
)

print(
    f"전체 Reset 횟수 : "
    f"{NUM_RESETS}"
)

print()

for group_name in [
    "BELOW",
    "TARGET",
    "ABOVE",
]:

    count = group_counts[
        group_name
    ]

    ratio = (
        count
        / NUM_RESETS
        * 100
    )

    print(
        f"{group_name:6s} : "
        f"{count:3d}회 "
        f"({ratio:5.1f}%)"
    )


# ============================================================
# 간단한 검증
# ============================================================

expected_ratio = (
    1.0 / 3.0
)

tolerance = 0.10


for group_name in [
    "BELOW",
    "TARGET",
    "ABOVE",
]:

    actual_ratio = (
        group_counts[group_name]
        / NUM_RESETS
    )

    if abs(
        actual_ratio
        - expected_ratio
    ) > tolerance:

        raise AssertionError(
            f"{group_name} Reset 비율이 "
            f"너무 치우쳐 있습니다: "
            f"{actual_ratio:.1%}"
        )


print()
print(
    "Balanced Reset 테스트 통과"
)

env.close()
