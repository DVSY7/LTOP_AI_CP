# 학습·환경 분석 스크립트

이 폴더는 모델과 환경의 동작을 사람이 확인하기 위한 진단 스크립트다. 자동 회귀 테스트가 아니므로
`cathodic_rl/tests`와 분리한다.

대표 실행 예시는 다음과 같다. AI_CP 루트에서 실행한다.

```bat
cathodic_rl\.venv\Scripts\python.exe -m cathodic_rl.analysis.action_values
cathodic_rl\.venv\Scripts\python.exe -m cathodic_rl.analysis.environment_controllability
cathodic_rl\.venv\Scripts\python.exe -m cathodic_rl.analysis.sac_policy_scan
```

| 영역 | 스크립트 |
|---|---|
| Reset 데이터·도달 가능성 | `reset_pool.py`, `reset_reachability.py`, `reset_balance.py` |
| 환경 응답 | `action_values.py`, `real_data_reset.py`, `real_data_step.py`, `environment_controllability.py`, `environment_reachability.py` |
| SAC 정책·학습 진단 | `sac_observation.py`, `sac_reward.py`, `sac_policy_scan.py`, `sac_q_curve.py`, `sac_q_value.py`, `sac_replay_buffer.py` |

자동 검증은 `cathodic_rl/tests`에만 두며 다음 명령으로 실행한다.

```bat
cathodic_rl\.venv\Scripts\python.exe -m unittest discover -s cathodic_rl/tests -v
```
