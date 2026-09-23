# 테스트와 분석 안내

이 프로젝트에서는 **자동 테스트**와 **분석 스크립트**를 구분한다. 자동 테스트는 명확한 성공 조건을 코드로 확인하며, 분석 스크립트는 출력값을 보고 환경모델과 학습 정책의 성질을 해석한다.

## 자동 테스트 실행

AI_CP 루트에서 다음을 실행한다.

```bat
cathodic_rl\.venv\Scripts\python.exe -m unittest discover -s edge_control/tests -v
cathodic_rl\.venv\Scripts\python.exe -m unittest discover -s cathodic_rl/tests -v
cathodic_rl\.venv\Scripts\python.exe -m unittest discover -s shared/tests -v
```

## edge_control

| 파일 | 확인하는 내용 |
|---|---|
| `test_read_only.py` | YAML 설정, Modbus 읽기·재연결, 오류 주기의 무효화, JSONL 기록 |
| `test_control_guard.py` | REMOTE·ON·TB NORMAL·운전모드에 따른 Action/Write 허용 판정 |
| `test_single_write.py` | Write 활성화 조건, 공학값 인코딩, 단일 Write와 Read-back 검증 |
| `test_explore_controller.py` | EXPLORE Safety Filter, 42~45 V 왕복 및 Hold 패턴 |
| `test_ai_controller.py` | SAC 입력 정규화, AI Safety Filter, 42~45 V 제한·복구 |
| `test_state_processor.py` | 4개 정책 State와 TB trend의 준비·초기화 규칙 |

## cathodic_rl

| 파일 | 확인하는 내용 |
|---|---|
| `test_environment_contract.py` | DQN 3차원 State, SAC 4차원 State, Gymnasium `reset`/`step` 형식, 종료 주기 |
| `test_model_factory.py` | DQN·SAC 모델 생성 후 해당 Action 공간과 연결되는지 |
| `test_policy_gate.py` | 방향성 있는 정책은 통과하고 한 방향으로 포화된 정책은 실패하는지 |
| `test_sac_reward_direction.py` | BELOW·ABOVE·TARGET 상태에서 Reward가 의도한 변화 방향을 선호하는지 |

`cathodic_rl/analysis`는 Reset 분포, 환경 반응, SAC Q값·정책 스캔처럼 사람이 결과를 해석하는 스크립트다. 목록은 [analysis README](../cathodic_rl/analysis/README.md)에 있다.

## shared

| 파일 | 확인하는 내용 |
|---|---|
| `test_state.py` | edge와 학습이 공유하는 State 필드 순서, 단위, 정규화 |
| `test_safety_filter.py` | 전압 범위와 제어 주기당 변화량을 제한하는 공통 Safety Filter |
| `test_model_loader.py` | 운영 모델 메타데이터, State/Action 계약, SHA-256 무결성 |

## env_model 진단 스크립트

`cathodic_rl/env_model/tests`는 현재 모두 **수동 실행용 진단 스크립트**다. 파일명이 `test_`로 시작해도 `unittest discover`로 실행하면 안 된다. 데이터 파일을 읽거나 많은 출력값을 만들며, 일부는 최상위 코드에서 즉시 실행된다.

| 파일 | 확인하는 내용 | 결과를 볼 때 확인할 점 |
|---|---|---|
| `test_current_model.py` | 설정전압 변화 `Delta_V`에 따른 전류 예측 | `Delta_I`와 `next_current`의 방향·크기 |
| `test_tb_model.py` | 제어 전류 변화가 TB 예측에 더하는 보정 | 자연 TB와 제어 보정값을 구분 |
| `test_environment_model.py` | CurrentModel과 TBModel을 결합한 여러 Step 전이 | 각 Step의 History 갱신과 다음 V/I/TB |
| `test_environment_rollout.py` | Hold·고정 증가·무작위 변화로 144 Step 연속 실행 | 전압·전류·TB가 발산하거나 비현실적 범위를 갖지 않는지 |
| `test_env_model_preprocessing.py` | 원본 데이터의 전처리 결과와 학습용 표 | 행 수와 필수 열·결측치 |
| `test_reset_candidates.py` | Gymnasium reset에 사용할 TB History 후보 생성 | 후보 수, 7개 TB History, 결측치 |
| `analyze_current_control.py` | 실제 데이터와 CurrentModel의 전류 제어 반응 비교 | `Delta_V → Delta_I` 기울기와 모델 예측의 차이 |

이 스크립트들은 데이터·환경모델을 바꾼 뒤 필요한 항목만 직접 실행한다. 다음 정리 단계에서는 이 폴더도 자동 테스트와 분석 폴더로 분리할 수 있다.
