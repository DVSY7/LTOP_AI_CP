# AI_CP Codex 작업 지침

## 1. 목적

이 저장소는 전기방식 설비의 강화학습 정책을 개발하고, 학습된 정책을 엣지 디바이스에서 실행하는 소프트웨어를 함께 관리한다.

학습 코드와 현장 구동 코드는 분리하고, 상태 정의·행동 정의·전처리·보상·Safety Filter처럼 양쪽에서 동일해야 하는 로직만 공통 패키지로 관리한다.

## 2. 시스템 역할

### 엣지 디바이스

- 산업용 PC와 Modbus TCP로 통신한다.
- 산업용 PC의 운전모드와 상태 레지스터를 읽는다.
- MANUAL 모드에서는 제어값을 생성하거나 Write하지 않는다.
- EXPLORE 모드에서는 탐색 규칙으로 Action을 생성한다.
- AI 모드에서는 학습된 SAC 정책으로 Action을 생성한다.
- EXPLORE와 AI의 Action에는 반드시 Safety Filter를 적용한다.
- Simulator 또는 실제 정류기와 직접 통신하지 않는다.

### 산업용 PC

- 엣지 디바이스의 직접 통신 대상이다.
- 엣지와는 Modbus TCP로 통신한다.
- 현재는 정션 기준 프로그램을 사용한다.
- Simulator 또는 실제 정류기와 Modbus RTU로 통신한다.
- 엣지가 변경한 제어 설정값을 RTU 측 제어 대상에 반영한다.

### 산업용 PC Simulator

- 실제 정류기 역할을 대신하는 시험 프로그램이다.
- 산업용 PC와 Modbus RTU로 통신한다.
- 엣지 디바이스와 직접 통신하지 않는다.
- 현장 적용 시 Simulator 자리를 실제 정류기가 대체한다.

## 3. 절대 준수 규칙

1. 레지스터 주소, IP, 포트, Unit ID, 배율을 Python 코드에 직접 작성하지 않는다.
2. 장비별 값은 YAML 설정 파일에서 읽는다.
3. 제어 로직은 `address=5`와 같은 물리 주소 대신 `rectifier_voltage`와 같은 논리 이름을 사용한다.
4. MANUAL 모드에서는 Modbus Write를 수행하지 않는다.
5. 초기 개발 단계의 기본값은 `write_enabled: false`로 한다.
6. 통신 오류, 데이터 누락, TB 비정상 상태가 발생한 제어 주기에는 이전 상태값으로 제어하지 않는다.
7. TB 상태코드가 `NORMAL`인 경우에만 해당 TB 측정값을 유효 데이터로 사용한다.
8. TB 상태코드가 `EER_COM`이면 TB 꺼짐 또는 통신 불가로 처리한다.
9. EXPLORE와 AI의 제어 출력에는 항상 Safety Filter를 적용한다.
10. 모델 원본 Action, 필터 적용값, 실제 Write 값을 각각 기록한다.
11. 학습용 Gymnasium Environment와 환경모델을 엣지 런타임에서 import하지 않는다.
12. 실제 장비 Write 기능을 구현하거나 활성화하기 전에 Read-only 통합시험을 완료한다.
13. 실제 장비 제어 범위와 레지스터 사양이 불명확하면 추측하여 구현하지 말고 확인 항목으로 남긴다.
14. 기존 Git 이력, 태그, 사용자 변경사항을 임의로 삭제하거나 재작성하지 않는다.

## 4. 권장 저장소 구조

```text
AI_CP/
├─ AGENTS.md
├─ README.md
├─ docs/
│  └─ PROJECT_CONTEXT.md
├─ cathodic_rl/                 # 학습 프로젝트
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ environments/
│  └─ configs/
├─ edge_control/                # 엣지 구동 프로젝트
│  ├─ src/
│  │  ├─ main.py
│  │  ├─ communication/
│  │  ├─ controller/
│  │  ├─ state/
│  │  └─ logging/
│  ├─ configs/
│  │  ├─ junction_test.yaml
│  │  └─ rectifier.yaml
│  └─ tests/
├─ shared/
│  └─ control_core/
│     ├─ state.py
│     ├─ action.py
│     ├─ preprocessing.py
│     ├─ reward.py
│     ├─ safety_filter.py
│     └─ model_loader.py
└─ models/
   ├─ sac_v1.zip
   └─ model_metadata.yaml
```

현재 코드 구조가 위와 다르면 한 번에 전체를 이동하지 않는다. 현재 Git 루트와 import 구조를 먼저 확인하고, 단계적으로 정리한다.

## 5. 학습과 운영 코드의 경계

### `cathodic_rl`에 둘 기능

- Gymnasium Environment
- 환경모델과 Simulator
- SAC 학습 및 평가
- Replay Buffer
- 학습 결과 분석
- 모델 Export

### `edge_control`에 둘 기능

- Modbus TCP 연결과 재연결
- 레지스터 Read/Write
- YAML 기반 레지스터 매핑
- MANUAL·EXPLORE·AI 모드 분기
- 일정 주기의 메인 제어 루프
- Transition 및 운영 로그 저장
- 엣지 배포와 자동 실행

### `shared/control_core`에 둘 기능

- State 필드와 순서
- Action 형식
- 상태 전처리와 정규화
- Reward 계산
- Safety Filter
- 모델 로딩 및 메타데이터 검증

공통 코드는 복사하지 말고 한 모듈을 양쪽에서 import한다.

## 6. 현재 임시 레지스터 매핑

현재 정류기 프로그램이 완성되지 않았으므로 정션 기준 산업용 PC의 레지스터를 정류기 데이터로 간주하여 시험한다.

| 논리 데이터 | 현재 정션 시험 주소 | 향후 정류기 주소 |
|---|---:|---:|
| 정류기 전압 | Addr 5 | Addr 3 |
| 정류기 전류 | Addr 6 | Addr 5 |

주소 변경은 YAML 프로파일로 처리한다. Python 제어 로직은 변경하지 않는다.

Holding/Input Register 종류, 0-base/1-base 주소 체계, Unit ID, signed 여부, 전압·전류 배율은 실제 프로토콜 확인 후 확정한다.

## 7. 개발 순서

다음 순서를 건너뛰지 않는다.

1. Modbus TCP 연결 및 재연결
2. Read-only 레지스터 수집
3. 원시값, 변환값, 주소, 시각을 로그로 기록
4. 모드 판별
5. MANUAL 모드에서 Write가 발생하지 않는지 검증
6. 시험용 단일 Write 기능 검증
7. Safety Filter 단위 테스트
8. EXPLORE Controller 구현
9. State Processor 구현
10. SAC 모델 연동
11. Transition Logger 구현
12. 산업용 PC 및 RTU Simulator 통합시험
13. Shadow Mode 시험
14. 제한된 실제 제어시험
15. 현장 데이터 기반 재학습과 모델 재배포

## 8. 최초 구현 범위

첫 번째 구현에서는 아래 기능만 개발한다.

- 산업용 PC `192.168.1.10:502` 연결값을 설정 파일에서 읽기
- 현재 시험용 Addr 5, Addr 6 읽기
- 논리값 `rectifier_voltage`, `rectifier_current`로 변환
- 일정 주기로 반복
- 통신 실패 시 해당 주기를 건너뛰고 재연결
- 콘솔 및 파일 로그 저장
- 모든 Write 기능 비활성화

첫 번째 구현에서 EXPLORE, AI 모델, Reward 학습, 실제 제어 Write를 함께 구현하지 않는다.

## 9. 테스트 기준

- 설정 파일을 바꾸면 코드 수정 없이 레지스터 주소가 바뀌어야 한다.
- 통신이 끊겨도 프로세스가 비정상 종료되지 않아야 한다.
- 재연결 후 자동으로 Read를 재개해야 한다.
- 읽기 실패 주기에는 제어 판단을 수행하지 않아야 한다.
- MANUAL 모드 테스트에서 Write 호출 횟수는 0이어야 한다.
- Safety Filter 입력, 출력, 제한 사유를 테스트해야 한다.
- 학습과 엣지의 State 필드 순서와 전처리가 동일해야 한다.
- 모델 로드 시 메타데이터와 런타임 State/Action 정의의 호환성을 확인해야 한다.

## 10. Codex 작업 방식

- 작업 전 현재 저장소 구조, Git 상태, 관련 문서를 확인한다.
- 변경 범위를 작게 유지하고 단계별로 검증한다.
- 기존 사용자 변경사항과 충돌하는 파일은 임의로 덮어쓰지 않는다.
- 기능 추가 시 관련 테스트와 설정 예시를 함께 갱신한다.
- 프로토콜이 확정되지 않은 항목에는 명시적인 TODO를 남긴다.
- 실제 Write, 장비 제어, 모델 교체처럼 운전에 영향을 주는 기능은 기본 비활성 상태로 구현한다.
- 완료 보고에는 변경 파일, 실행 방법, 시험 결과, 남은 확인사항을 포함한다.

