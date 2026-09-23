# 전기방식 강화학습 및 엣지 제어 프로젝트 인수인계

## 1. 프로젝트 목표

이 프로젝트의 목표는 전기방식 설비의 전압·전류·TB 전위를 바탕으로 강화학습 정책이 제어값을 결정하고, 엣지 디바이스가 산업용 PC를 통해 실제 설비를 제어하도록 만드는 것이다.

개발은 두 영역으로 구분한다.

1. 학습 프로젝트: 환경모델과 강화학습 Agent를 이용하여 제어정책을 학습한다.
2. 엣지 제어 프로젝트: 학습된 정책을 로드하여 현장 상태를 읽고 제어 설정값을 전달한다.

초기에는 고정된 학습 정책으로 폐루프 제어하며 현장 데이터를 축적한다. 이후 데이터를 이용해 현장 특성에 맞는 모델을 별도로 재학습하고, 검증된 모델만 엣지에 다시 배포한다.

## 2. 시스템 구성

```text
[엣지 디바이스]
  AI 제어 소프트웨어
        │
        │ Modbus TCP
        ▼
[산업용 PC]
  HMI, 레지스터 관리, 통신 중계
        │
        │ Modbus RTU
        ▼
[산업용 PC Simulator]
  현재 시험 단계의 가상 정류기

현장 적용 시 Simulator를 실제 정류기로 교체
```

### 통신 경계

- 엣지 디바이스는 산업용 PC하고만 Modbus TCP 통신을 수행한다.
- 엣지 디바이스는 Simulator 또는 실제 정류기와 직접 통신하지 않는다.
- 산업용 PC는 RTU 통신을 통해 Simulator 또는 실제 정류기의 상태와 설정값을 주고받는다.
- 엣지가 산업용 PC의 제어 레지스터를 변경하면 산업용 PC가 그 값을 RTU 대상에 전달한다.

## 3. 운전모드

### MANUAL

- 사용자가 산업용 PC에서 설정값을 입력한다.
- 산업용 PC가 설정값을 RTU 대상에 전달한다.
- 엣지는 현재 모드와 상태를 확인하고 로그만 남긴다.
- 엣지는 제어값을 생성하거나 Write하지 않는다.

### EXPLORE

- 엣지가 탐색 규칙으로 Action을 만든다.
- 기본 탐색 형태는 `+ΔV`, `HOLD`, `-ΔV`이다.
- Action을 절대 설정값으로 변환하고 Safety Filter를 적용한다.
- 최종 설정값을 산업용 PC에 Write한다.
- 현장 학습용 Transition을 저장한다.

### AI

- 엣지가 산업용 PC에서 상태를 읽는다.
- State Processor가 모델 입력을 구성한다.
- 학습된 SAC 정책이 Action을 출력한다.
- Action은 설정전압의 변화량 `ΔV`로 취급한다.
- 현재 설정전압에 `ΔV`를 더하여 절대 설정전압을 만든다.
- Safety Filter를 적용한 뒤 산업용 PC에 Write한다.

## 4. 학습 프로젝트의 현재 방향

- 초기 강화학습 구현은 DQN으로 수행했다.
- 최종 방향은 연속 제어가 가능한 SAC이다.
- 환경모델은 학습을 위한 Simulator이다.
- 엣지 운영에서는 환경모델이나 Gymnasium 전체를 실행하지 않는다.
- 엣지는 학습 완료된 정책의 추론만 수행한다.

### 초기 상태와 행동 개념

```text
State  = [정류기 전압, 정류기 전류, TB 전위, TB 변화량]
Action = 설정전압 변화량 ΔV
```

기존 실험용 `control_current_offset`은 환경모델 내부 누적값이라 현장에서 직접 관측할 수 없고
정책 구분 성능 개선도 확인되지 않아 SAC 입력에서 제외했다. 환경모델의 상태 전이와 진단값에는
남겨 두며, 현재 정책 입력은 현장에서 계산 가능한 4개 값만 사용한다.

State의 실제 필드, 순서, 단위, 정규화 방법은 모델 메타데이터에 명시하고 학습과 엣지에서 동일하게 유지한다.

### Reward 방향

- 목표 TB 전위 구간 유지에 양의 보상을 부여한다.
- 미방식과 과방식에는 패널티를 부여한다.
- Reward는 학습에서 모델 최적화에 사용한다.
- 엣지 운영 초기에는 모델 가중치를 변경하지 않고 Reward를 Transition 로그에 기록한다.

### Safety Filter 방향

- 모델 출력과 독립적으로 항상 적용한다.
- 설정전압 절대 범위를 제한한다.
- 제어 주기당 최대 변화량을 제한한다.
- 통신 오류나 유효하지 않은 State에서는 Write를 차단한다.
- 구체적인 한계값은 장비 사양과 현장 시험 결과에 따라 설정 파일에서 관리한다.

## 5. 현재 시험 환경

정류기 프로그램이 아직 완성되지 않았기 때문에 정션 기준으로 개발된 산업용 PC를 사용한다.

산업용 PC Simulator는 RTU 통신을 담당하며 실제 정류기 역할을 대신한다. 엣지 프로그램은 산업용 PC와 TCP 통신하고, 산업용 PC의 제어값 변경이 Simulator에 반영되는 구조다.

### 임시 레지스터 가정

| 논리 신호 | 현재 시험 주소 | 향후 정류기 주소 |
|---|---:|---:|
| 정류기 전압 | Addr 5 | Addr 3 |
| 정류기 전류 | Addr 6 | Addr 5 |

임시 매핑은 `junction_test.yaml`에 두고, 향후 매핑은 `rectifier.yaml`에 둔다.

```yaml
registers:
  rectifier_voltage:
    address: 5
    register_type: holding
    access: read
    scale: 1.0  # TODO: 프로토콜 배율 확인
    unit: V

  rectifier_current:
    address: 6
    register_type: holding
    access: read
    scale: 1.0  # TODO: 프로토콜 배율 확인
    unit: A
```

Python 코드에서는 주소를 직접 사용하지 않고 다음과 같이 논리 이름을 사용한다.

```python
voltage = register_reader.read("rectifier_voltage")
current = register_reader.read("rectifier_current")
```

YAML 파일은 주소를 보관할 뿐이며, 실제 `.read()` 메서드는 Modbus Client를 호출하도록 직접 구현해야 한다.

## 6. TB 데이터 처리

- 현재 단계에서는 TB 상태코드로 데이터 사용 가능 여부를 판단한다.
- `NORMAL`: 통신 상태가 정상이며 TB 측정값을 사용한다.
- `EER_COM`: TB가 꺼져 있거나 통신할 수 없는 상태로 처리한다.
- 값이 이전과 같다는 이유만으로 오래된 데이터라고 판정하지 않는다.
- 프로토콜 변경이 가능해지면 TB Update Flag 또는 Update Counter 추가를 검토한다.

## 7. 엣지 제어 서비스 실행 방식

엣지 프로그램은 산업용 PC에 접속하는 Modbus TCP Client이며, 종료될 때까지 일정 주기로 실행되는 제어 서비스다.

```text
프로그램 시작
    ↓
설정 및 모델 로드
    ↓
Modbus TCP 연결
    ↓
운전모드와 State Read
    ↓
데이터 유효성 확인
    ↓
MANUAL / EXPLORE / AI 분기
    ↓
필요 시 Safety Filter와 Write
    ↓
로그 저장
    ↓
다음 주기까지 대기
    ↓
반복
```

초기 개발은 Read-only로 진행한다. 통신 오류가 발생하면 해당 제어 주기를 건너뛰고 재연결하며, 마지막 정상값으로 제어하지 않는다.

## 8. 현장 데이터와 모델 고도화

초기에는 미리 학습된 모델의 가중치를 고정한다. 엣지에서 실시간 상태 피드백을 받아 매 주기 Action을 다시 계산하지만, 운전 중 모델을 즉시 학습하지 않는다.

```text
초기 데이터로 SAC V1 학습
    ↓
Simulator 검증
    ↓
엣지에 V1 배포
    ↓
현장 상태·Action·Reward 축적
    ↓
현장 내부 개발 환경에서 V2 재학습
    ↓
V1과 V2 비교 검증
    ↓
Shadow Mode 검증
    ↓
검증 통과 시 V2 배포
```

외부 클라우드는 필수 조건이 아니다. 데이터 외부 반출 없이 현장 내부 개발 PC와 엣지 장치만으로 재학습과 배포를 수행할 수 있다.

### Transition 로그 권장 필드

```text
timestamp
model_version
operation_mode
rectifier_voltage
rectifier_current
tb_potential
tb_status
raw_action
filtered_action
previous_set_voltage
applied_set_voltage
next_rectifier_voltage
next_rectifier_current
next_tb_potential
reward
communication_status
safety_filter_reason
```

## 9. 모델 배포 원칙

- 새 모델이 기존 모델보다 우수한지 검증한 후 배포한다.
- 목표 전위 유지율, 미방식, 과방식, 제어 진동, 변화량, Safety Filter 개입률을 비교한다.
- 모델을 덮어쓰지 않고 버전별로 보관한다.
- 활성 모델과 이전 모델을 설정 파일로 관리한다.
- 문제 발생 시 이전 모델로 즉시 롤백할 수 있어야 한다.

```yaml
active_model: sac_v2.zip
previous_model: sac_v1.zip
model_version: "2.0"
```

## 10. 우선 개발할 최소 기능

### 목표

산업용 PC의 임시 정류기 전압·전류 레지스터를 일정 주기로 읽고 기록한다.

### 구현 항목

1. YAML 설정 로더
2. Modbus TCP Client
3. 논리 이름 기반 Register Reader
4. 일정 주기의 Read-only 메인 루프
5. 파일 및 콘솔 Logger
6. 통신 실패와 재연결 처리
7. 단위 테스트

### 완료 조건

- 산업용 PC에 정상 연결한다.
- Addr 5, Addr 6을 설정 파일에 따라 읽는다.
- 원시값과 변환값을 함께 확인할 수 있다.
- 설정 파일의 주소를 변경해도 Python 코드를 수정하지 않는다.
- 통신 단절 후 자동으로 재연결한다.
- 통신 오류 주기에는 제어 판단을 수행하지 않는다.
- Write가 한 번도 호출되지 않는다.

## 11. 다음 Codex 작업 프롬프트 예시

```text
AGENTS.md와 docs/PROJECT_CONTEXT.md를 먼저 읽고 지침을 준수해줘.

이번 작업 범위는 엣지 제어 프로그램의 Read-only 1단계 구현이야.

1. 현재 저장소와 Git 상태를 확인한다.
2. 기존 cathodic_rl 코드는 변경하지 않는다.
3. edge_control의 최소 프로젝트 구조를 만든다.
4. junction_test.yaml에서 산업용 PC 접속 정보와 레지스터 매핑을 읽는다.
5. rectifier_voltage=Addr 5, rectifier_current=Addr 6을 주기적으로 읽는다.
6. 원시값과 변환값을 콘솔 및 로그 파일에 기록한다.
7. 통신 실패 시 Write 없이 해당 주기를 건너뛰고 재연결한다.
8. 모든 제어 Write, EXPLORE, AI 모델 연동은 이번 작업에서 제외한다.
9. 구현 후 단위 테스트와 실행 방법을 정리한다.

불명확한 프로토콜 값은 추측하지 말고 TODO로 남겨줘.
```

## 12. 아직 확인해야 할 사항

- Modbus 주소가 0-base인지 1-base인지
- Holding Register인지 Input Register인지
- Unit ID
- 전압·전류 배율과 자료형
- TB 전위의 signed 변환 방식
- 운전모드별 실제 코드값
- `NORMAL`, `EER_COM`의 실제 숫자값
- 제어 설정전압 Write 주소와 Function Code
- 제어 주기와 설비 반응 지연
- 실제 장비의 절대전압 및 변화량 안전 한계
