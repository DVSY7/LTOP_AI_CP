# 산업용 PC Read-only 수집

엣지가 산업용 PC의 전압·전류를 Modbus TCP로 읽어 콘솔과 회전 JSONL 파일에 기록한다.
운전 상태와 TB 값을 수집하고 Control Guard 판정을 기록한다. EXPLORE 모드에서는 Read-only
Action 후보와 Safety Filter 결과를 기록한다. AI 추론과 Reward는 아직 포함하지 않는다.
학습 코드나 환경모델을 import하지 않는다. 기존 cathodic_rl은 변경하지 않았다.

## 설치 및 실행

현재 프로젝트는 `cathodic_rl/.venv` 가상환경을 공용으로 사용한다.

```powershell
cathodic_rl/.venv/Scripts/python.exe -m edge_control.src.main --config edge_control/configs/junction_test.yaml --check-config
cathodic_rl/.venv/Scripts/python.exe -m edge_control.src.main --config edge_control/configs/junction_test.yaml
```

`--check-config`는 통신 없이 설정만 검사한다. `--max-cycles 10`을 추가하면 10주기 후 종료한다.
기본은 Ctrl+C까지 실행한다. Ctrl+C 및 루프 종료 시 연결을 닫는다.

Linux edge 장비 배포와 systemd 자동 실행 절차는 [deploy/linux README](deploy/linux/README.md)에 정리했다.

현재 `junction_test.yaml`은 사용자 확인과 TCP 읽기 시험에 맞춘 활성 시험 설정이다.
필수 `null` 값을 확인하여 채워야 실제 연결을 시작한다. `rectifier.yaml`은 추가로 `enabled: false`이다.
일반 수집기는 `write_enabled: true` 설정을 거부한다. 시험용 단일 Write는 별도 명령과 이중 활성화가 필요하다.

## 프로토콜 확인

| 설정 | 의미 / 현재 확인 상태 |
|---|---|
| connection.host / port | 정션 시험 대상 192.168.1.10:502 |
| connection.unit_id | TODO: PC 서버가 사용하는 장치 식별자 |
| address_base | TODO: 문서 주소가 0 기준이면 0, 1 기준이면 1. 요청 주소 = address - address_base |
| register_type | TODO: holding 또는 input. 정상 상태 코드와 관련 없음 |
| data_type | TODO: uint16, int16, uint32, int32, float32 |
| byte_order | TODO: 각 16-bit word 안의 byte 순서 big / little |
| word_order | TODO: 여러 word의 순서 big / little. 단일 word는 big 지정 |
| scale / offset | TODO: 공학값 = decoded × scale + offset |
| polling / timeout / reconnect | 예시 운영값이며 PC 갱신 주기·응답시간에 맞춰 조정 |

사용자가 설명한 전압 `4500 → 45 V`는 곱셈 배율 `0.01`에 해당한다.
다만 추정 정보이므로 아직 활성 설정에 반영하지 않았다. 전류의 배율도 별도 확인이 필요하다.
Holding Register는 읽기·쓰기에, Input Register는 읽기 전용으로 사용한다.
기존 PC의 읽기 호출과 레지스터 표에서 종류와 주소 기준을 확인해야 한다.
근거: [Modbus Application Protocol](https://www.modbus.org/specs.php).

후속 단계 TODO: TB 상태 코드의 최종 명칭, 제어 설정값 Write 사양 확인,
RTU 오류·데이터 갱신 상태, 현재 설정전압 주소, Write 사양, 현장 안전 한계.
TCP 읽기가 성공했다는 사실만으로 RTU 데이터 최신성이 보장되지는 않는다.

## 수집 및 로그 규칙

- 제어 로직은 논리 이름 `rectifier_voltage`, `rectifier_current`만 사용한다.
- 설정한 자료형에 따라 1개 또는 2개 레지스터를 읽는다. 추가 자료형은 사양 확인 후 확장한다.
- 전압·전류는 순차 읽기이며 동일 시점의 원자적 스냅샷은 보장하지 않는다.
- 읽기 실패 시 해당 주기의 `valid`는 false, `measurements`는 null이다. 이전 값으로 채우지 않는다.
- 실패 지점 이후 신호는 그 주기에 읽지 않는다. 로그에는 실패한 논리 신호와 오류 메시지만 `read_error`로 남긴다.
- 실패 후 연결을 닫고 설정한 간격 뒤 다음 주기에 재연결한다.
- 정상 주기는 처리 시간을 뺀 나머지만 대기하며 처리 시간이 주기를 넘으면 즉시 다음 주기를 시작한다.
- 콘솔과 JSONL의 시각은 한국 표준시(KST) 기준으로 초 단위까지 기록한다. 콘솔에는 모드, V/I/TB, 제안·Write 결과, 오류를 한 줄로 요약해 표시한다. JSONL은 재학습 transition에 필요한 공학 단위 측정값, 운전 상태, TB trend, 적용 Action, Read-back 결과와 오류만 기록한다.
- 파일 경로는 YAML 파일 기준이며 기본 파일은 `edge_control/logs/junction_test.jsonl`이다.
- 기본 로그는 약 5 MB마다 회전하며 이전 파일 3개를 보관한다.

## 시험

```powershell
cathodic_rl/.venv/Scripts/python.exe -m unittest discover -s edge_control/tests -v
```

모의 통신 테스트는 실제 장비에 접속하지 않는다. `tests/fixtures/confirmed_mock.yaml`의 값은
가상 프로토콜이며 현장 설정으로 복사하지 않는다.
설정 변경, signed/float·byte/word 변환, 오류 응답과 길이 검사,
단절 후 복구, 이전 값 재사용 방지, 일반 수집기의 Write 차단, 종료 처리, 콘솔·파일 로그를 검증한다.

단일 Write 차단·Read-back을 포함한 모의 테스트 39개가 통과한다.

전압·전류는 TCP 주소 6·7에서 원시값 4500·1200 수신을 확인했다.
사용자가 통신 단절 후 복구도 확인했다. 추가한 모드 레지스터는 실제 HMI 표시와 대조가 필요하다.
통합시험 절차는 다음과 같다.

1. YAML을 확정하고 `--check-config` 통과 확인.
2. 제한된 주기 수로 수집하고 원시값·변환값을 산업용 PC 표시값과 대조.
3. 통신 단절 및 복구 시 무효 주기·재연결·새 값 수집 확인.
4. PC 측 통신 로그에서 읽기 요청만 발생했는지 확인.
5. 결과를 기록한 후 모드 판별 및 후속 제어 단계 검토.

## 정션 운전 상태 읽기

사용자가 확인한 TCP 주소와 코드값을 `junction_test.yaml`에 추가했다.

| 논리 이름 | 주소 | 코드 |
|---|---:|---|
| remote_mode | 1 | 0=LOCAL, 1=REMOTE |
| operation_status | 2 | 0=OFF, 1=ON |
| operation_mode | 3 | 0=AI, 1=MANUAL, 2=EXPLORE |

숫자 매핑은 YAML `codes`로 관리한다. `"OFF"`, `"ON"`은 YAML boolean 해석을 막기 위해 따옴표가 필요하다.
전압·전류만 있는 기존 프로파일도 지원한다. 상태 항목을 하나라도 추가하면 세 항목 모두 필요하다.
정상 주기의 `states`에는 판별한 이름이, `measurements`에는 공학 단위 숫자값이 기록된다.
미정의 코드는 오류 사유를 기록하고 해당 주기의 measurements/states를 null로 처리한다.
읽기 오류가 발생하면 이전 모드를 재사용하지 않는다.
MANUAL을 포함한 모든 상태에서 Action 생성과 Write는 수행하지 않는다.
각 신호는 순차 읽기이므로 이 로그는 제어 시점의 원자적 모드 확인을 대신하지 않는다.

```powershell
python -m edge_control.src.main --config edge_control/configs/junction_test.yaml --max-cycles 3
```

HMI 상태와 로그의 states를 대조하여 확인한다. 모의 시험은 LOCAL/OFF/AI,
REMOTE/ON/MANUAL, 미정의 코드, 통신 실패, 복구 및 Write 호출 0회를 검증한다.

## Control Guard

`src/controller/control_guard.py`는 통신이나 Action 계산 없이 현재 주기의 제어 가능 여부만 판정한다.
정상적인 `REMOTE + ON + (AI 또는 EXPLORE) + TB NORMAL` 조합에서만 `action_allowed=true`가 된다.
통신 오류·데이터 누락, LOCAL, OFF, MANUAL, TB 오류는 사유와 함께 차단한다.

현재 `write_enabled: false`이므로 정상 EXPLORE 상태에서도 `write_allowed=false`이며
차단 사유에 `write_disabled`가 기록된다. Guard는 Write 메서드를 호출하지 않는다.
각 수집 주기의 JSONL 로그에서 `control_guard` 항목을 확인할 수 있다.

## 시험용 단일 Write

`src/test_write.py`는 설정에서 `access: read_write`로 지정한 16-bit Holding Register에만
한 번 Write하고 같은 주소를 즉시 Read-back한다. 일반 수집 루프와 분리되어 있으며 반복 Write를 하지 않는다.

실행하려면 YAML의 `write_enabled`를 명시적으로 `true`로 바꾸고 확인 문자열도 전달해야 한다.
현재 `junction_test.yaml`은 `false`이므로 아래 명령을 실행해도 장비에 연결하기 전에 차단된다.

```powershell
python -m edge_control.src.test_write `
  --config edge_control/configs/junction_test.yaml `
  --signal set_voltage `
  --value 45.0 `
  --confirm I_UNDERSTAND_SINGLE_TEST_WRITE
```

실제 시험은 산업용 PC와 RTU Simulator 연결 상태, 설정전압 주소 20과 배율 0.01,
시험값을 확인한 후 수행한다. 시험이 끝나면 `write_enabled: false`로 되돌린다.

## Safety Filter 준비

학습 코드와 엣지 런타임이 같은 계산을 사용하도록 순수 계산 모듈을
`shared/control_core/safety_filter.py`에 두었다. 필터는 현재 설정전압과 요청 변화량을 받아
다음 순서로 처리하고, 원본 요청값·필터 결과·개입 여부·제한 사유를 함께 반환한다.

1. 제어 주기당 최대 전압 변화량 제한
2. 최종 설정전압의 최소·최대 범위 제한

비정상 숫자와 허용 범위 밖의 현재 전압은 이전 값에 기대어 제어하지 않고 오류로 처리한다.
기존 `cathodic_rl/safety/safety_filter.py`도 이 공통 모듈을 호출하되 기존 함수 인터페이스는 유지한다.

확인된 안전 한계는 설정전압 0~60 V, 제어 주기당 최대 변화량 ±3 V다.
`junction_test.yaml`의 `safety`에서 이 값을 관리하며 설정 로더가 범위와 필수값을 검증한다.
현재는 Read-only 단계이므로 Safety Filter가 Action을 계산하거나 Write를 수행하지 않는다.

```powershell
cathodic_rl/.venv/Scripts/python.exe -m unittest discover -s shared/tests -v
```

## EXPLORE Controller

`src/controller/explore_controller.py`는 Guard가 허용한 EXPLORE 주기에만 연속 Action 후보를
생성한다. 요청 변화량은 확인된 주기당 한계인 `-3~+3 V`의 균등분포에서 선택하며,
현재 `set_voltage`를 기준으로 공통 Safety Filter를 반드시 거친다.

주기 로그의 `control_action`에는 다음 값이 분리되어 기록된다.

- `original_delta_voltage`: EXPLORE가 처음 요청한 변화량
- `filtered_delta_voltage`: Safety Filter가 허용한 변화량
- `target_set_voltage`: 필터 적용 후 목표 설정전압
- `safety_intervened`, `safety_reasons`: 필터 개입 여부와 사유
- `actual_write_voltage`: Read-only 단계에서는 항상 `null`

전압 관련 Action 값은 내부 계산 정밀도를 유지하며 JSONL 로그에는 소수점 둘째 자리로
반올림해 기록한다.

LOCAL, OFF, MANUAL, TB 오류, 통신 오류에서는 Action을 생성하지 않는다. AI 모드도 모델이
연결되기 전이므로 EXPLORE Controller를 호출하지 않는다. `write_enabled: false`를 유지하며
EXPLORE 결과는 산업용 PC에 Write하지 않는다.

## State Processor

`shared/control_core/state.py`에 현재 확인 가능한 기본 State의 필드 순서와 단위를 고정했다.

```text
[rectifier_voltage (V), rectifier_current (A), tb_potential (mV)]
```

`src/state/state_processor.py`는 현재 주기가 유효하고 TB 상태가 NORMAL일 때만 이 State를 만든다.
필드 누락, 숫자가 아닌 값, NaN 또는 무한대는 차단하며 이전 주기의 값을 채우지 않는다.
주기 로그의 `processed_state`에는 `field_order`, `units`, `values`, `vector`가 기록된다.

정책 입력은 현장에서 관측 가능한 다음 4개 값으로 정의한다.

```text
[rectifier_voltage, rectifier_current, tb_potential, tb_trend]
```

`tb_trend`는 현재 정상 TB와 직전 연속 정상 주기의 TB 차이다. 시작 직후 첫 정상 주기는
`warming_up`이며, 통신 오류나 TB 오류가 발생하면 이력을 폐기하고 다음 정상 주기부터 다시
준비한다. 오류 전 값을 연결해 trend를 만들지 않는다.

기존 5차원 SAC 입력의 `control_current_offset`은 환경모델 내부 누적값이므로 정책 입력에서
제외했다. 환경모델의 상태 전이 및 진단값에는 남겨 향후 실험 시 복구할 수 있다. 입력 차원이
달라졌으므로 기존 5차원 SAC 모델은 호환되지 않으며 4차원 입력으로 재학습해야 한다.
State Processor의 `normalized_vector`는 원시 처리 기록과 구분하기 위해 `null`로 두며,
AI Action 로그에 실제 모델 입력인 `normalized_state`를 기록한다.

## AI Shadow Mode

`junction_test.yaml`은 `sac-4state-v1` 메타데이터를 사용해 AI Shadow Mode를 실행한다.
시작할 때 모델 SHA-256, State 필드 순서, 입력 차원 `(4,)`, Action 차원 `(1,)`을 검증한다.
AI 모드의 첫 정상 주기는 TB trend 준비를 위해 Action을 만들지 않고, 두 번째 연속 정상
주기부터 결정론적 SAC 추론을 수행한다.

SAC의 `-1~+1` Action은 메타데이터의 `±0.05 V`로 변환한 다음 Safety Filter를 적용한다.
로그에는 정규화 State, 모델 원본 Action, 요청 변화량, 필터 적용 변화량과 목표 설정전압을
분리해 기록한다. `actual_write_voltage`는 항상 `null`이며 Shadow Mode는 Write를 호출하지 않는다.

실제 Write는 AI Shadow 로그와 Simulator 대조시험을 완료한 뒤 별도 단계에서 활성화한다.
현재 설정은 `write_enabled: false`, `ai.shadow_mode: true`이며 일반 실행기는 이 상태만 허용한다.
