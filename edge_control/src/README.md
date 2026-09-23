# edge_control 실행 코드

실행은 `python -m edge_control.src.main --config ...`로 시작한다. 전체 데이터 흐름은 [코드 구조와 읽는 순서](../../docs/ARCHITECTURE.md)를 참고한다.

| 위치 | 역할 |
|---|---|
| `main.py` | 한 제어 주기 실행, 모드 분기, 로그 기록, Write 간격 관리 |
| `config_loader.py` | YAML을 검증된 `Config` 객체로 변환 |
| `communication/` | Modbus TCP 읽기·재연결·Write와 Read-back |
| `controller/` | Guard, EXPLORE 경로, SAC Action, 전압 경계 처리 |
| `state/` | 4차원 정책 State와 TB trend 생성 |
| `logging_setup.py` | 콘솔과 JSONL 회전 로그 설정 |

장비별 주소나 제한값은 이 폴더의 코드에 넣지 않고 `edge_control/configs`에서 변경한다.
