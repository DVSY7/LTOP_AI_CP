# Linux edge 배포

## 배포 방식

이 프로그램은 Linux에서 Python 소스와 가상환경으로 실행한다. Windows의 `.exe` 패키징은 필요하지 않다.
`stable-baselines3` 모델을 읽으므로 edge 장비의 가상환경에는 Python 의존성을 설치해야 한다.

Linux 표준 압축 형식은 `tar.gz`다. 

```text
AI_CP/
├─ edge_control/src/
├─ edge_control/configs/
├─ edge_control/deploy/linux/
├─ edge_control/requirements.txt
├─ edge_control/__init__.py
├─ shared/control_core/
├─ shared/__init__.py
├─ cathodic_rl/models/trained/
└─ cathodic_rl/__init__.py
```

학습 환경, 원본 데이터, notebook, 분석 스크립트, 테스트는 edge 실행에 필요 없다.

## 1. Windows에서 배포 파일 만들기

AI_CP 루트 PowerShell에서 실행한다.

```powershell
tar -czf ai_cp-edge.tar.gz `
  edge_control/src edge_control/configs edge_control/deploy `
  edge_control/requirements.txt edge_control/__init__.py `
  shared/control_core shared/__init__.py `
  cathodic_rl/models cathodic_rl/__init__.py
```

이 파일을 edge 장비의 임시 경로로 복사한다.

## 2. Linux에서 설치

아래 예시는 `/opt/ai_cp`에 설치하고 `edgecontrol` 사용자로 실행하는 경우다.

```bash
sudo mkdir -p /opt/ai_cp
sudo tar -xzf ai_cp-edge.tar.gz -C /opt/ai_cp
sudo useradd --system --create-home --shell /usr/sbin/nologin edgecontrol
sudo chown -R edgecontrol:edgecontrol /opt/ai_cp

cd /opt/ai_cp
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r edge_control/requirements.txt
```

`python3 -m venv`가 없다면 배포판의 `python3-venv` 패키지를 먼저 설치한다.

## 3. 부팅 서비스 등록 전 수동 시험

`junction_test.yaml`의 현재 값은 자동 Write가 활성화되어 있다. 연결·로그 시험만 한다면 먼저 아래처럼 설정한다.

```yaml
write_enabled: false
ai:
  shadow_mode: true
control:
  enabled: false
```

설정 검증과 짧은 수동 실행은 다음과 같다.

```bash
cd /opt/ai_cp
.venv/bin/python -m edge_control.src.main \
  --config edge_control/configs/junction_test.yaml --check-config

.venv/bin/python -m edge_control.src.main \
  --config edge_control/configs/junction_test.yaml --max-cycles 3
```

로그는 `edge_control/logs/junction_test.jsonl`에 남는다. 수동 시험에서 레지스터 값·Guard·로그를 확인한 뒤에만 Write 관련 YAML 값을 다시 활성화한다.

## 4. systemd 자동 실행 등록

수동 시험이 끝난 뒤에만 서비스 파일을 설치한다.

```bash
cd /opt/ai_cp
sudo install -m 0755 edge_control/deploy/linux/run-edge-control.sh \
  /opt/ai_cp/edge_control/deploy/linux/run-edge-control.sh
sudo install -m 0644 edge_control/deploy/linux/edge-control.service \
  /etc/systemd/system/edge-control.service
sudo install -m 0644 edge_control/deploy/linux/edge-control.env.example \
  /etc/edge-control.env

sudo systemctl daemon-reload
sudo systemctl enable --now edge-control.service
sudo systemctl status edge-control.service
```

실시간 로그는 다음으로 본다.

```bash
sudo journalctl -u edge-control.service -f
```

서비스를 잠시 중지하려면 다음을 사용한다.

```bash
sudo systemctl stop edge-control.service
```

`edge-control.service`의 `User`, `Group`, `WorkingDirectory`, `ExecStart`는 실제 Linux 계정과 설치 경로가 다르면 바꿔야 한다.
