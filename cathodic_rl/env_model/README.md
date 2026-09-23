# 환경모델

환경모델은 현장 데이터를 바탕으로 학습 환경의 다음 상태를 예측한다. 실제 장비를 제어하지 않는다.

```text
현재 V/I + Delta_V
    → CurrentModel → 자연 전류 변화 + 제어 전류 변화
TB History + 제어 전류 변화
    → TBModel → 자연 TB + 제어 TB 보정
두 결과 → EnvironmentModel → 다음 V/I/TB
```

| 파일 | 역할 |
|---|---|
| `preprocessing.py` | 원본 시계열을 CurrentModel·TBModel 학습 표와 reset 후보로 전처리 |
| `current_model.py` | 현재 전압·전류와 `Delta_V`에서 다음 전류를 예측 |
| `tb_model.py` | 최근 TB History와 제어 전류 변화에서 다음 TB를 예측 |
| `environment_model.py` | 두 예측기를 결합해 하나의 Step 전이를 계산하고 TB History를 관리 |
| `train_current_model.py`, `train_tb_model.py` | joblib 예측 모델 재학습 |
| `tests/` | 현재 수동 실행 진단 스크립트. 파일별 의미는 [테스트 안내](../../docs/TEST_GUIDE.md) 참고 |

예측 모델이 학습된 전압 범위 밖에서 동작하는 경우에는 물리적 외삽이 아니라 설정된 모델 유효범위 제한이 적용될 수 있다.
