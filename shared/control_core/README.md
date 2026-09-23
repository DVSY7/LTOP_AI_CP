# 학습·edge 공통 제어 규약

이 폴더의 코드는 학습과 edge에서 같은 정의를 써야 하는 부분만 둔다. 한쪽에 복사해 수정하면 모델 입력과 현장 입력이 달라질 수 있다.

| 파일 | 역할 |
|---|---|
| `state.py` | 기본 측정 State와 SAC 정책 State의 필드 순서·단위 |
| `preprocessing.py` | 정책 State를 모델 입력 범위 `[0, 1]`로 정규화 |
| `safety_filter.py` | 설정전압 범위와 주기당 변화량을 제한하고 개입 사유를 반환 |
| `model_loader.py` | 모델 메타데이터, State/Action 계약, SHA-256 검증 |

이 규약의 자동 검증은 `shared/tests`에 있으며 설명은 [테스트 안내](../../docs/TEST_GUIDE.md)에 있다.
