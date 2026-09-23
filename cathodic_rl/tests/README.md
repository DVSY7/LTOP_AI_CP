# cathodic_rl 자동 테스트

환경 규약, 모델 생성, 정책 게이트, Reward 방향을 자동 검증한다. 각 파일의 의미는 [테스트 안내](../../docs/TEST_GUIDE.md)에 있다.

```bat
cathodic_rl\.venv\Scripts\python.exe -m unittest discover -s cathodic_rl/tests -v
```
