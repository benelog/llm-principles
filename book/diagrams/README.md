# 다이어그램

원고에 들어가는 draw.io 다이어그램 모음이다. 각 PNG에는 draw.io의 다이어그램 XML이 메타데이터로 포함되어 있어서(편집 가능한 PNG), 별도의 소스 파일 없이 이 파일 하나로 관리한다.

## 편집 방법

1. [draw.io](https://app.diagrams.net) 또는 draw.io 데스크톱 앱에서 `.drawio.png` 파일을 그대로 연다. 도형과 텍스트가 편집 가능한 상태로 열린다.
2. 편집 후 같은 파일로 저장하면 이미지와 메타데이터가 함께 갱신된다.

CLI로 다시 내보낼 때는 다음 명령을 쓴다(메타데이터 포함 옵션 `--embed-diagram`이 핵심이다).

```bash
xvfb-run -a drawio --no-sandbox --export --format png --embed-diagram \
  --border 10 --scale 2 --output 파일명.drawio.png 파일명.drawio.png
```

## 목록

| 파일 | 대상 장 | 내용 |
|------|--------|------|
| `ch01-training-pipeline.drawio.png` | 1장 | 학습 파이프라인: 사전학습 → SFT → 선호 학습과 변형 경로(연속 사전학습, RLVR, 증류, 병합, 모델 편집) |
| `ch02-context-accumulation.drawio.png` | 2장 | 대화형 애플리케이션에서 턴마다 앞 턴의 프롬프트와 응답이 누적되는 구조와 프롬프트 캐싱 적중 구간 |
| `ch02-context-window.drawio.png` | 2장 | 컨텍스트 윈도우의 구성 요소와 프롬프트 캐싱을 고려한 배치 규칙 |
| `ch02-tool-loop.drawio.png` | 2장 | 도구 사용의 순환: 호출 생성 → 실행 → 결과를 토큰으로 추가 → 이어서 생성 |
| `ch03-layer-map.drawio.png` | 1~3장 | 정보 레이어 지도: 컨텍스트 공급 → 프롬프트 → 모델 실행 → 디코딩, 회색지대 포함 |
| `ch04-microgpt-flow.drawio.png` | 4장 | microGPT 예제의 전체 흐름: Main의 학습 루프와 생성 루프, Tokenizer의 encode·decode, GPT.forward 내부 단계, Value의 연산 기록과 backward()를 메서드 이름으로 연결 |
| `ch05-gpt-forward.drawio.png` | 5장 | GPT forward 계산: 임베딩 → 어텐션(Q·K·V) → MLP → 잔차 → 로짓 → 샘플링 루프 |
| `ch05-training-loop.drawio.png` | 5장 | 학습 루프 다섯 단계: 샘플 → forward → 손실 → 역전파 → 옵티마이저 |
| `ch06-prefill-decode.drawio.png` | 6장 | 프리필(병렬)과 디코드(순차)의 비대칭, KV 캐시 · 프롬프트 캐싱 · 배칭 |
| `ch07-vector-dot.drawio.png` | 7장 | 벡터와 내적: 2차원 벡터의 내적·코사인 계산과 사잇각에 따른 내적의 부호 |
| `ch07-low-rank.drawio.png` | 7장 | 저랭크 근사: d×d 행렬을 d×r과 r×d의 곱으로 대신할 때의 저장 숫자 비교, LoRA와의 연결 |
| `ch07-softmax-temperature.drawio.png` | 7장 | softmax와 temperature: 같은 로짓을 T = 0.5, 1, 2로 바꿨을 때의 확률 막대 그래프 |
| `ch07-loss-curve.drawio.png` | 7장 | 교차 엔트로피 손실 -ln p 곡선과 균등 추측 손실 ln 27 |
| `ch07-chain-rule.drawio.png` | 7장 | 체인 룰과 역전파: y = (2x+1)²의 계산 그래프에서 forward 값과 backward 기울기 |
| `ch07-gradient-descent.drawio.png` | 7장 | 경사 하강법과 학습률: 학습률 0.1의 수렴과 1.1의 발산 |
| `ch07-normal-grid.drawio.png` | 7장 | 정규 분포와 양자화 격자: 68%·95% 구간, 균등 격자와 분위수 격자(NF4) 비교 |
| `ch07-data-directions.drawio.png` | 7장 | 주성분 분석, 판별 분석, k-평균 군집화를 2차원 산점도로 비교 |
| `ch07-token-mdp.drawio.png` | 7장 | 토큰 생성을 마르코프 결정 과정으로: 상태·행동·보상·정책의 대응과 벨만 방정식 |
| `ch07-saturation-error.drawio.png` | 7장 | BM25의 빈도 포화 곡선과 평가셋 문항 수에 따른 표준 오차 곡선 |
| `ch08-model-repo.drawio.png` | 8~9장 | 모델 저장소 파일과 레이어 대응, safetensors 내부 구조 |
| `ch10-chat-template.drawio.png` | 10장 | 채팅 템플릿: 메시지 배열이 단일 토큰 열로 직렬화되는 과정과 불일치 장애 |
| `ch11-rag-pipeline.drawio.png` | 11장 | RAG 파이프라인: 인덱싱 시점과 질의 시점, 하이브리드 검색과 RRF · rerank |
| `ch13-diagnosis-flow.drawio.png` | 13장 | RAG 성능 진단 흐름: 평가셋 → 검색 실패 → 순위 실패 → 생성 실패, 개선 우선순위 |
| `ch14-agentic-rag.drawio.png` | 14장 | 고정 파이프라인과 에이전틱 검색의 대비 |
