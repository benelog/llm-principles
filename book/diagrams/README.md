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
| `ch00-generation-loop.drawio.png` | 도입 장 | 다음 토큰 예측의 반복: 생성한 토큰을 입력 끝에 붙여 모델을 다시 실행하고 종료 토큰에서 멈춤, temperature가 개입하는 곳 |
| `ch01-training-pipeline.drawio.png` | 1장 | 학습 파이프라인: 사전학습 → SFT → 선호 학습과 변형 경로(연속 사전학습, RLVR, 증류, 병합, 모델 편집) |
| `ch02-context-accumulation.drawio.png` | 2장 | 대화형 애플리케이션에서 턴마다 앞 턴의 프롬프트와 응답이 누적되는 구조와 프롬프트 캐싱 적중 구간 |
| `ch02-context-window.drawio.png` | 2장 | 컨텍스트 윈도우의 구성 요소와 프롬프트 캐싱을 고려한 배치 규칙 |
| `ch02-tool-loop.drawio.png` | 2장 | 도구 사용의 순환: 호출 생성 → 실행 → 결과를 토큰으로 추가 → 이어서 생성 |
| `ch03-layer-map.drawio.png` | 1~3장 | 정보 레이어 지도: 컨텍스트 공급 → 프롬프트 → 모델 실행 → 디코딩, 회색지대 포함 |
| `ch04-microgpt-flow.drawio.png` | 4장 | microGPT 예제의 전체 흐름: Main의 학습 루프와 생성 루프, Tokenizer의 encode·decode, Gpt.forward 내부 단계, Value의 연산 기록과 backward()를 메서드 이름으로 연결 |
| `ch05-attention.drawio.png` | 5장 | 한 위치의 어텐션: 쿼리와 캐시의 키로 점수(scores)를 매기고 softmax 가중치(weights)로 밸류를 섞는 과정, causal 구조 |
| `ch05-gpt-forward.drawio.png` | 5장 | GPT forward 계산: 임베딩 → 어텐션(Q·K·V) → MLP → 잔차 → 로짓 → 샘플링 루프 |
| `ch05-training-loop.drawio.png` | 5장 | 학습 루프 다섯 단계: 샘플 → forward → 손실 → 역전파 → 옵티마이저 |
| `ch06-batching.drawio.png` | 6장 | 정적 배칭과 연속 배칭의 자리 사용 비교, 연속 예약과 PagedAttention 블록 할당의 메모리 비교 |
| `ch06-gqa.drawio.png` | 6장 | MHA·GQA·MQA의 K·V 헤드 공유와 128K 컨텍스트의 KV 캐시 크기(64·16·2GiB) |
| `ch06-prefill-decode.drawio.png` | 6장 | 프리필(병렬)과 디코드(순차)의 비대칭, KV 캐시 · 프롬프트 캐싱 · 배칭 |
| `ch06-speculative-decoding.drawio.png` | 6장 | 투기적 디코딩: 초안 모델의 순차 생성, 큰 모델의 병렬 검증, 채택·기각과 고친 토큰 |
| `ch07-chain-rule.drawio.png` | 7장 | 체인 룰과 역전파: y = (2x+1)²의 계산 그래프에서 forward 값과 backward 기울기 |
| `ch07-conditioning-levels.drawio.png` | 7장 | 조건 유지의 정도: LLM(모든 앞 토큰), n-gram(직전 두 토큰), 나이브 베이즈(순서 없는 단어 집합) |
| `ch07-data-directions.drawio.png` | 7장 | 주성분 분석, 판별 분석, k-평균 군집화를 2차원 산점도로 비교 |
| `ch07-eigenvector.drawio.png` | 7장 | 고유벡터: [[2, 1], [1, 2]]가 단위 벡터 8개를 변환할 때 방향을 유지하는 두 방향 |
| `ch07-entropy-perplexity.drawio.png` | 7장 | 엔트로피와 perplexity: 확신하는 분포, 예제 분포, 균등 분포의 비교 |
| `ch07-gradient-contour.drawio.png` | 7장 | 그래디언트와 등고선: x² + 4y²의 타원 등고선 위의 그래디언트 화살표와 경사 하강 경로의 지그재그 |
| `ch07-gradient-descent.drawio.png` | 7장 | 경사 하강법과 학습률: 학습률 0.1의 수렴과 1.1의 발산 |
| `ch07-kl-penalty.drawio.png` | 7장 | KL 발산의 항별 기여와 비대칭, RLHF의 보상 − β·KL 목적 함수 |
| `ch07-loss-curve.drawio.png` | 7장 | 교차 엔트로피 손실 -ln p 곡선과 균등 추측 손실 ln 27 |
| `ch07-low-rank.drawio.png` | 7장 | 저랭크 근사: d×d 행렬을 d×r과 r×d의 곱으로 대신할 때의 저장 숫자 비교, LoRA와의 연결 |
| `ch07-markov-bigram.drawio.png` | 7장 | 마르코프 체인: 이름 데이터에서 센 글자 a의 전이 확률 그래프와 LLM의 상태(토큰 열 전체) |
| `ch07-nonlinearity.drawio.png` | 7장 | 비선형 함수의 역할: 행렬 곱 두 번은 직선 하나, 사이에 tanh를 끼우면 곡선 |
| `ch07-normal-grid.drawio.png` | 7장 | 정규 분포와 양자화 격자: 68%·95% 구간, 균등 격자와 분위수 격자(NF4) 비교 |
| `ch07-prefill-decode-matmul.drawio.png` | 7장 | 프리필의 행렬×행렬(Q Kᵀ)과 디코드의 벡터×행렬(q Kᵀ) 비교, 6장 병목 비대칭과의 연결 |
| `ch07-sampling-cutoff.drawio.png` | 7장 | 샘플링과 후보 제한: 누적 확률 막대 위의 난수, top-k와 top-p의 꼬리 자르기와 재정규화 |
| `ch07-saturation-error.drawio.png` | 7장 | BM25의 빈도 포화 곡선과 평가셋 문항 수에 따른 표준 오차 곡선 |
| `ch07-scaling-law.drawio.png` | 7장 | 스케일링 법칙: 보통 축과 로그-로그 축의 초과 손실 곡선, 작은 모델에서 큰 모델로의 외삽 |
| `ch07-softmax-temperature.drawio.png` | 7장 | softmax와 temperature: 같은 로짓을 T = 0.5, 1, 2로 바꿨을 때의 확률 막대 그래프 |
| `ch07-token-mdp.drawio.png` | 7장 | 토큰 생성을 마르코프 결정 과정으로: 상태·행동·보상·정책의 대응과 벨만 방정식 |
| `ch07-ttest-overlap.drawio.png` | 7장 | 독립표본과 대응표본: 79%와 83% 분포의 겹침, 질문별 점수 차이가 일정한 대응표본 |
| `ch07-variance-norm.drawio.png` | 7장 | 값의 규모 관리: 블록을 지날수록 폭주·소멸하는 표준편차와 RMSNorm, sqrt(d) 스케일링 전후의 점수 분포 |
| `ch07-vector-dot.drawio.png` | 7장 | 벡터와 내적: 2차원 벡터의 내적·코사인 계산과 사잇각에 따른 내적의 부호 |
| `ch08-block-quantization.drawio.png` | 8장 | 아웃라이어 하나와 4비트 양자화: 텐서 전체 배율과 블록마다 배율의 정수·복원 값 비교 |
| `ch08-model-repo.drawio.png` | 8~9장 | 모델 저장소 파일과 레이어 대응, safetensors 내부 구조 |
| `ch09-model-loading.drawio.png` | 9장 | config.json으로 모델 조립: model_type → 구현 클래스 → 빈 구조 → 텐서 이름으로 채우기, 직접 읽는 엔진과 변환 경로 |
| `ch10-bpe-merge.drawio.png` | 10장 | BPE 병합 규칙의 학습(예제 코퍼스의 12개 규칙)과 새 텍스트의 인코딩 |
| `ch10-chat-template.drawio.png` | 10장 | 채팅 템플릿: 메시지 배열이 단일 토큰 열로 직렬화되는 과정과 불일치 장애 |
| `ch11-rag-pipeline.drawio.png` | 11장 | RAG 파이프라인: 인덱싱 시점과 질의 시점, 하이브리드 검색과 RRF · rerank |
| `ch11-rrf.drawio.png` | 11장 | Reciprocal Rank Fusion: BM25와 벡터 검색 순위를 1/(k+순위)로 합산하는 계산(k = 60) |
| `ch12-hierarchical-chunking.drawio.png` | 12장 | 계층적 청킹: 작은 청크로 검색하고 그 청크가 속한 절을 LLM에 넘기는 구조(휴가 규정 예제) |
| `ch12-reranker.drawio.png` | 12장 | 바이 인코더와 크로스 인코더의 비교, 1차 검색 후 재순위로 후보를 좁히는 두 단계 검색 |
| `ch13-diagnosis-flow.drawio.png` | 13장 | RAG 성능 진단 흐름: 평가셋 → 검색 실패 → 순위 실패 → 생성 실패, 개선 우선순위 |
| `ch14-agentic-rag.drawio.png` | 14장 | 고정 파이프라인과 에이전틱 검색의 대비 |
