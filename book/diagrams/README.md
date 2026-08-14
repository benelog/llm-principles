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
| `ch02-context-window.drawio.png` | 2장 | 컨텍스트 윈도우의 구성 요소와 프롬프트 캐싱을 고려한 배치 규칙 |
| `ch02-tool-loop.drawio.png` | 2장 | 도구 사용의 순환: 호출 생성 → 실행 → 결과를 토큰으로 추가 → 이어서 생성 |
| `ch03-layer-map.drawio.png` | 1~3장 | 정보 레이어 지도: 컨텍스트 공급 → 프롬프트 → 모델 실행 → 디코딩, 회색지대 포함 |
| `ch05-gpt-forward.drawio.png` | 5장 | GPT forward 계산: 임베딩 → 어텐션(Q·K·V) → MLP → 잔차 → 로짓 → 샘플링 루프 |
| `ch05-training-loop.drawio.png` | 5장 | 학습 루프 다섯 단계: 샘플 → forward → 손실 → 역전파 → 옵티마이저 |
| `ch06-prefill-decode.drawio.png` | 6장 | 프리필(병렬)과 디코드(순차)의 비대칭, KV 캐시 · 프롬프트 캐싱 · 배칭 |
| `ch07-model-repo.drawio.png` | 7~8장 | 모델 저장소 파일과 레이어 대응, safetensors 내부 구조 |
| `ch09-chat-template.drawio.png` | 9장 | 채팅 템플릿: 메시지 배열이 단일 토큰 열로 직렬화되는 과정과 불일치 장애 |
| `ch10-rag-pipeline.drawio.png` | 10장 | RAG 파이프라인: 인덱싱 시점과 질의 시점, 하이브리드 검색과 RRF · rerank |
| `ch12-diagnosis-flow.drawio.png` | 12장 | RAG 성능 진단 흐름: 평가셋 → 검색 실패 → 순위 실패 → 생성 실패, 개선 우선순위 |
| `ch13-agentic-rag.drawio.png` | 13장 | 고정 파이프라인과 에이전틱 검색의 대비 |
