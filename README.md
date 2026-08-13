# 애플리케이션 개발자를 위한 LLM의 원리

동작 원리부터 오픈 모델 표준, RAG까지, LLM을 쓰는 시스템을 만드는 개발자에게 필요한 원리를 정리한 책의 원고와 실행 가능한 예제 프로젝트입니다.

**📖 책 읽기: https://benelog.github.io/llm-principles/**

main 브랜치에 push하면 GitHub Actions가 원고를 빌드해서 위 주소로 자동 배포합니다.

## 구성

```
book/                  AsciiDoc 원고 (4개 부, 13개 장)
  book.adoc            책 전체 (부 구성과 장별 파일 include)
  ch01-model-layer.adoc            1부 1장. 모델 레이어: 가중치에 새기는 정보
  ch02-prompt-layer.adoc           1부 2장. 프롬프트 레이어: 추론 시점에 넣는 정보
  ch03-gray-zone.adoc              1부 3장. 회색지대와 컨텍스트 공급 레이어
  ch04-microgpt-basics.adoc        2부 4장. 다음 토큰 예측과 자동 미분
  ch05-microgpt-model.adoc         2부 5장. GPT의 구조와 학습, 추론
  ch06-inference-serving.adoc      2부 6장. 추론 서빙의 원리: 토큰은 왜 그 가격인가
  ch07-weight-formats.adoc         3부 7장. 가중치 저장 포맷
  ch08-architecture-conventions.adoc  3부 8장. 아키텍처 규약과 추론 엔진
  ch09-tokenizer-chat-template.adoc   3부 9장. 토크나이저와 채팅 템플릿
  ch10-rag-pipeline.adoc           4부 10장. RAG 파이프라인과 검색의 원리
  ch11-korean-rag.adoc             4부 11장. 한국어 처리와 실무 구성
  ch12-rag-tuning.adoc             4부 12장. 성능 진단과 개선 우선순위
  ch13-rag-frontier.adoc           4부 13장. RAG의 경계: 긴 컨텍스트와 에이전틱 검색
examples/
  go/microgpt/         microGPT의 Go 포팅 (자동 미분, GPT 학습과 추론)
  go/serving/          프리필·디코드와 KV 캐시의 추론 비용 구조 측정
  go/safetensors/      safetensors 파일을 만들고 읽는 파서
  go/bpe/              BPE 토크나이저의 병합 학습과 인코딩
  go/minirag/          청킹, BM25, 벡터 검색, RRF 하이브리드 검색
  spring-ai/           Spring AI RAG 파이프라인과 디코딩 파라미터 제어
```

원고의 코드는 전부 `examples/` 아래 소스를 AsciiDoc의 `include` 태그 문법으로 인용합니다. 책의 코드와 예제 프로젝트의 코드는 항상 같습니다.

## 책 빌드

```bash
cd book
asciidoctor book.adoc          # book.html 생성
```

문법 강조까지 보려면 rouge를 설치합니다: `gem install rouge`

## 예제 실행

Go 예제는 표준 라이브러리만 사용하므로 Go만 있으면 됩니다.

```bash
cd examples/go/microgpt && go run .      # 이름 생성 GPT 학습 (약 20초)
cd examples/go/serving && go run .       # KV 캐시·프리필 비용 측정 (약 10초)
cd examples/go/safetensors && go run .   # safetensors 쓰기/읽기
cd examples/go/bpe && go run .           # BPE 병합 학습과 인코딩
cd examples/go/minirag && go run .       # 하이브리드 검색 파이프라인
```

Spring AI 예제는 JDK 21 이상과 Maven이 필요하며, 임베딩 모델이 라이브러리에 내장되어 있어 API 키 없이 실행됩니다.

```bash
cd examples/spring-ai
mvn -q package
java -jar target/spring-ai-example-0.0.1-SNAPSHOT.jar --mode=rag       # RAG 검색
java -jar target/spring-ai-example-0.0.1-SNAPSHOT.jar --mode=decoding  # 디코딩 파라미터
```

로컬에 Ollama가 있으면 검색 이후의 답변 생성 단계까지 실행됩니다. 없으면 검색과 프롬프트 조립까지만 실행하고 안내 메시지를 출력합니다.

## 원고의 바탕이 된 글

- [Hugging Face 오픈 모델 호환성 표준](https://blog.benelog.net/huggingface-open-model-standards)
- [LLM에 정보를 전달하는 계층 분류와 회색지대](https://blog.benelog.net/llm-information-layers)
- [RAG 시스템의 구성요소와 성능을 좌우하는 요소](https://blog.benelog.net/rag-system-components)
- [microGPT (Andrej Karpathy)](https://karpathy.github.io/2026/02/12/microgpt/)
