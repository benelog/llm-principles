# 프로젝트 개요

이 저장소는 "애플리케이션 개발자를 위한 LLM의 원리"(부제: 동작 원리부터 오픈 모델 표준, RAG까지) 책 프로젝트다 (2026-08-14 시작). GitHub 저장소는 `benelog/llm-principles`(SSH: git@github.com:benelog/llm-principles.git, main 브랜치)다. 기존 `llm-principle`(단수) 저장소를 rename해서 만들었다.

## 원고 구조

- `book/`에 AsciiDoc 원고가 도입 장(ch00, 번호 없는 장)과 4개 부, 13개 장으로 구성된다. ch00은 book.adoc에서 `:sectnums!:`로 감싸 번호 없이 include되므로 "N장" 번호에 포함되지 않는다.
  - 1부 정보 레이어: ch01 모델 레이어, ch02 프롬프트 레이어, ch03 회색지대
  - 2부 microGPT와 동작 원리: ch04, ch05, ch06 추론 서빙의 원리
  - 3부 오픈 모델 표준: ch07 가중치 포맷, ch08 아키텍처 규약, ch09 토크나이저·채팅 템플릿
  - 4부 RAG: ch10 파이프라인, ch11 한국어, ch12 진단, ch13 RAG의 경계
- 본문의 장 상호 참조는 "N장" 텍스트로 쓰므로 장 추가·삭제 시 번호를 일괄 갱신해야 한다.
- 부 구성과 partintro는 `book.adoc`에 있고, 장 파일은 `= 제목`으로 시작해 leveloffset=+1로 include된다.
- `book/book.html`은 gitignore된 로컬 빌드 산출물이고 배포는 GitHub Actions가 한다.
- 블로그 출처 표기는 본문에 넣지 않는다.
- 원고는 사용자의 블로그 글 3편(huggingface-open-model-standards, llm-information-layers, rag-system-components)과 Karpathy의 microGPT 포스트를 바탕으로 한다.

## 예제 코드

- `examples/go/`: microgpt 포팅, serving 추론 비용 측정, safetensors 파서, quantize 양자화 오차 측정, bpe 토크나이저, minirag 하이브리드 검색. 순수 표준 라이브러리만 사용.
- `examples/spring-ai/`: Spring Boot 3.5 + Spring AI 1.0.9. 내장 ONNX 임베딩으로 API 키 없이 실행 가능하고 Ollama는 선택.
- 원고의 코드는 반드시 `include::../examples/...[tag=...,indent=0]` 태그 문법으로 예제 소스를 인용한다(코드 복사 금지).
- 예제 코드에는 `// tag::이름[]` 주석이 있으니 수정 시 태그를 깨뜨리면 안 된다. 수정 후 `cd book && asciidoctor book.adoc`으로 include 해석을 검증한다.

## 한국어 문체 지침

한국어 원고나 문서(README, 주석 포함)를 쓸 때 https://gist.github.com/benelog/5073ac36d8f879873139e3edb7d9d3a8 의 "클로드 문체 없애기" 지침을 따른다. 금지/제한 표현:

- 줄표(—)는 괄호나 콜론으로 대체
- "무너진다", "맞물린다", "가른다/갈린다"는 구체적 서술로 대체
- "가볍다/무겁다"는 메모리·크기·시간 등 기준 명시
- "지점", "축", "겹", "손잡이"는 물리적 실체에만 사용
- "사실상"은 굳은 용어(예: 사실상 표준)에만 사용
- "~인 셈이다"는 비유 마무리에만 사용

한국어 산문을 쓴 뒤 위 표현들을 grep으로 검사하고 대체한다.
