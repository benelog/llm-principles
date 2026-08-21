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

## 글쓰기 지침

한국어 문체·표기 지침과 AsciiDoc 공통 마크업 규칙은 `korean-tech-writing` 플러그인의 두 스킬(`korean-tech-writing`, `korean-asciidoc`)을 따른다. 아래는 이 저장소에만 해당하는 항목이다.

- **적용 범위**: 원고뿐 아니라 README와 주석에도 적용한다.
- **어조**: 문어체 평서형("~다")으로 쓴다. 경어체와 섞지 않는다.
- **마크업 검증**: `cd book && asciidoctor book.adoc`으로 빌드한 뒤 생성된 HTML 본문에 `*`나 백틱이 그대로 남아 있는지 확인한다.
- **"층"**: 신경망의 층(layer)처럼 실체가 있을 때는 그대로 쓴다. 추상적 구분에 쓰는 "층위"만 스킬 지침대로 실제 차이 기준으로 바꾼다.
- **은유 검사에서 제외하는 용어**: 표면 문자열, 회색지대, RAG의 경계. 이 책에서 기술 용어로 굳은 표현이다.
- **의도적으로 남기는 비유**: 아래 두 문장은 교정 대상이 아니다.
  - `book.adoc`의 "정보 전달 방법의 전체 지도를 그린다". 책 전체를 관통하도록 쓴 비유다. 단, 여기에 다른 은유를 겹쳐 쓰지는 않는다("그 지도의 밑바탕인" → "그 분류의 바탕인").
  - `ch01-model-layer.adoc`의 "코퍼스의 분포가 곧 모델의 세계관이 된다". 바로 다음 문장에서 구체적 서술로 풀린다.
