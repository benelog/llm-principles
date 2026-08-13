# Spring AI RAG / 디코딩 파라미터 예제

Spring AI로 RAG(검색 증강 생성) 파이프라인과 디코딩 파라미터(temperature, top-p) 제어를 실습하는 콘솔 예제입니다.

- Spring Boot 3.5.16 + Spring AI 1.0.9 (spring-ai-bom)
- 임베딩: `spring-ai-starter-model-transformers` 의 로컬 ONNX 모델(all-MiniLM-L6-v2, jar에 내장). **API 키와 네트워크 연결이 필요 없습니다.**
- 채팅 생성: `spring-ai-starter-model-ollama`. 로컬에 Ollama가 있을 때만 실행되는 선택 단계입니다.

## 빌드

JDK 21 이상(권장 21~25)과 Maven이 필요합니다.

```bash
mvn -q package
```

## 실행

### RAG 예제

문서 적재 -> 청킹(TokenTextSplitter) -> 임베딩(TransformersEmbeddingModel) -> 벡터 저장(SimpleVectorStore) -> 유사도 검색 -> 프롬프트 조립 순서로 진행합니다.

```bash
java -jar target/spring-ai-example-0.0.1-SNAPSHOT.jar --mode=rag
# 또는
mvn spring-boot:run -Dspring-boot.run.arguments=--mode=rag
```

질의 3개에 대해 top-3 검색 결과(유사도 점수, 출처 파일, 내용 미리보기)와 컨텍스트를 붙인 최종 프롬프트를 출력합니다.

### 디코딩 파라미터 예제

```bash
java -jar target/spring-ai-example-0.0.1-SNAPSHOT.jar --mode=decoding
```

temperature 0.0(보수적)과 1.2(다양성)로 구성한 `OllamaOptions` 두 가지를 만들어 비교합니다.

## Ollama를 설치하면 추가로 되는 것

Ollama 없이도 위 예제의 검색 단계까지는 모두 동작합니다. Ollama가 `localhost:11434` 에서 응답하면 다음 단계가 추가로 실행됩니다.

- RAG 예제: 조립된 프롬프트를 채팅 모델에 보내 실제 답변을 생성합니다.
- 디코딩 예제: 같은 프롬프트를 temperature 0.0과 1.2로 각각 호출해 출력 차이를 비교합니다.

준비 방법:

```bash
# https://ollama.com 에서 설치 후
ollama pull llama3.2
```

사용할 모델 이름은 `src/main/resources/application.properties` 의 `spring.ai.ollama.chat.options.model` 로 바꿀 수 있습니다. Ollama가 없으면 해당 단계를 건너뛰고 안내 메시지만 출력하며, 애플리케이션이 실패하지 않습니다.

## 파일 구성

```
pom.xml                                  # Spring Boot 3.5.16, spring-ai-bom 1.0.9
src/main/java/com/example/springai/
  SpringAiExampleApplication.java        # --mode 인자로 예제를 선택하는 콘솔 앱
  RagExample.java                        # RAG 파이프라인 예제
  DecodingExample.java                   # temperature, top-p 비교 예제
  OllamaStatus.java                      # Ollama 동작 여부를 짧은 타임아웃으로 확인
src/main/resources/
  application.properties                 # 모델 선택, 지연 초기화 등 설정
  docs/
    remote-work-policy.txt               # 샘플 문서: 원격 근무 규정
    vacation-policy.txt                  # 샘플 문서: 연차 휴가 지침
    cloud-backup-faq.txt                 # 샘플 문서: 클라우드 백업 FAQ
    security-guideline.txt               # 샘플 문서: 정보 보안 지침
```

## 원고 인용용 AsciiDoc 태그

| 파일 | 태그 |
|---|---|
| `RagExample.java` | `chunking`, `embedding-store`, `retrieval`, `prompt-assembly`, `generation` |
| `DecodingExample.java` | `chat-options` |
