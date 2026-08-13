package com.example.springai;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Component;

/**
 * RAG(검색 증강 생성) 예제.
 *
 * 문서 적재 -> 청킹 -> 임베딩 -> 벡터 저장 -> 유사도 검색 -> 프롬프트 조립 순서로 진행한다.
 * 임베딩은 로컬 ONNX 모델(all-MiniLM-L6-v2)을 사용하므로 API 키 없이 실행할 수 있고,
 * 마지막 답변 생성 단계만 로컬 Ollama가 있을 때 실행된다.
 */
@Component
public class RagExample {

    private static final List<String> QUERIES = List.of(
            "원격 근무는 일주일에 며칠까지 신청할 수 있나요?",
            "백업 데이터는 어디에 저장되나요?",
            "쓰지 않고 남은 연차는 내년으로 이월되나요?");

    private final EmbeddingModel embeddingModel;
    private final ChatClient.Builder chatClientBuilder;

    public RagExample(EmbeddingModel embeddingModel, ChatClient.Builder chatClientBuilder) {
        this.embeddingModel = embeddingModel;
        this.chatClientBuilder = chatClientBuilder;
    }

    public void run() throws Exception {
        System.out.println("===== RAG 예제 시작 =====");

        // 1단계: resources/docs 의 텍스트 파일을 Document로 적재
        List<Document> documents = loadDocuments();
        System.out.printf("문서 %d건을 읽었습니다.%n", documents.size());

        // tag::chunking[]
        // 2단계: 문서를 토큰 수 기준으로 청킹
        // 청크당 최대 250토큰, 청크 최소 100자를 기준으로 나눈다
        TokenTextSplitter splitter = new TokenTextSplitter(250, 100, 10, 5000, true);
        List<Document> chunks = splitter.apply(documents);
        // end::chunking[]
        System.out.printf("문서 %d건을 청크 %d개로 나누었습니다.%n", documents.size(), chunks.size());

        // tag::embedding-store[]
        // 3단계: 청크를 임베딩해서 벡터 스토어에 저장
        // add() 호출 시 EmbeddingModel(로컬 ONNX 모델)이 각 청크를 벡터로 변환한다
        SimpleVectorStore vectorStore = SimpleVectorStore.builder(embeddingModel).build();
        vectorStore.add(chunks);
        // end::embedding-store[]
        System.out.printf("청크 %d개의 임베딩을 벡터 스토어에 저장했습니다.%n", chunks.size());

        boolean ollamaAvailable = OllamaStatus.available();
        if (!ollamaAvailable) {
            System.out.println("[안내] localhost:11434 에서 Ollama 응답이 없습니다. 검색과 프롬프트 조립까지만 실행합니다.");
        }

        for (String query : QUERIES) {
            System.out.println();
            System.out.println("---------------------------------------------");
            System.out.println("질의: " + query);

            // tag::retrieval[]
            // 4단계: 질의를 임베딩해서 유사도가 높은 청크 상위 3개를 검색
            List<Document> results = vectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query(query)
                            .topK(3)
                            .build());
            // end::retrieval[]

            System.out.println("[검색 결과]");
            for (Document result : results) {
                System.out.printf("  - 유사도 %.4f | 출처 %s%n", result.getScore(),
                        result.getMetadata().get("source"));
                System.out.printf("    내용: %s%n", preview(result.getText(), 80));
            }

            // tag::prompt-assembly[]
            // 5단계: 검색된 청크를 컨텍스트로 붙여 최종 프롬프트를 조립
            String context = results.stream()
                    .map(Document::getText)
                    .collect(Collectors.joining("\n---\n"));
            String prompt = """
                    당신은 사내 문서를 근거로 답하는 도우미입니다.
                    아래 컨텍스트에 있는 내용만 사용해서 질문에 답하세요.
                    컨텍스트에 없는 내용이면 모른다고 답하세요.

                    [컨텍스트]
                    %s

                    [질문]
                    %s
                    """.formatted(context, query);
            // end::prompt-assembly[]

            System.out.println("[조립된 최종 프롬프트]");
            System.out.println(indent(prompt));

            // tag::generation[]
            // 6단계: 조립한 프롬프트로 답변 생성 (로컬 Ollama가 있을 때만 실행)
            if (ollamaAvailable) {
                String answer = chatClientBuilder.build()
                        .prompt()
                        .user(prompt)
                        .call()
                        .content();
                System.out.println("[생성된 답변]");
                System.out.println(indent(answer));
            }
            else {
                System.out.println("[안내] Ollama 미실행, 생성 단계 생략");
            }
            // end::generation[]
        }
        System.out.println();
        System.out.println("===== RAG 예제 종료 =====");
    }

    /** classpath:docs 아래의 .txt 파일을 읽어 Document 목록으로 만든다. */
    private List<Document> loadDocuments() throws Exception {
        var resolver = new PathMatchingResourcePatternResolver();
        List<Document> documents = new ArrayList<>();
        for (Resource resource : resolver.getResources("classpath:docs/*.txt")) {
            String text = resource.getContentAsString(StandardCharsets.UTF_8);
            documents.add(new Document(text, Map.of("source", resource.getFilename())));
        }
        return documents;
    }

    private static String preview(String text, int maxLength) {
        String singleLine = text.replaceAll("\\s+", " ").strip();
        return singleLine.length() <= maxLength ? singleLine : singleLine.substring(0, maxLength) + "...";
    }

    private static String indent(String text) {
        return text.lines().map(line -> "    " + line).collect(Collectors.joining(System.lineSeparator()));
    }
}
