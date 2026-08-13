package com.example.springai;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * 로컬 Ollama 서버의 동작 여부를 짧은 타임아웃으로 확인하는 도우미 클래스.
 * Ollama가 없어도 애플리케이션이 시작에 실패하지 않도록,
 * 채팅 모델 호출 전에 이 확인을 먼저 거친다.
 */
public final class OllamaStatus {

    private static final String TAGS_URL = "http://localhost:11434/api/tags";

    private OllamaStatus() {
    }

    /** localhost:11434 의 Ollama가 응답하면 true를 반환한다. */
    public static boolean available() {
        try (HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(1))
                .build()) {
            HttpRequest request = HttpRequest.newBuilder(URI.create(TAGS_URL))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();
            HttpResponse<Void> response = client.send(request, HttpResponse.BodyHandlers.discarding());
            return response.statusCode() == 200;
        }
        catch (Exception ex) {
            return false;
        }
    }
}
