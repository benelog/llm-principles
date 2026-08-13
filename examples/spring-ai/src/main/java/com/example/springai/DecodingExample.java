package com.example.springai;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.stereotype.Component;

/**
 * 디코딩 파라미터(temperature, top-p) 예제.
 *
 * 같은 프롬프트에 서로 다른 디코딩 옵션을 적용했을 때 출력이 어떻게 달라지는지 비교한다.
 * 로컬 Ollama가 있으면 실제로 호출해서 결과를 비교하고,
 * 없으면 구성한 옵션 객체의 내용만 출력한다.
 */
@Component
public class DecodingExample {

    private static final String PROMPT = "가을 바다를 주제로 두 문장짜리 짧은 글을 한국어로 써 주세요.";

    private final ChatClient.Builder chatClientBuilder;

    public DecodingExample(ChatClient.Builder chatClientBuilder) {
        this.chatClientBuilder = chatClientBuilder;
    }

    public void run() {
        System.out.println("===== 디코딩 파라미터 예제 시작 =====");

        // tag::chat-options[]
        // temperature를 0으로 낮추면 매번 거의 같은, 가장 확률 높은 답이 나온다
        OllamaOptions deterministicOptions = OllamaOptions.builder()
                .temperature(0.0)
                .topP(0.5)
                .build();

        // temperature를 높이면 확률 분포가 평평해져 더 다양한 표현이 나온다
        OllamaOptions creativeOptions = OllamaOptions.builder()
                .temperature(1.2)
                .topP(0.95)
                .build();
        // end::chat-options[]

        System.out.println("프롬프트: " + PROMPT);
        System.out.printf("옵션 A (보수적): temperature=%.1f, top-p=%.2f%n",
                deterministicOptions.getTemperature(), deterministicOptions.getTopP());
        System.out.printf("옵션 B (다양성): temperature=%.1f, top-p=%.2f%n",
                creativeOptions.getTemperature(), creativeOptions.getTopP());

        if (!OllamaStatus.available()) {
            System.out.println();
            System.out.println("[안내] localhost:11434 에서 Ollama 응답이 없습니다. 옵션 구성만 출력했습니다.");
            System.out.println("[안내] Ollama를 설치하고 모델을 받은 뒤 다시 실행하면 두 옵션의 결과를 비교할 수 있습니다.");
            System.out.println("===== 디코딩 파라미터 예제 종료 =====");
            return;
        }

        ChatClient chatClient = chatClientBuilder.build();

        System.out.println();
        System.out.println("[옵션 A: temperature 0.0 결과]");
        System.out.println(chatClient.prompt()
                .user(PROMPT)
                .options(deterministicOptions)
                .call()
                .content());

        System.out.println();
        System.out.println("[옵션 B: temperature 1.2 결과]");
        System.out.println(chatClient.prompt()
                .user(PROMPT)
                .options(creativeOptions)
                .call()
                .content());

        System.out.println();
        System.out.println("같은 프롬프트라도 옵션 A는 반복 실행 시 거의 같은 문장을, 옵션 B는 더 다양한 문장을 만들어 냅니다.");
        System.out.println("===== 디코딩 파라미터 예제 종료 =====");
    }
}
