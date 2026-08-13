package com.example.springai;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

/**
 * Spring AI 예제 콘솔 애플리케이션.
 *
 * 실행 인자 --mode 값에 따라 예제를 선택한다.
 * - --mode=rag      : 문서 청킹, 임베딩, 벡터 검색, 프롬프트 조립 (API 키 없이 실행 가능)
 * - --mode=decoding : temperature, top-p 등 디코딩 파라미터 비교
 *
 * application.properties 에서 lazy-initialization 을 켜 두었기 때문에
 * 선택한 예제에 필요한 빈만 만들어진다. 예를 들어 decoding 모드에서는
 * 임베딩 모델(ONNX 파일 다운로드)을 준비하지 않는다.
 */
@SpringBootApplication
public class SpringAiExampleApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringAiExampleApplication.class, args);
    }

    @Bean
    ApplicationRunner exampleRunner(ObjectProvider<RagExample> ragExample,
                                    ObjectProvider<DecodingExample> decodingExample) {
        return (ApplicationArguments args) -> {
            String mode = args.containsOption("mode")
                    ? args.getOptionValues("mode").getFirst()
                    : "rag";
            switch (mode) {
                case "rag" -> ragExample.getObject().run();
                case "decoding" -> decodingExample.getObject().run();
                default -> System.out.println(
                        "알 수 없는 모드입니다: " + mode + " (사용 가능: rag, decoding)");
            }
        };
    }
}
