package llmprinciples.mathlab;

/**
 * "수학·통계 기초" 부록이 본문에 적어 둔 값들을 Apache Commons Math로 다시 계산해서
 * 맞는지 확인하는 실습 프로그램이다. 부록의 절 순서를 그대로 따르므로, 부록을 읽으면서
 * 해당 절의 메서드를 열어 보면 된다. 값을 바꿔 가며 결과가 어떻게 달라지는지 보는 것이
 * 이 코드의 사용법이다.
 *
 * <p>실행: ./gradlew run
 */
public final class Main {

    private Main() {
    }

    public static void main(String[] args) {
        Checker c = new Checker();

        System.out.println("수학·통계 부록 검산 (Apache Commons Math)");

        LinearAlgebra.vectorsAndMatrices(c);      // 벡터와 행렬
        LinearAlgebra.rankAndPca(c);              // 랭크와 주성분 분석
        LinearAlgebra.discriminant(c);            // 판별 분석
        Calculus.run(c);                          // 미분과 체인 룰
        Statistics.moments(c);                    // 평균, 분산, 정규화
        Statistics.normalDistribution(c);         // 정규 분포
        Statistics.covarianceAndCorrelation(c);   // 공분산과 상관
        Statistics.kmeans(c);                     // k-평균 군집화
        Probability.probabilityAndBayes(c);       // 확률 분포와 조건부 확률
        Probability.softmaxAndTemperature(c);     // 로짓, softmax, temperature
        Probability.crossEntropy(c);              // 로그, 교차 엔트로피, 최대 우도
        Models.regression(c);                     // 선형 회귀와 로지스틱 회귀
        Probability.entropyAndPerplexity(c);      // 엔트로피와 perplexity
        Probability.klDivergence(c);              // KL 발산
        Models.markovChain(c);                    // 마르코프 체인
        Models.bellman(c);                        // 벨만 방정식
        Models.scalingLaw(c);                     // 스케일링 법칙
        SearchStats.searchStatistics(c);          // 검색의 통계
        SearchStats.significanceTests(c);         // t-검정과 분산 분석

        if (!c.summary()) {
            System.exit(1);
        }
    }
}
