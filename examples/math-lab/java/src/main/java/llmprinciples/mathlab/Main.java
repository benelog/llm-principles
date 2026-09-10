package llmprinciples.mathlab;

/**
 * 7장 "코드 뒤의 수학과 통계"가 본문에 적어 둔 값들을 Apache Commons Math로 다시 계산해서
 * 맞는지 확인하는 실습 프로그램이다. 7장의 절 순서를 그대로 따르므로, 7장을 읽으면서
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

        System.out.println("7장 코드 뒤의 수학과 통계 검산 (Apache Commons Math)");

        // 7.2 벡터와 행렬
        LinearAlgebra.vectorsAndMatrices(c);      // 벡터, 내적, 코사인 유사도, 행렬
        LinearAlgebra.rankAndPca(c);              // 랭크와 저랭크 근사 (주성분 분석 검산 포함)
        // 7.3 확률
        Probability.probabilityAndBayes(c);       // 확률 분포, 조건부 확률, 베이즈 정리
        Probability.softmaxAndTemperature(c);     // 로짓, softmax, temperature, 후보 제한
        // 7.4 로그와 정보량
        Probability.crossEntropy(c);              // 로그, 최대 우도, 교차 엔트로피
        Probability.entropyAndPerplexity(c);      // 엔트로피와 perplexity
        Probability.klDivergence(c);              // KL 발산
        // 7.5 미분
        Calculus.run(c);                          // 도함수, 체인 룰, 경사 하강법
        // 7.6 평균과 분산
        Statistics.moments(c);                    // 평균, 분산, 정규화와 sqrt(d) 스케일링
        Statistics.normalDistribution(c);         // 정규 분포와 중심 극한 정리
        // 7.7 데이터의 구조
        Statistics.covarianceAndCorrelation(c);   // 공분산과 상관계수
        LinearAlgebra.discriminant(c);            // 판별 분석
        Statistics.kmeans(c);                     // k-평균 군집화
        // 7.8 회귀
        Models.regression(c);                     // 선형 회귀와 로지스틱 회귀
        // 7.9 확률 과정과 결정
        Models.markovChain(c);                    // 마르코프 체인
        Models.bellman(c);                        // 강화학습과 벨만 방정식
        // 7.10 규모와 측정
        Models.scalingLaw(c);                     // 스케일링 법칙
        SearchStats.searchStatistics(c);          // 검색의 통계: TF-IDF와 BM25
        SearchStats.significanceTests(c);         // t-검정과 분산 분석

        if (!c.summary()) {
            System.exit(1);
        }
    }
}
