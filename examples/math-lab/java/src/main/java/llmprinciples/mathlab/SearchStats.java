package llmprinciples.mathlab;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.inference.OneWayAnova;
import org.apache.commons.math3.stat.inference.TTest;

/** 부록 통계 참고의 검색 통계와 유의성 검정 절을 검산한다. */
public final class SearchStats {

    private SearchStats() {
    }

    /** 부록 "검색의 통계: TF-IDF와 BM25" */
    public static void searchStatistics(Checker c) {
        c.section("검색의 통계: TF-IDF와 BM25");

        // 부록: 모든 문서에 나오는 단어는 IDF가 log(1) = 0이 되어 검색에 기여하지 못한다.
        c.near("모든 문서에 나오는 단어의 IDF", 0, idf(1000, 1000), 1e-12);

        // 부록: 드문 단어일수록 값이 커진다.
        double rare = idf(1000, 2);
        double common = idf(1000, 500);
        c.ok("드문 단어의 IDF가 더 큼", rare > common,
                "df=2에서 " + Checker.num(rare) + ", df=500에서 " + Checker.num(common));

        // 부록: 문서 1,000개 가운데 10개에만 나오는 단어는 IDF가 ln(1000 / 10) = ln(100) ≈ 4.61이고,
        // 절반인 500개에 나오는 단어는 ln(2) ≈ 0.69다.
        c.near("df=10의 IDF", 4.61, idf(1000, 10), 5e-3);
        c.near("df=500의 IDF", 0.69, common, 5e-3);

        boolean monotone = true;
        double prev = Double.POSITIVE_INFINITY;
        for (double df : new double[] {1, 10, 100, 500, 1000}) {
            double v = idf(1000, df);
            if (v > prev) {
                monotone = false;
            }
            prev = v;
        }
        c.ok("df가 커질수록 IDF가 단조 감소", monotone, "흔할수록 벌점");

        // 부록: TF-IDF는 두 빈도의 곱이다.
        c.near("TF-IDF = TF × IDF", 3 * idf(1000, 10), 3 * idf(1000, 10), 1e-12);
        c.ok("같은 단어라도 많이 나오면 점수가 큼",
                5 * idf(1000, 10) > 2 * idf(1000, 10), "TF에 비례");

        // 부록: 문서 길이가 평균과 같을 때 빈도 1, 10, 20의 기여는 1.0, 1.96, 2.08이다.
        c.near("BM25 빈도 1의 기여", 1.0, bm25TermFrequency(1, AVG_LEN), 5e-3);
        c.near("BM25 빈도 10의 기여", 1.96, bm25TermFrequency(10, AVG_LEN), 5e-3);
        c.near("BM25 빈도 20의 기여", 2.08, bm25TermFrequency(20, AVG_LEN), 5e-3);

        // 부록: 문서 길이가 평균과 같으면 괄호 안이 1이 되어 식이 tf * 2.2 / (tf + 1.2)가 되고,
        // 상한은 k1 + 1 = 2.2다.
        c.near("평균 길이일 때의 식 tf * 2.2 / (tf + 1.2)", 10 * 2.2 / (10 + 1.2),
                bm25TermFrequency(10, AVG_LEN), 1e-12);
        c.near("상한 k1 + 1", 2.2, K1 + 1, 1e-12);

        // 부록: 길이가 평균의 두 배인 문서는 괄호 안이 1 - 0.75 + 0.75 * 2 = 1.75가 되어,
        // 빈도 1의 기여가 1.0에서 0.71로 내려간다.
        c.near("길이가 두 배일 때의 괄호 안", 1.75, 1 - B + B * 2, 1e-12);
        c.near("길이가 두 배일 때 빈도 1의 기여", 0.71, bm25TermFrequency(1, 2 * AVG_LEN), 5e-3);

        // 부록: BM25는 단어 빈도의 기여에 상한을 두어 포화시키고 문서 길이로 보정한다.
        double tf10 = bm25TermFrequency(10, AVG_LEN);
        double tf20 = bm25TermFrequency(20, AVG_LEN);
        c.ok("BM25는 빈도가 두 배여도 점수가 두 배가 아님", tf20 < 2 * tf10,
                "tf=10에서 " + Checker.num(tf10) + ", tf=20에서 " + Checker.num(tf20));
        c.note("TF-IDF의 빈도 항이라면 10에서 20으로 정확히 두 배가 된다");

        // 빈도를 아무리 올려도 k1+1을 넘지 못한다는 것이 포화의 내용이다.
        c.ok("BM25 빈도 항의 상한", bm25TermFrequency(1e9, AVG_LEN) < K1 + 1,
                "상한 " + Checker.num(K1 + 1) + "에 수렴");

        // 부록: 문서 길이로 점수를 보정한다. 같은 빈도라면 긴 문서가 손해를 본다.
        double shortDoc = bm25TermFrequency(5, 50);
        double longDoc = bm25TermFrequency(5, 400);
        c.ok("같은 빈도면 긴 문서의 점수가 낮음", longDoc < shortDoc,
                "길이 50에서 " + Checker.num(shortDoc)
                        + ", 길이 400에서 " + Checker.num(longDoc));
    }

    /** 부록 "t-검정과 분산 분석: 지표의 차이가 진짜인지 확인하기" */
    public static void significanceTests(Checker c) {
        c.section("t-검정과 분산 분석");

        // 부록: 정확도 80% 근처, 문항 100개면 표준 오차가 약 4%p다.
        double p = 0.8;
        double n = 100;
        double se = Math.sqrt(p * (1 - p) / n);
        c.near("정확도 80%, 문항 100개의 표준 오차", 0.04, se, 1e-12);

        // 부록: 표준 오차를 절반으로 줄이려면 문항이 400개 필요하다.
        // 정밀도를 두 배로 만들려면 표본이 네 배 필요하다는 제곱근의 법칙이다.
        c.near("문항 400개의 표준 오차", se / 2, Math.sqrt(p * (1 - p) / 400), 1e-12);

        // 부록: 문항 하나의 채점 결과(맞으면 1, 틀리면 0)의 분산이 p * (1 - p)이고,
        // n개의 평균을 내면 분산이 n분의 1로 준다. 분산은 E[x²] - E[x]²로 구한다.
        double scoreVariance = (p * 1 * 1 + (1 - p) * 0 * 0) - p * p;
        c.near("채점 결과의 분산 p * (1 - p)", 0.16, scoreVariance, 1e-12);

        // 정답률 80%인 문항 100개짜리 평가를 2만 번 되풀이해 정확도가 흩어지는 폭을 잰다.
        // 뒤쪽 검정들의 난수 순서를 바꾸지 않도록 난수 생성기를 따로 둔다.
        Random evalRnd = new Random(2324);
        final int evalRuns = 20000;
        double[] accuracies = new double[evalRuns];
        int inside = 0;
        for (int run = 0; run < evalRuns; run++) {
            int correct = 0;
            for (int item = 0; item < 100; item++) {
                if (evalRnd.nextDouble() < p) {
                    correct++;
                }
            }
            accuracies[run] = correct / 100.0;
            if (correct >= 72 && correct <= 88) {
                inside++;
            }
        }
        c.near("되풀이한 정확도의 표준편차 = 표준 오차", se,
                Math.sqrt(StatUtils.variance(accuracies)), 2e-3);

        // 부록: 실제 정답률이 정확히 80%여도 문항 100개로 재면 평균 ±2 표준 오차인
        // 72%에서 88% 사이의 값이 나올 수 있다.
        c.near("평균 - 2 표준 오차", 0.72, p - 2 * se, 1e-12);
        c.near("평균 + 2 표준 오차", 0.88, p + 2 * se, 1e-12);
        c.ok("되풀이한 정확도 대부분이 72%~88% 안", inside / (double) evalRuns > 0.93,
                Checker.num(inside * 100.0 / evalRuns) + "%가 범위 안");

        // 부록: 79%와 83%의 차이는 우연의 범위를 벗어나지 못한다.
        double diff = 0.83 - 0.79;
        double seDiff = Math.sqrt(0.79 * 0.21 / 100 + 0.83 * 0.17 / 100);
        double z = diff / seDiff;
        double pValue = 2 * (1 - new NormalDistribution(0, 1).cumulativeProbability(Math.abs(z)));
        c.ok("79% vs 83%는 유의하지 않음", pValue > 0.05, "p값 " + Checker.num(pValue));
        // 부록: 각 정확도의 표준 오차는 약 0.041과 0.038, 차이의 표준 오차는 약 0.055,
        // t는 약 0.7로 2에 한참 못 미친다.
        c.near("79%의 표준 오차", 0.041, Math.sqrt(0.79 * 0.21 / 100), 5e-4);
        c.near("83%의 표준 오차", 0.038, Math.sqrt(0.83 * 0.17 / 100), 5e-4);
        c.near("차이의 표준 오차", 0.055, seDiff, 5e-4);
        c.near("t 통계량", 0.7, z, 5e-2);
        c.note("t = %.3f로, 유의 판정선인 약 2에 못 미친다", z);

        // 부록: 대응표본 t-검정은 질문별 점수 차이를 먼저 구하고 그 평균이
        // 0에서 벗어났는지 본다. 질문 난이도가 만드는 흔들림이 상쇄되므로 더 민감하다.
        Random rnd = new Random(1920);
        final int items = 100;
        double[] before = new double[items];
        double[] after = new double[items];
        for (int i = 0; i < items; i++) {
            double difficulty = 6.0 * rnd.nextGaussian(); // 질문마다 크게 다른 난이도
            before[i] = 70 + difficulty + 0.5 * rnd.nextGaussian();
            after[i] = before[i] + 1.0 + 0.5 * rnd.nextGaussian(); // 일정하게 1점 개선
        }

        TTest tTest = new TTest();
        double pairedP = tTest.pairedTTest(before, after);
        double unpairedP = tTest.tTest(before, after);
        c.ok("대응표본 검정이 더 민감함", pairedP < unpairedP,
                "대응 p=" + Checker.num(pairedP) + " < 독립 p=" + Checker.num(unpairedP));
        c.ok("대응표본 검정은 개선을 잡아냄", pairedP < 0.05, "p값 " + Checker.num(pairedP));
        c.ok("독립표본 검정은 같은 개선을 놓침", unpairedP > 0.05,
                "p값 " + Checker.num(unpairedP));
        c.note("질문 난이도의 흔들림이 두 구성에 똑같이 들어 있어 대응표본에서는 상쇄된다");

        // 부록: 분산 분석은 집단 사이의 분산과 집단 안의 분산의 비율(F비)로
        // "차이가 하나라도 있는가"를 한 번에 검정한다.
        OneWayAnova anova = new OneWayAnova();
        List<double[]> groups = new ArrayList<>();
        groups.add(sample(rnd, 40, 70));
        groups.add(sample(rnd, 40, 70));
        groups.add(sample(rnd, 40, 76)); // 이 집단만 다르다
        double fValue = anova.anovaFValue(groups);
        double anovaP = anova.anovaPValue(groups);
        c.ok("차이가 있는 세 집단에서 유의한 F비", anovaP < 0.01,
                "F=" + Checker.num(fValue) + ", p=" + Checker.num(anovaP));

        // 세 집단이 모두 같은 분포라면 유의 판정이 나오는 비율이 유의수준 5% 근처여야 한다.
        // 한 번만 돌려서는 알 수 없으므로 같은 실험을 여러 번 반복해서 비율을 잰다.
        final int anovaRuns = 2000;
        int falseAlarm = 0;
        for (int r = 0; r < anovaRuns; r++) {
            List<double[]> same = new ArrayList<>();
            for (int g = 0; g < 3; g++) {
                same.add(sample(rnd, 40, 70));
            }
            if (anova.anovaPValue(same) < 0.05) {
                falseAlarm++;
            }
        }
        c.near("차이 없는 세 집단의 위양성률 = 유의수준", 0.05,
                (double) falseAlarm / anovaRuns, 0.015);

        // 부록: 쌍마다 t-검정을 반복하면 우연히 유의해 보이는 결과가 나올 확률이 누적된다.
        final int runs = 2000;
        int falsePositives = 0;
        for (int r = 0; r < runs; r++) {
            double[][] four = new double[4][];
            for (int g = 0; g < 4; g++) {
                four[g] = sample(rnd, 30, 70);
            }
            boolean flagged = false;
            for (int i = 0; i < four.length; i++) {
                for (int j = i + 1; j < four.length; j++) {
                    if (tTest.tTest(four[i], four[j]) < 0.05) {
                        flagged = true;
                    }
                }
            }
            if (flagged) {
                falsePositives++;
            }
        }
        double rate = (double) falsePositives / runs;
        c.ok("쌍별 반복 검정의 위양성률이 5%를 넘음", rate > 0.05,
                "집단 4개에서 " + Checker.num(rate * 100) + "%");
        c.note("검정 6번을 반복하면 이론값은 1-0.95^6 = %.1f%%다",
                (1 - Math.pow(0.95, 6)) * 100);

        // 부록: 모델이 세 개면 쌍도 세 개다. 세 검정이 서로 독립이라고 어림하면
        // 한 번이라도 우연히 유의하다고 나올 확률은 1 - 0.95³ ≈ 14%다.
        c.near("모델 세 개의 쌍 수", 3, 3 * 2 / 2, 0);
        c.near("검정 세 번의 위양성 누적 1 - 0.95³", 0.14, 1 - Math.pow(0.95, 3), 5e-3);
    }

    private static final double K1 = 1.2;
    private static final double B = 0.75;
    private static final double AVG_LEN = 100.0;

    /** 부록: IDF = log(전체 문서 수 / 그 단어가 나온 문서 수) */
    private static double idf(double total, double documentFrequency) {
        return Math.log(total / documentFrequency);
    }

    /** BM25에서 단어 빈도가 기여하는 항. 빈도가 커져도 K1+1에서 포화한다. */
    private static double bm25TermFrequency(double tf, double docLen) {
        return tf * (K1 + 1) / (tf + K1 * (1 - B + B * docLen / AVG_LEN));
    }

    private static double[] sample(Random rnd, int size, double mean) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = mean + 3 * rnd.nextGaussian();
        }
        return out;
    }
}
