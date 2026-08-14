package llmprinciples.mathlab;

import java.util.Arrays;

/** 부록의 확률, softmax, 정보량 관련 절을 검산한다. */
public final class Probability {

    private Probability() {
    }

    /** 부록 "확률 분포와 조건부 확률" */
    public static void probabilityAndBayes(Checker c) {
        c.section("확률 분포와 조건부 확률");

        // 부록: 확률 분포는 모든 경우에 확률을 배정한 것으로 전체 합이 1이다.
        double[] p = softmax(new double[] {2.0, 1.0, 0.1}, 1.0);
        c.near("softmax 출력의 합", 1, Arrays.stream(p).sum(), 1e-12);

        // 부록: 곱의 법칙에 따라 문장 전체의 확률은 조건부 확률들의 곱으로 분해된다.
        // P(t1, t2) = P(t1) * P(t2 | t1)을 결합 분포에서 직접 확인한다.
        double[][] joint = {
                {0.10, 0.30}, // P(t1=0, t2=0), P(t1=0, t2=1)
                {0.45, 0.15}, // P(t1=1, t2=0), P(t1=1, t2=1)
        };
        double pT1 = joint[1][0] + joint[1][1];        // P(t1=1)
        double pT2GivenT1 = joint[1][1] / pT1;         // P(t2=1 | t1=1)
        c.near("곱의 법칙 P(t1,t2) = P(t1)·P(t2|t1)", joint[1][1], pT1 * pT2GivenT1, 1e-12);

        // 부록: 베이즈 정리는 P(A|B) = P(B|A) * P(A) / P(B)다.
        // 스팸 필터를 예로 든다. 스팸 비율 20%, 스팸에 "무료"가 들어갈 확률 60%,
        // 정상 메일에 들어갈 확률 5%로 두고 "무료"를 본 뒤 스팸일 확률을 구한다.
        double pSpam = 0.20;
        double pWordGivenSpam = 0.60;
        double pWordGivenHam = 0.05;
        double pWord = pWordGivenSpam * pSpam + pWordGivenHam * (1 - pSpam); // 전체 확률의 법칙
        double pSpamGivenWord = pWordGivenSpam * pSpam / pWord;

        c.near("전체 확률의 법칙으로 구한 P(단어)", 0.16, pWord, 1e-12);
        c.near("베이즈 정리로 뒤집은 P(스팸|단어)", 0.75, pSpamGivenWord, 1e-12);
        c.ok("증거를 보고 확률이 갱신됨", pSpamGivenWord > pSpam,
                "사전 " + Checker.num(pSpam) + " → 사후 " + Checker.num(pSpamGivenWord));

        // 부록: 나이브 베이즈는 클래스가 주어지면 단어들이 독립이라고 가정해서
        // P(단어들|클래스)를 단어별 확률의 곱, 실제 계산으로는 로그 확률의 합으로 바꾼다.
        double[] spamWordProbs = {0.60, 0.40, 0.30};
        double[] hamWordProbs = {0.05, 0.10, 0.20};

        double prodSpam = pSpam;
        double logSpam = Math.log(pSpam);
        for (double q : spamWordProbs) {
            prodSpam *= q;
            logSpam += Math.log(q);
        }
        c.near("곱과 로그 합이 같은 값", Math.log(prodSpam), logSpam, 1e-12);

        double prodHam = 1 - pSpam;
        for (double q : hamWordProbs) {
            prodHam *= q;
        }
        c.ok("나이브 베이즈가 스팸으로 분류", prodSpam > prodHam,
                "스팸 점수 " + Checker.num(prodSpam) + " > 정상 점수 " + Checker.num(prodHam));
    }

    /** 부록 "로짓, softmax, temperature" */
    public static void softmaxAndTemperature(Checker c) {
        c.section("로짓, softmax, temperature");

        double[] logits = {2.0, 1.0, 0.1};

        // 부록: exp는 [7.39, 2.72, 1.11]이고 합은 11.21이다.
        double[] exps = Arrays.stream(logits).map(Math::exp).toArray();
        c.near("exp(2.0)", 7.39, exps[0], 5e-3);
        c.near("exp(1.0)", 2.72, exps[1], 5e-3);
        c.near("exp(0.1)", 1.11, exps[2], 5e-3);
        c.near("exp들의 합", 11.21, Arrays.stream(exps).sum(), 5e-3);

        // 부록: softmax는 [0.66, 0.24, 0.10]이고 합은 1.0이다.
        double[] p = softmax(logits, 1.0);
        c.near("softmax 1위 확률", 0.66, p[0], 5e-3);
        c.near("softmax 2위 확률", 0.24, p[1], 5e-3);
        c.near("softmax 3위 확률", 0.10, p[2], 5e-3);
        c.near("softmax 합", 1.0, Arrays.stream(p).sum(), 1e-12);

        // 부록: 점수가 1 크면 확률은 약 2.7배가 된다.
        c.near("로짓 1 차이의 확률 배율", Math.E, p[0] / p[1], 1e-9);

        // 부록: T = 0.5면 [0.86, 0.12, 0.02]로 뾰족해진다.
        double[] cold = softmax(logits, 0.5);
        c.near("T=0.5의 1위 확률", 0.86, cold[0], 5e-3);
        c.near("T=0.5의 2위 확률", 0.12, cold[1], 5e-3);
        c.near("T=0.5의 3위 확률", 0.02, cold[2], 5e-3);

        // 부록: T = 2.0이면 [0.50, 0.30, 0.19]로 평평해진다.
        double[] hot = softmax(logits, 2.0);
        c.near("T=2.0의 1위 확률", 0.50, hot[0], 5e-3);
        c.near("T=2.0의 2위 확률", 0.30, hot[1], 5e-3);
        c.near("T=2.0의 3위 확률", 0.19, hot[2], 5e-3);

        // 부록: T가 0에 가까우면 그리디 선택과 같아지고, T가 크면 균등 분포에 가까워진다.
        c.near("T→0이면 1위에 확률이 몰림", 1, softmax(logits, 0.01)[0], 1e-9);
        c.near("T→∞이면 균등 분포에 가까워짐", 1.0 / 3, softmax(logits, 1000)[0], 1e-3);

        // 부록: temperature는 분포의 뾰족함을 조절한다. 엔트로피로 재면 단조 증가한다.
        boolean monotone = true;
        double prev = -1;
        for (double t : new double[] {0.25, 0.5, 1.0, 2.0, 4.0}) {
            double h = entropy(softmax(logits, t));
            if (h < prev) {
                monotone = false;
            }
            prev = h;
        }
        c.ok("T가 커지면 엔트로피가 커진다", monotone, "0.25에서 4.0까지 단조 증가");

        // softmax는 로짓 전체에 같은 수를 더해도 결과가 변하지 않는다.
        double[] big = {1002.0, 1001.0, 1000.1};
        double naive = Math.exp(big[0])
                / (Math.exp(big[0]) + Math.exp(big[1]) + Math.exp(big[2]));
        c.ok("exp를 그대로 쓰면 값이 넘침", Double.isNaN(naive),
                "exp(1002)가 무한대가 되어 NaN");
        c.near("최댓값을 빼면 원래 분포와 같음", 0.66, softmax(big, 1.0)[0], 5e-3);
    }

    /** 부록 "로그, 교차 엔트로피, 최대 우도" */
    public static void crossEntropy(Checker c) {
        c.section("로그, 교차 엔트로피, 최대 우도");

        // 부록: log(a*b) = log(a) + log(b)
        double a = 0.3;
        double b = 0.017;
        c.near("log(a*b) = log a + log b", Math.log(a * b), Math.log(a) + Math.log(b), 1e-12);

        // 부록: 손실 = -log(정답 토큰에 배정된 확률)
        c.near("확률 0.9의 손실", 0.11, -Math.log(0.9), 5e-3);
        c.near("확률 0.5의 손실", 0.69, -Math.log(0.5), 5e-3);
        c.near("확률 0.01의 손실", 4.61, -Math.log(0.01), 5e-3);

        // 부록: 정답에 확률 1을 주면 손실이 0이다.
        c.near("확률 1의 손실", 0, -Math.log(1), 1e-12);

        // 부록: 파라미터가 무작위인 모델은 어휘 27개에 균등하게 1/27을 배정하므로
        // 손실이 -log(1/27) = ln(27) ≈ 3.3에서 시작한다. 5장의 학습 전 손실이다.
        c.near("어휘 27개 균등 분포의 손실", 3.3, -Math.log(1.0 / 27), 5e-3);
        c.near("-log(1/27) = ln(27)", Math.log(27), -Math.log(1.0 / 27), 1e-12);

        // 부록: 문장의 확률은 조건부 확률 수백 개의 곱이라 부동소수점으로는
        // 금방 0으로 내려앉는다(언더플로). 그래서 로그 확률의 합으로 계산한다.
        double prob = 1.0;
        double logProb = 0.0;
        for (int i = 0; i < 700; i++) {
            prob *= 0.3;
            logProb += Math.log(0.3);
        }
        c.ok("확률 700개를 곱하면 0으로 언더플로", prob == 0, "곱셈 결과가 정확히 0");
        c.ok("로그 합은 유한한 값을 유지", !Double.isInfinite(logProb),
                "로그 확률 " + Checker.num(logProb));
        c.ok("언더플로 직전까지는 표현됨", Math.pow(0.3, 600) > 0,
                String.format("600개까지는 %.3e", Math.pow(0.3, 600)));

        // 부록: 최대 우도는 학습 데이터에 모델이 배정하는 확률을 최대로 만드는
        // 파라미터를 찾는 것이고, 부호를 뒤집으면 최소화 문제(교차 엔트로피)가 된다.
        int heads = 37;
        int total = 100;
        double bestQ = 0;
        double bestLoss = Double.POSITIVE_INFINITY;
        for (double q = 0.001; q < 1; q += 0.001) {
            double loss = -heads * Math.log(q) - (total - heads) * Math.log(1 - q);
            if (loss < bestLoss) {
                bestQ = q;
                bestLoss = loss;
            }
        }
        c.near("최대 우도 해 = 표본 비율", 0.37, bestQ, 1e-3);

        // 정답이 한 토큰에 확률 1을 주는 경우 교차 엔트로피는 -log(그 토큰의 확률)이다.
        c.near("정답이 확실할 때의 교차 엔트로피", -Math.log(0.5),
                crossEntropy(new double[] {0, 1, 0}, new double[] {0.2, 0.5, 0.3}), 1e-12);
    }

    /** 부록 "엔트로피와 perplexity" */
    public static void entropyAndPerplexity(Checker c) {
        c.section("엔트로피와 perplexity");

        // 부록: 한 토큰이 확률 1로 정해져 있으면 엔트로피는 0이다.
        c.near("확률 1인 분포의 엔트로피", 0, entropy(new double[] {1, 0, 0}), 1e-12);

        // 부록: V개 후보가 균등하면 ln(V)로 최대가 된다.
        double[] uniform = new double[27];
        Arrays.fill(uniform, 1.0 / 27);
        c.near("균등 분포 27개의 엔트로피", Math.log(27), entropy(uniform), 1e-12);

        boolean maxIsUniform = true;
        for (double[] dist : new double[][] {
                {0.5, 0.3, 0.2}, {0.9, 0.05, 0.05}, {0.4, 0.35, 0.25}}) {
            if (entropy(dist) > Math.log(3) + 1e-12) {
                maxIsUniform = false;
            }
        }
        c.ok("균등 분포의 엔트로피가 최대", maxIsUniform, "ln(3)을 넘는 분포 없음");

        // 부록: perplexity = exp(평균 손실)
        c.near("평균 손실 3.3의 perplexity", 27, Math.exp(3.2958), 0.1);
        c.near("평균 손실 2.0의 perplexity", 7.4, Math.exp(2.0), 0.05);
        c.near("ln(27) 손실의 perplexity", 27, Math.exp(Math.log(27)), 1e-9);
        c.near("균등 분포의 perplexity = 후보 수", 27, Math.exp(entropy(uniform)), 1e-9);
    }

    /** 부록 "KL 발산: 두 분포 사이의 차이 재기" */
    public static void klDivergence(Checker c) {
        c.section("KL 발산: 두 분포 사이의 차이 재기");

        double[] p = {0.5, 0.3, 0.2};
        double[] q = {0.4, 0.4, 0.2};

        // 부록: P와 Q가 같으면 0이고, 다를수록 커지며, 음수가 되지 않는다.
        c.near("같은 분포의 KL", 0, kl(p, p), 1e-12);
        c.ok("다른 분포의 KL은 양수", kl(p, q) > 0, Checker.num(kl(p, q)));

        // 부록: 순서를 바꾸면 값이 달라진다(비대칭).
        c.ok("KL은 비대칭", Math.abs(kl(p, q) - kl(q, p)) > 1e-6,
                "KL(P||Q) " + Checker.num(kl(p, q)) + " ≠ KL(Q||P) " + Checker.num(kl(q, p)));

        // 부록의 관계식: 교차 엔트로피(P, Q) = 엔트로피(P) + KL(P || Q)
        c.near("교차 엔트로피 = 엔트로피 + KL",
                crossEntropy(p, q), entropy(p) + kl(p, q), 1e-12);

        // 부록: 학습 데이터의 분포 P는 변하지 않으므로 교차 엔트로피를 줄이는 것은
        // 곧 KL을 줄이는 것이다. 모델 분포를 P 쪽으로 옮기며 두 값이 함께 줄어드는지 본다.
        boolean bothDrop = true;
        double prevCe = Double.POSITIVE_INFINITY;
        double prevKl = Double.POSITIVE_INFINITY;
        for (int step = 0; step <= 10; step++) {
            double t = step / 10.0;
            double[] model = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                model[i] = (1 - t) * q[i] + t * p[i]; // Q에서 P로 조금씩 이동
            }
            double ce = crossEntropy(p, model);
            double d = kl(p, model);
            if (ce > prevCe + 1e-12 || d > prevKl + 1e-12) {
                bothDrop = false;
            }
            prevCe = ce;
            prevKl = d;
        }
        c.ok("교차 엔트로피와 KL이 함께 줄어듦", bothDrop,
                "모델 분포를 데이터 분포로 옮기는 동안 둘 다 단조 감소");

        // 1장의 지식 증류에서 student가 teacher를 따라 배울 때의 손실이 이 값이다.
        double[] teacher = softmax(new double[] {3.0, 1.0, 0.5}, 1.0);
        double before = kl(teacher, softmax(new double[] {2.0, 1.5, 1.0}, 1.0));
        double after = kl(teacher, softmax(new double[] {2.8, 1.2, 0.6}, 1.0));
        c.ok("student가 teacher에 가까워지면 KL 감소", after < before,
                Checker.num(after) + " < " + Checker.num(before));
    }

    /**
     * 로짓을 temperature로 나눈 뒤 확률 분포로 바꾼다.
     * 최댓값을 빼는 것은 exp가 넘치지 않게 하려는 표준 기법이다.
     */
    static double[] softmax(double[] logits, double temperature) {
        double[] scaled = Arrays.stream(logits).map(v -> v / temperature).toArray();
        double max = Arrays.stream(scaled).max().orElse(0);
        double[] out = Arrays.stream(scaled).map(v -> Math.exp(v - max)).toArray();
        double sum = Arrays.stream(out).sum();
        return Arrays.stream(out).map(v -> v / sum).toArray();
    }

    /** 확률 분포의 엔트로피. Commons Math에는 없으므로 정의 그대로 구현한다. */
    static double entropy(double[] p) {
        double total = 0;
        for (double v : p) {
            if (v > 0) {
                total -= v * Math.log(v);
            }
        }
        return total;
    }

    /** 교차 엔트로피. 정답 분포 p로 모델 분포 q를 평가한다. */
    static double crossEntropy(double[] p, double[] q) {
        double total = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) {
                total -= p[i] * Math.log(q[i]);
            }
        }
        return total;
    }

    /** KL 발산. P를 기준으로 Q를 잰다. */
    static double kl(double[] p, double[] q) {
        double total = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) {
                total += p[i] * Math.log(p[i] / q[i]);
            }
        }
        return total;
    }
}
