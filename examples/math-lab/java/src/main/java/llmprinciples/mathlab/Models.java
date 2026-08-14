package llmprinciples.mathlab;

import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.SimpleRegression;

/** 부록의 회귀, 마르코프 체인, 강화학습, 스케일링 법칙 절을 검산한다. */
public final class Models {

    private Models() {
    }

    /** 부록 "선형 회귀와 로지스틱 회귀: 가장 작은 신경망" */
    public static void regression(Checker c) {
        c.section("선형 회귀와 로지스틱 회귀: 가장 작은 신경망");

        // 부록: 선형 회귀는 데이터에 가장 잘 맞는 직선 y = ax + b를 찾는다.
        // 기울기 2.5, 절편 -1인 직선에 잡음을 얹고 계수를 되찾아 본다.
        Random rnd = new Random(1516);
        final int n = 300;
        SimpleRegression reg = new SimpleRegression();
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = rnd.nextDouble() * 10;
            ys[i] = 2.5 * xs[i] - 1 + 0.3 * rnd.nextGaussian();
            reg.addData(xs[i], ys[i]);
        }
        c.near("회귀가 되찾은 기울기 a", 2.5, reg.getSlope(), 0.05);
        c.near("회귀가 되찾은 절편 b", -1, reg.getIntercept(), 0.1);

        // 부록: "가장 잘 맞는"의 기준은 오차 제곱합을 최소로 만드는 것이다(최소제곱법).
        double base = sse(xs, ys, reg.getSlope(), reg.getIntercept());
        boolean worse = true;
        for (double delta : new double[] {-0.05, -0.01, 0.01, 0.05}) {
            if (sse(xs, ys, reg.getSlope() + delta, reg.getIntercept()) < base) {
                worse = false;
            }
        }
        c.ok("계수를 흔들면 오차 제곱합이 늘어남", worse, "최소제곱 해가 맞음");
        c.near("잔차 제곱합이 라이브러리 값과 일치", reg.getSumSquaredErrors(), base, 1e-6);

        // 부록: 시그모이드는 선택지가 두 개일 때의 softmax와 같은 함수다.
        double z = 1.3;
        double sig = 1 / (1 + Math.exp(-z));
        c.near("시그모이드 = 2항 softmax", sig,
                Probability.softmax(new double[] {z, 0}, 1.0)[0], 1e-12);

        // 부록: 로지스틱 회귀의 손실(로그 손실)은 교차 엔트로피와 같은 식이다.
        double label = 1.0;
        double logLoss = -(label * Math.log(sig) + (1 - label) * Math.log(1 - sig));
        c.near("로그 손실 = 교차 엔트로피",
                Probability.crossEntropy(new double[] {1, 0}, new double[] {sig, 1 - sig}),
                logLoss, 1e-12);

        // 부록: RLHF의 보상 모델은 "A가 B보다 선호될 확률 = sigmoid(A점수 - B점수)"라는
        // Bradley-Terry 모형으로 학습된다.
        c.near("점수가 같으면 선호 확률 0.5", 0.5, sigmoid(0), 1e-12);
        c.ok("점수 차이가 커지면 확률이 1에 접근",
                sigmoid(5) > 0.99 && sigmoid(1) > sigmoid(0.5),
                "diff=5에서 " + Checker.num(sigmoid(5)));

        // 부록: 다중공선성은 입력 변수들끼리 강하게 상관되어 있을 때,
        // 예측은 멀쩡한데 개별 계수의 추정이 불안정해지는 현상이다.
        double[][] inputs = new double[n][2];
        double[] target = new double[n];
        for (int i = 0; i < n; i++) {
            double v1 = rnd.nextGaussian();
            double v2 = v1 + 0.15 * rnd.nextGaussian(); // v1과 거의 같은 변수
            inputs[i] = new double[] {v1, v2};
            target[i] = 3 * v1 + 2 * v2 + 0.1 * rnd.nextGaussian();
        }
        double[] col1 = column(inputs, 0);
        double[] col2 = column(inputs, 1);
        double corr = new PearsonsCorrelation().correlation(col1, col2);
        c.ok("두 입력이 강하게 상관됨", corr > 0.98, "상관계수 " + Checker.num(corr));

        double[] coefA = ols(inputs, target);
        // 표본의 절반만 써서 다시 적합하면 계수가 크게 흔들린다.
        double[][] halfInputs = new double[n / 2][];
        double[] halfTarget = new double[n / 2];
        System.arraycopy(inputs, 0, halfInputs, 0, n / 2);
        System.arraycopy(target, 0, halfTarget, 0, n / 2);
        double[] coefB = ols(halfInputs, halfTarget);

        double coefShift = Math.abs(coefA[0] - coefB[0]) + Math.abs(coefA[1] - coefB[1]);
        // 예측은 두 계수의 합이 결정하므로 훨씬 안정적이다.
        double sumShift = Math.abs((coefA[0] + coefA[1]) - (coefB[0] + coefB[1]));
        c.ok("계수는 흔들려도 예측은 안정적", sumShift < coefShift,
                "계수 변화 " + Checker.num(coefShift)
                        + " vs 계수 합 변화 " + Checker.num(sumShift));
        c.note("표본을 바꾸면 계수가 [%.2f %.2f] → [%.2f %.2f]로 이동한다",
                coefA[0], coefA[1], coefB[0], coefB[1]);

        // 분산 팽창 계수(VIF)는 이 불안정성을 재는 표준 지표다.
        // 한 변수를 나머지로 회귀했을 때의 결정계수 R²로 1/(1-R²)을 구한다.
        OLSMultipleLinearRegression aux = new OLSMultipleLinearRegression();
        double[][] single = new double[n][1];
        for (int i = 0; i < n; i++) {
            single[i][0] = col1[i];
        }
        aux.newSampleData(col2, single);
        double vif = 1 / (1 - aux.calculateRSquared());
        c.ok("VIF가 경험적 기준 10을 넘음", vif > 10, "VIF " + Checker.num(vif));
    }

    /** 부록 "마르코프 체인: 상태가 미래를 결정한다" */
    public static void markovChain(Checker c) {
        c.section("마르코프 체인: 상태가 미래를 결정한다");

        // 상태 세 개의 전이 확률표를 만든다. n-gram 언어 모델이 말뭉치에서 센 표와
        // 같은 물건이다. 각 행의 합은 1이어야 한다.
        RealMatrix p = MatrixUtils.createRealMatrix(new double[][] {
                {0.7, 0.2, 0.1},
                {0.3, 0.4, 0.3},
                {0.2, 0.3, 0.5},
        });
        boolean rowsSumToOne = true;
        for (int i = 0; i < 3; i++) {
            double sum = 0;
            for (double v : p.getRow(i)) {
                sum += v;
            }
            if (Math.abs(sum - 1) > 1e-12) {
                rowsSumToOne = false;
            }
        }
        c.ok("전이 확률표의 각 행 합이 1", rowsSumToOne, "확률 분포의 조건");

        // 부록: 확률표만 있으면 과정 전체를 전개할 수 있다.
        RealVector stationary = converge(p, new double[] {1, 0, 0});
        double sum = 0;
        for (double v : stationary.toArray()) {
            sum += v;
        }
        c.near("정상 분포의 합", 1, sum, 1e-9);

        // 정상 분포는 자기 자신으로 되돌아온다. πP = π다.
        RealVector again = p.transpose().operate(stationary);
        c.ok("정상 분포는 전이해도 그대로",
                again.subtract(stationary).getL1Norm() < 1e-9, "πP = π");
        c.note("정상 분포 [%.4f %.4f %.4f]",
                stationary.getEntry(0), stationary.getEntry(1), stationary.getEntry(2));

        // 부록: 다음에 일어날 일이 현재 상태에만 의존하고 거기까지 온 경로에는
        // 의존하지 않는다. 서로 다른 출발점에서 시작해도 같은 정상 분포로 간다.
        RealVector other = converge(p, new double[] {0, 0, 1});
        c.ok("출발점이 달라도 같은 곳으로 수렴",
                other.subtract(stationary).getL1Norm() < 1e-9, "경로가 아니라 상태만 남는다");

        // n단계 전이 확률은 전이 행렬의 n제곱이며, 중간 경로를 기억할 필요가 없다.
        c.near("2단계 전이 = 전이 행렬의 제곱", 0,
                p.power(2).subtract(p.multiply(p)).getFrobeniusNorm(), 1e-12);
    }

    /** 부록 "벨만 방정식: 강화학습의 뼈대" */
    public static void bellman(Checker c) {
        c.section("벨만 방정식: 강화학습의 뼈대");

        // 부록: 매 걸음 보상 1을 받고 할인율이 γ면, 가치는 등비급수의 합 1/(1-γ)다.
        final double gamma = 0.9;
        c.near("등비급수의 합 1/(1-γ)", 10, 1 / (1 - gamma), 1e-12);

        double partial = 0;
        for (int k = 0; k < 300; k++) {
            partial += Math.pow(gamma, k);
        }
        c.near("보상 1이 계속될 때의 가치", 1 / (1 - gamma), partial, 1e-9);

        // 부록: 할인율은 무한히 이어지는 보상의 합이 발산하지 않게 만드는 장치다.
        double noDiscount = 0;
        for (int k = 0; k < 300; k++) {
            noDiscount += 1.0;
        }
        c.ok("γ=1이면 합이 커지기만 함", noDiscount == 300, "300걸음에서 이미 300");

        // 부록: V(s) = E[r + γ * V(s')]. 상태 세 개짜리 마르코프 보상 과정에서
        // 가치 반복으로 구한 해가 선형 방정식의 해와 같은지 확인한다.
        RealMatrix trans = MatrixUtils.createRealMatrix(new double[][] {
                {0.5, 0.5, 0.0},
                {0.0, 0.5, 0.5},
                {0.5, 0.0, 0.5},
        });
        RealVector rewards = new ArrayRealVector(new double[] {1, 0, 2});

        // 가치 반복: V ← r + γPV를 반복한다.
        RealVector v = new ArrayRealVector(new double[] {0, 0, 0});
        for (int i = 0; i < 500; i++) {
            v = rewards.add(trans.operate(v).mapMultiply(gamma));
        }

        // 닫힌 해: (I - γP)V = r
        RealMatrix m = MatrixUtils.createRealIdentityMatrix(3)
                .subtract(trans.scalarMultiply(gamma));
        RealVector solved = new LUDecomposition(m).getSolver().solve(rewards);
        c.ok("가치 반복의 해 = 선형 방정식의 해",
                v.subtract(solved).getL1Norm() < 1e-9,
                "두 방법의 차이 " + Checker.num(v.subtract(solved).getL1Norm()));
        c.note("가치 함수 V = [%.3f %.3f %.3f]",
                v.getEntry(0), v.getEntry(1), v.getEntry(2));

        // 벨만 방정식이 실제로 성립하는지 한 상태에서 직접 확인한다.
        double rhs = rewards.getEntry(0) + gamma * trans.operate(v).getEntry(0);
        c.near("V(s) = r + γ·E[V(s')]", v.getEntry(0), rhs, 1e-9);

        // 부록: 먼 미래의 보상은 γ, γ², γ³처럼 할인된다.
        c.near("10걸음 뒤 보상의 할인율", Math.pow(gamma, 10), 0.34867844, 1e-6);
    }

    /** 부록 "스케일링 법칙: 손실을 예측하는 멱법칙" */
    public static void scalingLaw(Checker c) {
        c.section("스케일링 법칙: 손실을 예측하는 멱법칙");

        // 부록: 멱법칙 y = a * x^(-b)는 양변에 로그를 취하면
        // log y = log a - b * log x가 되어 로그-로그 그래프에서 직선이 된다.
        final double a = 12.0;
        final double b = 0.35;
        Random rnd = new Random(1718);

        SimpleRegression reg = new SimpleRegression();
        for (double x : new double[] {1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9}) {
            double y = a * Math.pow(x, -b) * (1 + 0.01 * rnd.nextGaussian());
            reg.addData(Math.log(x), Math.log(y));
        }
        c.near("로그-로그 회귀가 되찾은 지수 b", b, -reg.getSlope(), 0.01);
        c.near("로그-로그 회귀가 되찾은 계수 a", a, Math.exp(reg.getIntercept()), 0.5);

        // 부록: 작은 모델 여러 개로 계수를 추정하면 훨씬 큰 모델의 손실을 외삽할 수 있다.
        double bigN = 1e11;
        double predicted = Math.exp(reg.predict(Math.log(bigN)));
        double actual = a * Math.pow(bigN, -b);
        double relative = Math.abs(predicted - actual) / actual;
        c.ok("100배 큰 규모의 손실 외삽", relative < 0.05,
                "상대 오차 " + Checker.num(relative * 100) + "%");
        c.note("예측 %.5f, 참값 %.5f", predicted, actual);

        // 부록: 손실(N, D) = E + A/N^α + B/D^β. E는 아무리 키워도 남는 손실의 바닥이다.
        c.ok("N과 D를 키우면 손실이 줄어듦",
                chinchillaLoss(1e10, 1e11) < chinchillaLoss(1e9, 1e10), "두 항이 모두 감소");
        c.near("N, D를 무한히 키운 극한이 바닥 E", 1.69, chinchillaLoss(1e30, 1e30), 1e-3);
        c.ok("바닥 아래로는 내려가지 않음", chinchillaLoss(1e12, 1e13) > 1.69,
                "손실 " + Checker.num(chinchillaLoss(1e12, 1e13)));

        // 부록: Chinchilla의 결론은 파라미터 1개당 약 20토큰이라는 균형점이었다.
        c.near("20B 파라미터에 맞는 토큰 수", 400e9, 20e9 * 20, 1);
        c.note("파라미터 20B면 학습 토큰 400B가 균형점이라는 계산이다");
    }

    private static double chinchillaLoss(double n, double d) {
        return 1.69 + 406.4 / Math.pow(n, 0.34) + 410.7 / Math.pow(d, 0.28);
    }

    private static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    private static double sse(double[] xs, double[] ys, double slope, double intercept) {
        double total = 0;
        for (int i = 0; i < xs.length; i++) {
            double r = ys[i] - (slope * xs[i] + intercept);
            total += r * r;
        }
        return total;
    }

    /** 절편 없는 다변량 최소제곱 해를 돌려준다. */
    private static double[] ols(double[][] inputs, double[] target) {
        OLSMultipleLinearRegression model = new OLSMultipleLinearRegression();
        model.setNoIntercept(true);
        model.newSampleData(target, inputs);
        return model.estimateRegressionParameters();
    }

    private static double[] column(double[][] rows, int j) {
        double[] out = new double[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = rows[i][j];
        }
        return out;
    }

    /** 전이 행렬을 반복해서 곱해 정상 분포를 구한다. */
    private static RealVector converge(RealMatrix p, double[] start) {
        RealVector dist = new ArrayRealVector(start);
        for (int i = 0; i < 200; i++) {
            dist = p.transpose().operate(dist);
        }
        return dist;
    }
}
