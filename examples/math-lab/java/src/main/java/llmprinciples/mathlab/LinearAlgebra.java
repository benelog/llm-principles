package llmprinciples.mathlab;

import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.Covariance;

/** 부록의 선형대수 관련 절을 검산한다. */
public final class LinearAlgebra {

    private LinearAlgebra() {
    }

    /** 부록 "벡터와 행렬: 숫자 배열의 기하학" */
    public static void vectorsAndMatrices(Checker c) {
        c.section("벡터와 행렬: 숫자 배열의 기하학");

        RealVector a = new ArrayRealVector(new double[] {1, 2});
        RealVector b = new ArrayRealVector(new double[] {3, 4});

        // 부록: a·b = 1*3 + 2*4 = 11
        c.near("내적 a·b", 11, a.dotProduct(b), 1e-12);

        // 부록: |a| = 2.236, |b| = 5
        c.near("벡터 길이 |a|", 2.236, a.getNorm(), 5e-4);
        c.near("벡터 길이 |b|", 5, b.getNorm(), 1e-12);

        // 부록: cos(a, b) = 11 / (2.236 * 5) ≈ 0.98
        c.near("코사인 유사도 cos(a,b)", 0.98, a.cosine(b), 5e-3);

        // 라이브러리가 주는 cosine이 정의 그대로인지 직접 계산과 맞춰 본다.
        c.near("cosine = 내적 / (길이 곱)",
                a.dotProduct(b) / (a.getNorm() * b.getNorm()), a.cosine(b), 1e-12);

        // 부록: 코사인 유사도는 항상 -1에서 1 사이에 들어온다.
        Random rnd = new Random(12);
        boolean inRange = true;
        for (int i = 0; i < 1000; i++) {
            RealVector u = randomVector(rnd, 32);
            RealVector v = randomVector(rnd, 32);
            double s = u.cosine(v);
            if (s < -1 - 1e-12 || s > 1 + 1e-12) {
                inRange = false;
            }
        }
        c.ok("코사인 값의 범위", inRange, "무작위 32차원 벡터 1000쌍이 모두 [-1, 1]");

        // 부록: 방향이 같으면 내적이 크고, 수직이면 0, 반대 방향이면 음수다.
        c.near("수직 벡터의 내적", 0,
                new ArrayRealVector(new double[] {1, 0})
                        .dotProduct(new ArrayRealVector(new double[] {0, 1})), 1e-12);
        c.ok("반대 방향의 내적", a.dotProduct(a.mapMultiply(-1)) < 0, "a·(-a) < 0");

        // 부록: lmHead는 임베딩 차원의 벡터를 어휘 크기의 로짓으로 바꾸는 행렬 곱 하나다.
        RealVector x = new ArrayRealVector(new double[] {0.5, -1.0, 0.25, 2.0});
        RealMatrix lmHead = MatrixUtils.createRealMatrix(new double[][] {
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0.5, 0.5, 0.5, 0.5},
        });
        RealVector logits = lmHead.operate(x);
        c.ok("lmHead 투영의 출력 길이", logits.getDimension() == 3, "임베딩 4차원 → 로짓 3개");
        // 세 번째 행은 모든 성분의 절반을 더한 것이므로 (0.5-1+0.25+2)/2 = 0.875다.
        c.near("행렬 곱이 섞은 결과", 0.875, logits.getEntry(2), 1e-12);
    }

    /** 부록 "랭크와 주성분 분석: 행렬을 적은 숫자로 근사하기" */
    public static void rankAndPca(Checker c) {
        c.section("랭크와 주성분 분석: 행렬을 적은 숫자로 근사하기");

        Random rnd = new Random(34);
        final int d = 6;
        final int r = 2;

        // 부록: 랭크가 r이면 d×r 행렬과 r×d 행렬의 곱으로 정확히 표현할 수 있다.
        // LoRA의 ΔW = B×A가 정확히 이 형태다.
        RealMatrix bm = randomMatrix(rnd, d, r);
        RealMatrix am = randomMatrix(rnd, r, d);
        RealMatrix deltaW = bm.multiply(am);

        SingularValueDecomposition svd = new SingularValueDecomposition(deltaW);
        c.ok("B×A로 만든 행렬의 랭크", svd.getRank() == r, "6×2 곱하기 2×6의 랭크는 2");

        // 부록: 저장할 숫자가 d²개에서 2dr개로 줄어든다.
        c.near("저장할 숫자 d²", 36, d * d, 0);
        c.near("저장할 숫자 2dr", 24, 2 * d * r, 0);

        // 부록: 랭크가 정확히 낮지 않아도 "거의 낮은" 경우에는 근사할 수 있다.
        RealMatrix noisy = deltaW.copy();
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                noisy.addToEntry(i, j, 0.01 * rnd.nextGaussian());
            }
        }
        RealMatrix approx = truncatedSvd(noisy, r);
        double relative = noisy.subtract(approx).getFrobeniusNorm() / noisy.getFrobeniusNorm();
        c.ok("상위 2개 특이값만 남긴 근사", relative < 0.05,
                Checker.num(relative * 100) + "% 상대 오차");

        // 부록: 주성분 분석은 분산이 가장 큰 방향을 차례로 찾는다.
        // Commons Math에는 PCA가 없으므로 공분산 행렬의 고유분해로 직접 구한다.
        // x축 방향으로 길게 퍼진 데이터를 만들면 제1주성분이 x축과 나란해야 한다.
        final int n = 400;
        double[][] data = new double[n][2];
        for (int i = 0; i < n; i++) {
            data[i][0] = 5 * rnd.nextGaussian();
            data[i][1] = 0.5 * rnd.nextGaussian();
        }
        RealMatrix cov = new Covariance(data).getCovarianceMatrix();
        EigenDecomposition eig = new EigenDecomposition(cov);
        double[] values = eig.getRealEigenvalues();

        // Commons Math는 고유값을 내림차순으로 돌려주므로 첫 번째가 제1주성분이다.
        c.ok("제1주성분의 분산이 더 크다", values[0] > values[1],
                Checker.num(values[0]) + " > " + Checker.num(values[1]));
        c.near("제1주성분이 x축과 나란함", 1,
                Math.abs(eig.getEigenvector(0).getEntry(0)), 0.02);

        double ratio = values[0] / (values[0] + values[1]);
        c.ok("제1주성분의 설명 비율", ratio > 0.9,
                Checker.num(ratio * 100) + "%를 방향 하나로 설명");
    }

    /** 부록 "판별 분석: 집단을 구분하는 방향" */
    public static void discriminant(Checker c) {
        c.section("판별 분석: 집단을 구분하는 방향");

        // 3장 activation steering의 설정을 모사한다. 정직한 답변일 때의 활성값과
        // 그렇지 않을 때의 활성값을 각각 모아 두 집단으로 둔다.
        Random rnd = new Random(56);
        final int n = 300;
        double[][] honest = new double[n][2];
        double[][] other = new double[n][2];
        for (int i = 0; i < n; i++) {
            // 두 집단은 첫 번째 축으로 조금 떨어져 있고, 두 번째 축으로 크게 퍼져 있다.
            honest[i] = new double[] {1 + 0.5 * rnd.nextGaussian(), 4 * rnd.nextGaussian()};
            other[i] = new double[] {-1 + 0.5 * rnd.nextGaussian(), 4 * rnd.nextGaussian()};
        }

        // 부록: 가장 단순한 형태는 두 집단의 평균 벡터를 빼서 그 차이를 방향으로 삼는 것이다.
        RealVector meanDiff = meanVector(honest).subtract(meanVector(other));
        meanDiff = meanDiff.unitVector();
        c.near("평균 차이 방향의 첫 성분", 1, Math.abs(meanDiff.getEntry(0)), 0.1);

        // 부록: Fisher의 선형 판별 분석은 집단 안의 흩어짐까지 고려한다.
        RealVector fisher = fisherDirection(honest, other);

        double sepMean = separation(honest, other, meanDiff);
        double sepFisher = separation(honest, other, fisher);
        c.ok("Fisher 방향이 더 잘 분리한다", sepFisher >= sepMean - 1e-9,
                "F비 " + Checker.num(sepFisher) + " ≥ " + Checker.num(sepMean));

        // 위 데이터는 두 번째 축의 흩어짐이 훨씬 크므로, Fisher 방향은 그 축의 기여를 더 깎아 낸다.
        c.ok("Fisher 방향이 넓게 퍼진 축을 덜 쓴다",
                Math.abs(fisher.getEntry(1)) <= Math.abs(meanDiff.getEntry(1)) + 1e-9,
                "두 번째 성분 " + Checker.num(Math.abs(fisher.getEntry(1)))
                        + " ≤ " + Checker.num(Math.abs(meanDiff.getEntry(1))));
    }

    private static RealMatrix truncatedSvd(RealMatrix a, int k) {
        SingularValueDecomposition svd = new SingularValueDecomposition(a);
        RealMatrix out = MatrixUtils.createRealMatrix(a.getRowDimension(), a.getColumnDimension());
        double[] s = svd.getSingularValues();
        for (int i = 0; i < k; i++) {
            RealMatrix u = svd.getU().getColumnMatrix(i);
            RealMatrix v = svd.getV().getColumnMatrix(i).transpose();
            out = out.add(u.multiply(v).scalarMultiply(s[i]));
        }
        return out;
    }

    /** 집단 안 공분산의 역행렬에 평균 차이를 곱한 방향을 돌려준다. */
    private static RealVector fisherDirection(double[][] g1, double[][] g2) {
        RealMatrix within = new Covariance(g1).getCovarianceMatrix()
                .add(new Covariance(g2).getCovarianceMatrix());
        RealVector diff = meanVector(g1).subtract(meanVector(g2));
        RealVector w = MatrixUtils.inverse(within).operate(diff);
        return w.unitVector();
    }

    /** 주어진 방향으로 사영했을 때 집단 사이 분산과 집단 안 분산의 비율을 잰다. */
    private static double separation(double[][] g1, double[][] g2, RealVector dir) {
        double[] p1 = project(g1, dir);
        double[] p2 = project(g2, dir);
        double gap = StatUtils.mean(p1) - StatUtils.mean(p2);
        return gap * gap / (StatUtils.variance(p1) + StatUtils.variance(p2));
    }

    private static double[] project(double[][] rows, RealVector dir) {
        double[] out = new double[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = new ArrayRealVector(rows[i]).dotProduct(dir);
        }
        return out;
    }

    private static RealVector meanVector(double[][] rows) {
        int dim = rows[0].length;
        double[] out = new double[dim];
        for (int j = 0; j < dim; j++) {
            double sum = 0;
            for (double[] row : rows) {
                sum += row[j];
            }
            out[j] = sum / rows.length;
        }
        return new ArrayRealVector(out);
    }

    static RealVector randomVector(Random rnd, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = rnd.nextGaussian();
        }
        return new ArrayRealVector(out);
    }

    private static RealMatrix randomMatrix(Random rnd, int rows, int cols) {
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = rnd.nextGaussian();
            }
        }
        return MatrixUtils.createRealMatrix(out);
    }
}
