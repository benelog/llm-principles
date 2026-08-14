package llmprinciples.mathlab;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/** 부록의 기술통계와 분포 관련 절을 검산한다. */
public final class Statistics {

    private Statistics() {
    }

    /** 부록 "평균, 분산, 정규화" */
    public static void moments(Checker c) {
        c.section("평균, 분산, 정규화");

        double[] x = {2, 4, 4, 4, 5, 5, 7, 9};
        c.near("평균", 5, StatUtils.mean(x), 1e-12);
        // 표본 분산은 n-1로 나누므로 모분산 4와는 값이 다르다.
        c.near("표본 분산", 32.0 / 7, StatUtils.variance(x), 1e-12);

        DescriptiveStatistics ds = new DescriptiveStatistics(x);
        c.near("표준편차는 분산의 제곱근",
                Math.sqrt(StatUtils.variance(x)), ds.getStandardDeviation(), 1e-12);

        // 부록: 분산은 자기 자신과의 공분산이다.
        c.near("분산 = 자기 자신과의 공분산",
                StatUtils.variance(x), new Covariance().covariance(x, x), 1e-12);

        // 부록: 평균 0, 분산 1인 값으로 이루어진 d차원 벡터 두 개를 내적하면
        // 결과의 분산이 d가 되고 표준편차는 sqrt(d)가 된다.
        Random rnd = new Random(78);
        final int d = 64;
        final int trials = 20000;
        double[] dots = new double[trials];
        for (int i = 0; i < trials; i++) {
            dots[i] = LinearAlgebra.randomVector(rnd, d)
                    .dotProduct(LinearAlgebra.randomVector(rnd, d));
        }
        c.near("d차원 내적의 분산 (d=64)", 64, StatUtils.variance(dots), 3);
        c.near("d차원 내적의 표준편차", 8, Math.sqrt(StatUtils.variance(dots)), 0.3);

        // 부록: 내적을 sqrt(d)로 나누면 차원이 몇이든 점수의 분산이 1 근처로 유지된다.
        double[] scaled = Arrays.stream(dots).map(v -> v / Math.sqrt(d)).toArray();
        c.near("sqrt(d)로 나눈 뒤의 분산", 1, StatUtils.variance(scaled), 0.05);

        // 차원을 256으로 바꿔도 같은 결론이 나오는지 확인한다.
        final int d2 = 256;
        double[] dots2 = new double[trials];
        for (int i = 0; i < trials; i++) {
            dots2[i] = LinearAlgebra.randomVector(rnd, d2)
                    .dotProduct(LinearAlgebra.randomVector(rnd, d2)) / Math.sqrt(d2);
        }
        c.near("차원을 256으로 바꿔도 분산 1", 1, StatUtils.variance(dots2), 0.05);

        // 부록: 5장의 RMSNorm은 벡터를 자기 크기로 나눠 규모를 되돌린다.
        RealVector v = new ArrayRealVector(new double[] {3, -4, 12, 0.5});
        double rms = v.getNorm() / Math.sqrt(v.getDimension());
        RealVector normed = v.mapDivide(rms);
        c.near("RMSNorm 뒤의 RMS", 1,
                normed.getNorm() / Math.sqrt(normed.getDimension()), 1e-12);
    }

    /** 부록 "정규 분포: 무작위가 합쳐질 때 나타나는 모양" */
    public static void normalDistribution(Checker c) {
        c.section("정규 분포: 무작위가 합쳐질 때 나타나는 모양");

        NormalDistribution n = new NormalDistribution(0, 1);

        // 부록: 값의 약 68%가 평균 ±1 표준편차 안에, 약 95%가 ±2 표준편차 안에 들어온다.
        double within1 = n.cumulativeProbability(1) - n.cumulativeProbability(-1);
        double within2 = n.cumulativeProbability(2) - n.cumulativeProbability(-2);
        c.near("±1 표준편차 안의 비율", 0.68, within1, 0.005);
        c.near("±2 표준편차 안의 비율", 0.95, within2, 0.005);
        c.note("정확한 값은 %.4f와 %.4f다", within1, within2);

        // 부록: 평균을 중심으로 좌우 대칭이다.
        c.near("좌우 대칭 (CDF(0) = 0.5)", 0.5, n.cumulativeProbability(0), 1e-12);
        c.near("대칭성 P(X<-1.5) = P(X>1.5)", n.cumulativeProbability(-1.5),
                1 - n.cumulativeProbability(1.5), 1e-12);

        // 부록: 중심 극한 정리. 개별 값이 어떤 분포를 따르든 여러 개를 더하면
        // 그 합은 정규 분포에 가까워진다. 균등 분포 12개의 합으로 확인한다.
        Random rnd = new Random(910);
        double[] sums12 = uniformSums(rnd, 12, 200000);
        c.near("균등 분포 12개 합의 평균", 0, StatUtils.mean(sums12), 0.02);
        c.near("균등 분포 12개 합의 분산", 1, StatUtils.variance(sums12), 0.03);

        DescriptiveStatistics ds12 = new DescriptiveStatistics(sums12);
        // 왜도는 좌우 대칭이면 0이다. 균등 분포의 합은 대칭이므로 항이 몇 개든 0이다.
        c.near("합의 왜도 (정규 분포는 0)", 0, ds12.getSkewness(), 0.03);

        // 초과 첨도는 정규 분포에서 0이지만, 유한한 항의 합은 아직 거기에 닿지 않는다.
        // 균등 분포 n개 합의 이론값은 -6/(5n)이므로 12개면 -0.1이다. 정규 분포에
        // "가까워진다"는 것이지 같아진다는 뜻이 아니라는 사실이 이 값에 드러난다.
        double k12 = ds12.getKurtosis();
        c.near("12개 합의 초과 첨도 (이론값 -6/60)", -0.1, k12, 0.02);

        // 항을 늘리면 그 값이 0으로 다가간다. 이것이 중심 극한 정리의 내용이다.
        double k48 = new DescriptiveStatistics(uniformSums(rnd, 48, 200000)).getKurtosis();
        c.ok("항을 늘리면 정규 분포에 더 가까워짐", Math.abs(k48) < Math.abs(k12),
                "12개 " + Checker.num(k12) + " → 48개 " + Checker.num(k48));

        // 부록: 7장 NF4는 균등 간격 대신 정규 분포의 분위수에 격자를 배치해서
        // 같은 비트 수로 오차를 줄인다. 2비트(격자 4개)로 직접 비교한다.
        double[] sample = new double[20000];
        for (int i = 0; i < sample.length; i++) {
            sample[i] = rnd.nextGaussian();
        }
        double[] uniformGrid = {-3, -1, 1, 3};
        double[] quantileGrid = new double[4];
        for (int i = 0; i < 4; i++) {
            // 구간을 넷으로 나눈 각 구간의 가운데 분위수를 격자로 쓴다.
            quantileGrid[i] = n.inverseCumulativeProbability((i + 0.5) / 4);
        }
        double errUniform = quantizeError(sample, uniformGrid);
        double errQuantile = quantizeError(sample, quantileGrid);
        c.ok("분위수 격자가 균등 격자보다 오차가 작다", errQuantile < errUniform,
                Checker.num(errQuantile) + " < " + Checker.num(errUniform));
        c.note("분위수 격자: %s", Arrays.toString(rounded(quantileGrid)));
    }

    /** 부록 "공분산과 상관: 함께 움직이는 정도" */
    public static void covarianceAndCorrelation(Checker c) {
        c.section("공분산과 상관: 함께 움직이는 정도");

        Random rnd = new Random(1112);
        final int n = 500;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextGaussian();
            y[i] = 2 * x[i] + 0.8 * rnd.nextGaussian(); // 같은 방향으로 함께 움직인다
        }

        double cov = new Covariance().covariance(x, y);
        double corr = new PearsonsCorrelation().correlation(x, y);
        c.ok("함께 커지면 공분산이 양수", cov > 0, "공분산 " + Checker.num(cov));
        c.ok("상관계수는 -1에서 1 사이", corr > 0 && corr < 1, "상관계수 " + Checker.num(corr));

        // 부록: 공분산을 두 표준편차로 나눈 것이 상관계수다.
        double sdX = new DescriptiveStatistics(x).getStandardDeviation();
        double sdY = new DescriptiveStatistics(y).getStandardDeviation();
        c.near("상관계수 = 공분산 / (표준편차 곱)", cov / (sdX * sdY), corr, 1e-12);

        // 부록의 핵심 주장: 상관계수는 "평균을 뺀 두 값 배열의 코사인 유사도"와
        // 정확히 같은 계산이다. 10장의 벡터 검색에 쓴 그 코사인이다.
        RealVector cx = new ArrayRealVector(x).mapSubtract(StatUtils.mean(x));
        RealVector cy = new ArrayRealVector(y).mapSubtract(StatUtils.mean(y));
        c.near("상관계수 = 중심화한 벡터의 코사인", corr, cx.cosine(cy), 1e-12);

        // 부록: 반대로 움직이면 음수, 무관하면 0 근처다.
        double[] negated = Arrays.stream(y).map(v -> -v).toArray();
        double negCorr = new PearsonsCorrelation().correlation(x, negated);
        c.ok("반대로 움직이면 음수", negCorr < 0, Checker.num(negCorr));

        double[] independent = new double[n];
        for (int i = 0; i < n; i++) {
            independent[i] = rnd.nextGaussian();
        }
        c.near("무관하면 0 근처", 0,
                new PearsonsCorrelation().correlation(x, independent), 0.1);

        // 부록: 변수가 여러 개면 공분산 행렬이 되고, 주성분은 이 행렬에서 계산된다.
        double[][] data = new double[n][2];
        for (int i = 0; i < n; i++) {
            data[i][0] = x[i];
            data[i][1] = y[i];
        }
        RealMatrix covMatrix = new Covariance(data).getCovarianceMatrix();
        c.near("공분산 행렬의 (0,1) 성분", cov, covMatrix.getEntry(0, 1), 1e-12);

        // 분산이 가장 큰 방향은 공분산 행렬의 최대 고유벡터다.
        EigenDecomposition eig = new EigenDecomposition(covMatrix);
        RealVector top = eig.getEigenvector(0);
        double[] eigenvalues = eig.getRealEigenvalues();
        c.ok("최대 고유값이 첫 번째", eigenvalues[0] >= eigenvalues[1],
                Checker.num(eigenvalues[0]) + " ≥ " + Checker.num(eigenvalues[1]));

        // 그 방향으로 사영한 값의 분산이 최대 고유값과 같아야 한다.
        double[] projected = new double[n];
        for (int i = 0; i < n; i++) {
            projected[i] = new ArrayRealVector(data[i]).dotProduct(top);
        }
        c.near("제1주성분 방향의 분산 = 최대 고유값",
                eigenvalues[0], StatUtils.variance(projected), 1e-9);
    }

    /** 부록 "k-평균 군집화: 데이터에서 대표점 찾기" */
    public static void kmeans(Checker c) {
        c.section("k-평균 군집화: 데이터에서 대표점 찾기");

        // 부록: 대표점 k개를 놓고 "가장 가까운 대표점에 배정한다, 대표점을 배정된
        // 데이터의 평균으로 옮긴다"를 반복하면 대표점이 밀집한 곳으로 이동한다.
        // Commons Math는 Gonum과 달리 k-평균을 직접 제공한다.
        Random rnd = new Random(1314);
        double[] trueCenters = {-5, 0, 7};
        List<DoublePoint> points = new ArrayList<>();
        for (double m : trueCenters) {
            for (int i = 0; i < 400; i++) {
                points.add(new DoublePoint(new double[] {m + 0.6 * rnd.nextGaussian()}));
            }
        }

        double[] found = clusterCenters(points, 3, 100);
        double maxGap = 0;
        for (int i = 0; i < trueCenters.length; i++) {
            maxGap = Math.max(maxGap, Math.abs(found[i] - trueCenters[i]));
        }
        c.ok("대표점이 밀집한 곳으로 이동", maxGap < 0.15,
                "참값과 최대 차이 " + Checker.num(maxGap));
        c.note("찾은 대표점 %s", Arrays.toString(rounded(found)));

        // 부록: 분포를 모른 채 데이터에서 직접 격자를 찾는 일반해가 1차원 k-평균이고,
        // 값이 밀집한 곳에 대표점이 몰리는 최적의 양자화 격자가 같은 절차로 나온다.
        // 7장의 NF4 자리에서 정규 분포 샘플에 대해 균등 격자와 비교한다.
        double[] sample = new double[20000];
        List<DoublePoint> samplePoints = new ArrayList<>(sample.length);
        for (int i = 0; i < sample.length; i++) {
            sample[i] = rnd.nextGaussian();
            samplePoints.add(new DoublePoint(new double[] {sample[i]}));
        }
        double[] uniformGrid = {-3, -1, 1, 3};
        double[] learned = clusterCenters(samplePoints, 4, 200);

        double errUniform = quantizeError(sample, uniformGrid);
        double errLearned = quantizeError(sample, learned);
        c.ok("k-평균 격자가 균등 격자보다 오차가 작다", errLearned < errUniform,
                Checker.num(errLearned) + " < " + Checker.num(errUniform));
        c.note("k-평균이 찾은 격자: %s", Arrays.toString(rounded(learned)));

        // 값이 밀집한 0 근처에 대표점이 몰렸는지 확인한다.
        double inner = learned[2] - learned[1];
        double outer = learned[3] - learned[2];
        c.ok("밀집한 곳의 격자 간격이 더 좁다", inner < outer,
                "가운데 간격 " + Checker.num(inner) + " < 바깥 간격 " + Checker.num(outer));
    }

    private static double[] clusterCenters(List<DoublePoint> points, int k, int maxIterations) {
        KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(
                k, maxIterations, new EuclideanDistance(), new JDKRandomGenerator(42));
        List<CentroidCluster<DoublePoint>> clusters = clusterer.cluster(points);
        double[] centers = new double[k];
        for (int i = 0; i < k; i++) {
            centers[i] = clusters.get(i).getCenter().getPoint()[0];
        }
        Arrays.sort(centers);
        return centers;
    }

    /** 각 값을 가장 가까운 격자점으로 바꿨을 때의 평균 제곱 오차다. */
    static double quantizeError(double[] values, double[] grid) {
        double total = 0;
        for (double v : values) {
            double d = v - grid[nearest(grid, v)];
            total += d * d;
        }
        return total / values.length;
    }

    private static int nearest(double[] grid, double v) {
        int best = 0;
        double bestDist = Double.POSITIVE_INFINITY;
        for (int i = 0; i < grid.length; i++) {
            double d = Math.abs(grid[i] - v);
            if (d < bestDist) {
                best = i;
                bestDist = d;
            }
        }
        return best;
    }

    private static double[] uniformSums(Random rnd, int terms, int trials) {
        double[] out = new double[trials];
        for (int i = 0; i < trials; i++) {
            double s = 0;
            for (int j = 0; j < terms; j++) {
                s += rnd.nextDouble() - 0.5; // 평균 0, 분산 1/12
            }
            out[i] = s;
        }
        return out;
    }

    private static double[] rounded(double[] values) {
        return Arrays.stream(values).map(v -> Math.round(v * 1000) / 1000.0).toArray();
    }
}
