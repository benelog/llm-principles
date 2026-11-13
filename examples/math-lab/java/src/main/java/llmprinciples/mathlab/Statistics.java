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

/** 7장의 평균과 분산 절, 부록 통계 참고의 정규 분포·공분산·k-평균 절을 검산한다. */
public final class Statistics {

    private Statistics() {
    }

    /** 7장 "평균, 분산, 표준편차"와 "정규화와 sqrt(d) 스케일링" */
    public static void moments(Checker c) {
        c.section("평균, 분산, 정규화와 sqrt(d) 스케일링");

        // 7장: [2, 4, 4, 4, 5, 5, 7, 9]는 평균 5, 분산 4, 표준편차 2다.
        double[] sample = {2, 4, 4, 4, 5, 5, 7, 9};
        double mean = 0;
        for (double v : sample) {
            mean += v;
        }
        mean /= sample.length;
        double var = 0;
        for (double v : sample) {
            var += (v - mean) * (v - mean);
        }
        var /= sample.length;
        c.near("예제 값들의 평균", 5, mean, 1e-12);
        c.near("예제 값들의 분산 (편차 제곱의 평균)", 4, var, 1e-12);
        c.near("예제 값들의 표준편차", 2, Math.sqrt(var), 1e-12);

        // 7장: 편차를 그대로 평균 내면 양수와 음수가 상쇄되어 항상 0이 된다.
        double deviationSum = 0;
        for (double v : sample) {
            deviationSum += v - mean;
        }
        c.near("편차의 합", 0, deviationSum, 1e-12);

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

        // 7장: 평균 0, 분산 1인 값으로 이루어진 d차원 벡터 두 개를 내적하면
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

        // 7장: 내적을 sqrt(d)로 나누면 차원이 몇이든 점수의 분산이 1 근처로 유지된다.
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

        // 7장: 값을 c로 나누면 표준편차는 c로, 분산은 c²으로 나뉜다.
        double[] quartered = Arrays.stream(x).map(value -> value / 4).toArray();
        c.near("값을 4로 나눈 뒤의 분산 = 분산 / 16",
                StatUtils.variance(x) / 16, StatUtils.variance(quartered), 1e-12);
        c.near("값을 4로 나눈 뒤의 표준편차 = 표준편차 / 4",
                Math.sqrt(StatUtils.variance(x)) / 4, Math.sqrt(StatUtils.variance(quartered)), 1e-12);

        // 7장: d가 256이면 나누기 전 점수의 표준편차가 sqrt(256) = 16이고,
        // 점수 차이 16은 softmax에서 e^16, 약 890만 배의 확률 비율이 된다.
        c.near("sqrt(256)", 16, Math.sqrt(256), 1e-12);
        c.near("e^16", 890e4, Math.exp(16), 5e4);

        // 7장: x = [3, 4]이면 제곱 평균은 12.5, RMS는 약 3.54이고 나눈 결과는 약 [0.85, 1.13]이다.
        // 결과의 제곱 평균은 (0.72 + 1.28) / 2 = 1이고, 성분 사이의 비율 3 대 4는 그대로다.
        RealVector pair = new ArrayRealVector(new double[] {3, 4});
        double meanSquare = pair.dotProduct(pair) / pair.getDimension();
        RealVector pairNormed = pair.mapDivide(Math.sqrt(meanSquare));
        c.near("[3, 4]의 제곱 평균", 12.5, meanSquare, 1e-12);
        c.near("[3, 4]의 RMS", 3.54, Math.sqrt(meanSquare), 5e-3);
        c.near("정규화한 첫 성분", 0.85, pairNormed.getEntry(0), 5e-3);
        c.near("정규화한 둘째 성분", 1.13, pairNormed.getEntry(1), 5e-3);
        c.near("정규화한 첫 성분의 제곱", 0.72, Math.pow(pairNormed.getEntry(0), 2), 1e-12);
        c.near("정규화한 둘째 성분의 제곱", 1.28, Math.pow(pairNormed.getEntry(1), 2), 1e-12);
        c.near("성분 사이의 비율 3 대 4 유지", 3.0 / 4,
                pairNormed.getEntry(0) / pairNormed.getEntry(1), 1e-12);

        // 7장: 5장의 RMSNorm은 벡터를 자기 크기로 나눠 규모를 되돌린다.
        RealVector v = new ArrayRealVector(new double[] {3, -4, 12, 0.5});
        double rms = v.getNorm() / Math.sqrt(v.getDimension());
        RealVector normed = v.mapDivide(rms);
        c.near("RMSNorm 뒤의 RMS", 1,
                normed.getNorm() / Math.sqrt(normed.getDimension()), 1e-12);
    }

    /** 부록 "정규 분포와 중심 극한 정리" */
    public static void normalDistribution(Checker c) {
        c.section("정규 분포와 중심 극한 정리");

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

        // 부록: 주사위 두 개를 던진 합은 7이 되는 경우가 여섯 가지로 가장 많고,
        // 2나 12가 되는 경우는 한 가지뿐이라 가운데가 높은 삼각형 모양이 된다.
        int[] ways = new int[13];
        for (int first = 1; first <= 6; first++) {
            for (int second = 1; second <= 6; second++) {
                ways[first + second]++;
            }
        }
        boolean sevenIsMost = true;
        for (int total = 2; total <= 12; total++) {
            if (total != 7 && ways[total] >= ways[7]) {
                sevenIsMost = false;
            }
        }
        c.near("주사위 두 개의 합이 7인 경우의 수", 6, ways[7], 0);
        c.near("합이 2인 경우의 수", 1, ways[2], 0);
        c.near("합이 12인 경우의 수", 1, ways[12], 0);
        c.ok("7이 가장 흔한 합", sevenIsMost, "합 2부터 12까지의 경우의 수 " + Arrays.toString(Arrays.copyOfRange(ways, 2, 13)));

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

        // 부록: 표준 정규 분포를 확률이 같은 16개 구간으로 나누면 0 근처 구간의 폭은
        // 약 0.16이고 가장 바깥 경계는 ±1.53이라서 그 너머의 꼬리(각 6.25%)가 구간 하나에 담긴다.
        double[] cuts = new double[15];
        for (int i = 0; i < 15; i++) {
            cuts[i] = n.inverseCumulativeProbability((i + 1) / 16.0);
        }
        c.near("0 근처 구간의 폭", 0.16, cuts[7] - cuts[6], 5e-3);
        c.near("가장 바깥 경계", 1.53, cuts[14], 5e-3);
        c.near("바깥 경계 너머의 꼬리 비율", 0.0625, 1 - n.cumulativeProbability(cuts[14]), 1e-9);
        c.ok("경계 간격은 0에서 멀어질수록 넓어짐", cuts[14] - cuts[13] > cuts[7] - cuts[6],
                "바깥 " + Checker.num(cuts[14] - cuts[13]) + " > 중앙 " + Checker.num(cuts[7] - cuts[6]));

        // 부록: 8장 NF4는 균등 간격 대신 정규 분포의 분위수에 격자를 배치해서
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

    /** 부록 "공분산과 상관계수" */
    public static void covarianceAndCorrelation(Checker c) {
        c.section("공분산과 상관계수");

        // 부록: x = [1, 2, 3, 4](평균 2.5), y = [2, 4, 5, 8](평균 4.75)이면 편차의 곱은
        // [4.125, 0.375, 0.125, 4.875], 그 평균인 공분산은 2.375다.
        // σ_x ≈ 1.118, σ_y ≈ 2.165이고 상관계수는 2.375 / (1.118 * 2.165) ≈ 0.98이다.
        double[] ex = {1, 2, 3, 4};
        double[] ey = {2, 4, 5, 8};
        double meanX = StatUtils.mean(ex);
        double meanY = StatUtils.mean(ey);
        c.near("x의 평균", 2.5, meanX, 1e-12);
        c.near("y의 평균", 4.75, meanY, 1e-12);

        double[] expectedProducts = {4.125, 0.375, 0.125, 4.875};
        boolean productsMatch = true;
        double productSum = 0;
        for (int i = 0; i < ex.length; i++) {
            double product = (ex[i] - meanX) * (ey[i] - meanY);
            productSum += product;
            if (Math.abs(product - expectedProducts[i]) > 1e-12 || product <= 0) {
                productsMatch = false;
            }
        }
        c.ok("편차의 곱", productsMatch, "[4.125, 0.375, 0.125, 4.875], 네 쌍 모두 양수");

        double exampleCov = productSum / ex.length;
        double sigmaX = Math.sqrt(StatUtils.populationVariance(ex));
        double sigmaY = Math.sqrt(StatUtils.populationVariance(ey));
        double exampleCorr = exampleCov / (sigmaX * sigmaY);
        c.near("편차의 곱의 평균 cov(x, y)", 2.375, exampleCov, 1e-12);
        c.near("σ_x", 1.118, sigmaX, 5e-4);
        c.near("σ_y", 2.165, sigmaY, 5e-4);
        c.near("예제의 상관계수", 0.98, exampleCorr, 5e-3);
        c.near("손으로 구한 상관계수 = 라이브러리 값",
                new PearsonsCorrelation().correlation(ex, ey), exampleCorr, 1e-12);

        // 부록: y를 10배 큰 단위로 적으면 공분산도 10배가 되지만 상관계수는 그대로다.
        double[] ey10 = Arrays.stream(ey).map(value -> value * 10).toArray();
        c.near("y를 10배 하면 공분산도 10배", 10 * exampleCov,
                new Covariance().covariance(ex, ey10, false), 1e-9);
        c.near("y를 10배 해도 상관계수는 그대로", exampleCorr,
                new PearsonsCorrelation().correlation(ex, ey10), 1e-12);

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
        // 정확히 같은 계산이다. 11장의 벡터 검색에 쓴 그 코사인이다.
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

    /** 부록 "k-평균 군집화" */
    public static void kmeans(Checker c) {
        c.section("k-평균 군집화");

        // 부록: 1차원 값 [1, 2, 3, 10, 11, 12]에 대표점 두 개를 1과 2에 놓고 시작하면
        // 1회 뒤 1과 7.6(38 / 5), 2회 뒤 2와 11이 되고, 3회째에는 배정이 바뀌지 않아 멈춘다.
        double[] values = {1, 2, 3, 10, 11, 12};
        double[] centers = lloydStep(values, new double[] {1, 2});
        c.near("1회 뒤 첫째 대표점", 1, centers[0], 1e-12);
        c.near("1회 뒤 둘째 대표점 (38 / 5)", 7.6, centers[1], 1e-12);
        centers = lloydStep(values, centers);
        c.near("2회 뒤 첫째 대표점", 2, centers[0], 1e-12);
        c.near("2회 뒤 둘째 대표점", 11, centers[1], 1e-12);
        c.ok("3회째에는 대표점이 그대로", Arrays.equals(lloydStep(values, centers), centers),
                "배정이 바뀌지 않아 멈춘다");

        // 부록: 대표점 k개를 놓고 "가장 가까운 대표점에 배정한다, 대표점을 배정된
        // 데이터의 평균으로 옮긴다"를 반복하면 대표점이 밀집한 곳으로 이동한다.
        // Commons Math는 k-평균을 직접 제공한다.
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
        // 8장의 NF4 자리에서 정규 분포 샘플에 대해 균등 격자와 비교한다.
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

    /** Lloyd 알고리즘의 한 반복이다. 각 값을 가장 가까운 대표점에 배정하고, 대표점을 배정된 값들의 평균으로 옮긴다. */
    private static double[] lloydStep(double[] values, double[] centers) {
        double[] sums = new double[centers.length];
        int[] counts = new int[centers.length];
        for (double v : values) {
            int k = nearest(centers, v);
            sums[k] += v;
            counts[k]++;
        }
        double[] next = new double[centers.length];
        for (int k = 0; k < centers.length; k++) {
            next[k] = counts[k] == 0 ? centers[k] : sums[k] / counts[k];
        }
        return next;
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
