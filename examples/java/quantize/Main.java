// 양자화 알고리즘의 원리를 측정으로 확인하는 예제.
// 선형 양자화, 블록 단위 배율, 정규 분포 분위수 격자(NF4의 아이디어)를
// 외부 라이브러리 없이 JDK 표준 라이브러리만으로 구현한다.

import java.util.Arrays;
import java.util.Random;

public class Main {

    // tag::absmax[]
    // Quantized는 양자화 결과다. 정수 배열과 배율을 함께 저장해야 복원할 수 있다.
    record Quantized(int[] q, double scale) {}

    // quantizeAbsmax는 가장 단순한 선형(대칭) 양자화다.
    // 배열의 최대 절댓값이 정수 범위의 끝(maxQ)에 오도록 배율(scale)을 정하고,
    // 각 값을 배율로 나눠 가장 가까운 정수로 반올림한다.
    // INT8이면 maxQ=127, 4비트면 maxQ=7이다.
    static Quantized quantizeAbsmax(double[] w, int maxQ) {
        double absMax = 0.0;
        for (double v : w) {
            absMax = Math.max(absMax, Math.abs(v));
        }
        double scale = absMax / maxQ;
        int[] q = new int[w.length];
        for (int i = 0; i < w.length; i++) {
            q[i] = (int) Math.round(w[i] / scale);
        }
        return new Quantized(q, scale);
    }

    // dequantize는 정수에 배율을 도로 곱해 근사값을 복원한다.
    // 반올림에서 잃은 정보는 돌아오지 않는다. 그 차이가 양자화 오차다.
    static double[] dequantize(Quantized quantized) {
        double[] w = new double[quantized.q().length];
        for (int i = 0; i < w.length; i++) {
            w[i] = quantized.q()[i] * quantized.scale();
        }
        return w;
    }
    // end::absmax[]

    // tag::blocks[]
    // quantizeBlocks는 배열을 blockSize개씩 잘라 블록마다 따로 양자화한다.
    // 블록마다 배율 하나를 추가로 저장하는 대가로,
    // 큰 값 하나가 배열 전체의 격자를 성기게 만드는 것을 막는다.
    static double[] quantizeBlocks(double[] w, int maxQ, int blockSize) {
        double[] out = new double[w.length];
        for (int start = 0; start < w.length; start += blockSize) {
            int end = Math.min(start + blockSize, w.length);
            double[] block = dequantize(quantizeAbsmax(Arrays.copyOfRange(w, start, end), maxQ));
            System.arraycopy(block, 0, out, start, block.length);
        }
        return out;
    }
    // end::blocks[]

    // tag::normal-grid[]
    // normalGrid는 표준 정규 분포를 n개의 등확률 구간으로 나누고
    // 각 구간의 중앙 분위수를 격자점으로 삼는다. 값이 밀집한 0 근처에는
    // 격자점이 촘촘히, 값이 드문 꼬리 쪽에는 성기게 배치된다.
    // QLoRA의 NF4가 이 아이디어를 다듬은 4비트(16개 점) 격자다.
    static double[] normalGrid(int n) {
        double[] grid = new double[n];
        for (int i = 0; i < n; i++) {
            double p = (i + 0.5) / n;
            grid[i] = inverseNormalCdf(p); // 표준 정규 분포 누적 함수의 역함수
        }
        double last = grid[n - 1];
        for (int i = 0; i < n; i++) {
            grid[i] /= last; // [-1, 1] 범위로 정규화
        }
        return grid;
    }

    // uniformGrid는 [-1, 1]을 같은 간격으로 나눈 격자다.
    static double[] uniformGrid(int n) {
        double[] grid = new double[n];
        for (int i = 0; i < n; i++) {
            grid[i] = -1 + 2.0 * i / (n - 1);
        }
        return grid;
    }

    // quantizeGrid는 각 값을 최대 절댓값으로 정규화한 뒤
    // 격자에서 가장 가까운 점으로 바꾼다.
    static double[] quantizeGrid(double[] w, double[] grid) {
        double absMax = 0.0;
        for (double v : w) {
            absMax = Math.max(absMax, Math.abs(v));
        }
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            double v = w[i];
            double best = grid[0];
            for (int j = 1; j < grid.length; j++) {
                if (Math.abs(v / absMax - grid[j]) < Math.abs(v / absMax - best)) {
                    best = grid[j];
                }
            }
            out[i] = best * absMax;
        }
        return out;
    }
    // end::normal-grid[]

    // inverseNormalCdf는 표준 정규 분포 누적 함수의 역함수(분위수 함수)다.
    // JDK에는 없으므로 Acklam의 유리 함수 근사로 구현한다. 상대 오차는 1.15e-9 이하다.
    static double inverseNormalCdf(double p) {
        if (p <= 0 || p >= 1) {
            throw new IllegalArgumentException("p는 (0, 1) 사이여야 한다: " + p);
        }
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                3.754408661907416e+00};
        double pLow = 0.02425;
        double pHigh = 1 - pLow;
        if (p < pLow) { // 아래쪽 꼬리
            double q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        if (p > pHigh) { // 위쪽 꼬리
            double q = Math.sqrt(-2 * Math.log(1 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        // 중앙 영역
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }

    // rmsError는 원본과 복원값 사이의 평균 제곱근 오차를 잰다.
    static double rmsError(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum / a.length);
    }

    public static void main(String[] args) {
        // 학습된 가중치의 전형인 평균 0의 정규 분포에서 4,096개를 뽑는다.
        // 시드를 고정했으므로 실행할 때마다 같은 결과가 나온다.
        Random rng = new Random(8);
        double[] w = new double[4096];
        for (int i = 0; i < w.length; i++) {
            w[i] = rng.nextGaussian();
        }

        System.out.println("== 선형 양자화 (표준편차 1.0인 가중치 4,096개) ==");
        System.out.printf("INT8 (격자 255개):  RMS 오차 %.4f%n", rmsError(w, dequantize(quantizeAbsmax(w, 127))));
        System.out.printf("4비트 (격자 15개):  RMS 오차 %.4f%n", rmsError(w, dequantize(quantizeAbsmax(w, 7))));

        System.out.println("\n== 아웃라이어 하나가 끼면 (w[0]을 20.0으로) ==");
        double[] wo = w.clone();
        wo[0] = 20.0;
        System.out.printf("4비트, 텐서 전체에 배율 하나:  RMS 오차 %.4f%n", rmsError(wo, dequantize(quantizeAbsmax(wo, 7))));
        System.out.printf("4비트, 64개 블록마다 배율:     RMS 오차 %.4f%n", rmsError(wo, quantizeBlocks(wo, 7, 64)));

        System.out.println("\n== 격자를 어디에 둘 것인가 (아웃라이어 없는 원본, 격자 16개) ==");
        System.out.printf("균등 간격 격자:         RMS 오차 %.4f%n", rmsError(w, quantizeGrid(w, uniformGrid(16))));
        System.out.printf("정규 분포 분위수 격자:  RMS 오차 %.4f%n", rmsError(w, quantizeGrid(w, normalGrid(16))));
    }
}
