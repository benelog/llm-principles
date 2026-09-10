import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * quantize 예제는 세 가지 양자화(선형, 블록 배율, 정규 분포 격자)의 오차를 잰다.
 *
 * 8장에 적힌 RMS 오차 0.1541, 0.1058 같은 숫자는 시드 8로 뽑은 표본 4,096개의 결과다.
 * 시드를 고정했으므로 재현되지만 표본이 달라지면 값도 달라지므로, 여기서는 정확한 오차 값 대신
 * 어떤 표본에서도 성립하는 성질(오차의 상한, 격자의 모양)과 방식 사이의 크기 관계만 확인한다.
 */
class QuantizeTest {

    private static double[] gaussian(int n, long seed) {
        Random rng = new Random(seed);
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = rng.nextGaussian();
        }
        return w;
    }

    @Test
    @DisplayName("선형 양자화는 최대 절댓값을 정수 범위의 끝에 대응시킨다")
    void absmaxMapsTheLargestValueToMaxQ() {
        double[] w = {0.5, -2.0, 1.0};

        Main.Quantized q = Main.quantizeAbsmax(w, 127);

        assertThat(q.scale()).isCloseTo(2.0 / 127, within(1e-12));
        assertThat(q.q()).containsExactly(32, -127, 64);
    }

    @Test
    @DisplayName("복원 오차의 최대치는 격자 간격의 절반(배율의 절반)을 넘지 않는다")
    void roundingErrorIsBoundedByHalfTheScale() {
        double[] w = gaussian(1000, 1);

        Main.Quantized q = Main.quantizeAbsmax(w, 7);
        double[] restored = Main.dequantize(q);

        for (int i = 0; i < w.length; i++) {
            assertThat(Math.abs(w[i] - restored[i])).isLessThanOrEqualTo(q.scale() / 2 + 1e-12);
        }
    }

    @Test
    @DisplayName("격자가 촘촘할수록(INT8) 오차가 작다")
    void finerGridGivesSmallerError() {
        double[] w = gaussian(4096, 8);

        double int8 = Main.rmsError(w, Main.dequantize(Main.quantizeAbsmax(w, 127)));
        double int4 = Main.rmsError(w, Main.dequantize(Main.quantizeAbsmax(w, 7)));

        assertThat(int8).isLessThan(int4);
        assertThat(int8).as("격자 255개면 오차는 값 크기의 1% 아래").isLessThan(0.01);
    }

    @Test
    @DisplayName("아웃라이어 하나가 텐서 전체의 격자를 성기게 만들고, 블록 배율이 그 피해를 블록 하나로 가둔다")
    void blockScalesContainOutlierDamage() {
        double[] w = gaussian(4096, 8);
        double[] withOutlier = w.clone();
        withOutlier[0] = 20.0;

        double clean = Main.rmsError(w, Main.dequantize(Main.quantizeAbsmax(w, 7)));
        double wholeTensor = Main.rmsError(withOutlier, Main.dequantize(Main.quantizeAbsmax(withOutlier, 7)));
        double blocks = Main.rmsError(withOutlier, Main.quantizeBlocks(withOutlier, 7, 64));

        assertThat(wholeTensor).as("아웃라이어가 오차를 키운다").isGreaterThan(clean);
        assertThat(blocks).as("블록 배율은 아웃라이어가 없던 수준보다도 낮다").isLessThan(clean);
    }

    @Test
    @DisplayName("정규 분포 분위수 격자는 좌우 대칭이고 0 근처가 균등 격자보다 촘촘하다")
    void normalGridIsSymmetricAndDenseNearZero() {
        double[] normal = Main.normalGrid(16);
        double[] uniform = Main.uniformGrid(16);

        assertThat(normal[0]).isEqualTo(-1.0);
        assertThat(normal[15]).isEqualTo(1.0);
        for (int i = 0; i < 8; i++) {
            assertThat(normal[i]).isCloseTo(-normal[15 - i], within(1e-12));
        }
        double centerGapNormal = normal[8] - normal[7];
        double centerGapUniform = uniform[8] - uniform[7];
        assertThat(centerGapNormal).isLessThan(centerGapUniform);
        assertThat(normal[15] - normal[14]).as("꼬리 쪽은 성기다").isGreaterThan(centerGapUniform);
    }

    @Test
    @DisplayName("inverseNormalCdf는 표준 정규 분포의 분위수를 낸다")
    void inverseNormalCdfKnownQuantiles() {
        assertThat(Main.inverseNormalCdf(0.5)).isCloseTo(0.0, within(1e-9));
        assertThat(Main.inverseNormalCdf(0.975)).isCloseTo(1.959964, within(1e-6));
        assertThat(Main.inverseNormalCdf(0.8413)).isCloseTo(1.0, within(1e-3));
    }

    @Test
    @DisplayName("정규 분포에서 뽑은 값에는 분위수 격자가 균등 격자보다 오차가 작다")
    void normalGridBeatsUniformGridOnGaussianData() {
        double[] w = gaussian(4096, 8);

        double uniform = Main.rmsError(w, Main.quantizeGrid(w, Main.uniformGrid(16)));
        double normal = Main.rmsError(w, Main.quantizeGrid(w, Main.normalGrid(16)));

        assertThat(normal).isLessThan(uniform);
    }

    @Test
    @DisplayName("값이 고르게 퍼진 데이터에서는 분위수 격자의 이점이 사라진다 (분포를 안다는 전제가 핵심)")
    void normalGridLosesOnUniformData() {
        Random rng = new Random(3);
        double[] w = new double[4096];
        for (int i = 0; i < w.length; i++) {
            w[i] = rng.nextDouble() * 2 - 1; // [-1, 1] 균등 분포
        }

        double uniform = Main.rmsError(w, Main.quantizeGrid(w, Main.uniformGrid(16)));
        double normal = Main.rmsError(w, Main.quantizeGrid(w, Main.normalGrid(16)));

        assertThat(uniform).isLessThan(normal);
    }
}
