import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Main의 샘플링 보조 함수들을 다룬다. softmaxData는 temperature를 적용한 확률 분포를 만들고,
 * sampleFrom은 그 분포에서 토큰 하나를 뽑는다.
 *
 * 분포를 계산하는 부분은 결정적이라 정확히 단정할 수 있다. 반면 뽑기의 결과는 확률적이다.
 * 시드를 고정하면 재현되지만, "확률 0.66인 토큰이 약 66% 뽑힌다"는 주장은 대수의 법칙에 따른
 * 근사이므로 허용 범위를 두고 확인한다. 시드를 바꾸면 이 범위를 벗어날 확률이 작게나마 있다.
 */
class SamplingTest {

    private static Value[] logits(double... values) {
        Value[] out = new Value[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = new Value(values[i]);
        }
        return out;
    }

    @Test
    @DisplayName("temperature 1의 softmax는 7장의 계산 예시와 같다")
    void softmaxAtTemperatureOne() {
        double[] p = Main.softmaxData(logits(2.0, 1.0, 0.1), 1.0);

        assertThat(p[0]).isCloseTo(0.66, within(5e-3));
        assertThat(p[1]).isCloseTo(0.24, within(5e-3));
        assertThat(p[2]).isCloseTo(0.10, within(5e-3));
    }

    @Test
    @DisplayName("temperature가 낮으면 분포가 뾰족해지고 높으면 평평해진다")
    void temperatureChangesTheShape() {
        double[] cold = Main.softmaxData(logits(2.0, 1.0, 0.1), 0.5);
        double[] hot = Main.softmaxData(logits(2.0, 1.0, 0.1), 2.0);

        assertThat(cold[0]).isCloseTo(0.86, within(5e-3));
        assertThat(hot[0]).isCloseTo(0.50, within(5e-3));
        assertThat(cold[2]).isLessThan(hot[2]);
    }

    @Test
    @DisplayName("sampleFrom은 누적 확률 구간에 따라 토큰을 고른다")
    void sampleFromUsesCumulativeProbability() {
        double[] probs = {0.2, 0.5, 0.3};

        // 난수 값을 직접 지정해 구간 경계를 확인한다.
        assertThat(Main.sampleFrom(probs, fixed(0.10))).isEqualTo(0);
        assertThat(Main.sampleFrom(probs, fixed(0.25))).isEqualTo(1);
        assertThat(Main.sampleFrom(probs, fixed(0.69))).isEqualTo(1);
        assertThat(Main.sampleFrom(probs, fixed(0.71))).isEqualTo(2);
    }

    @Test
    @DisplayName("많이 뽑으면 빈도가 확률에 가까워진다 (통계적 주장이라 허용 범위를 둔다)")
    void frequenciesApproachProbabilities() {
        double[] probs = Main.softmaxData(logits(2.0, 1.0, 0.1), 1.0);
        Random rng = new Random(7);
        int draws = 20_000;
        int[] counts = new int[3];
        for (int i = 0; i < draws; i++) {
            counts[Main.sampleFrom(probs, rng)]++;
        }

        // 표준 오차 sqrt(p(1-p)/n)은 p = 0.66, n = 20,000에서 약 0.0034다.
        // 허용 범위 0.02는 표준 오차의 약 6배라서 시드가 달라도 실패할 확률은 무시할 만하지만 0은 아니다.
        assertThat((double) counts[0] / draws).isCloseTo(probs[0], within(0.02));
        assertThat((double) counts[1] / draws).isCloseTo(probs[1], within(0.02));
        assertThat((double) counts[2] / draws).isCloseTo(probs[2], within(0.02));
    }

    private static Random fixed(double value) {
        return new Random() {
            @Override
            public double nextDouble() {
                return value;
            }
        };
    }
}
