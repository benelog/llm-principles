import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Trace는 forward 한 번의 중간값(어텐션 점수와 가중치, 밸류의 가중합, 로짓)을 기록한다.
 * 기록은 Gpt.forward 안에서 계산된 값을 그대로 옮긴 것이어야 하고, 기록 여부가 계산을 바꾸면 안 된다.
 *
 * 2장 본문의 수치(a의 확률 0.232 등)는 1,500 스텝 학습을 마친 모델의 값이다. 학습은 시간이 걸리고
 * 그 값은 시드가 정한 우연이므로, 여기서는 학습하지 않은 모델로 어떤 가중치에서도 성립하는 관계만 확인한다.
 */
class TraceTest {

    private static final int VOCAB = 27;
    private static final int[] INPUT = {26, 4, 12, 12}; // BOS, e, m, m

    private Gpt newModel() {
        return new Gpt(VOCAB, new Random(42));
    }

    @Test
    @DisplayName("헤드마다 입력 위치 수만큼의 점수와, 합이 1인 가중치가 기록된다")
    void recordsScoresAndWeightsForEveryPosition() {
        Trace trace = Trace.capture(newModel(), INPUT);

        for (int h = 0; h < Gpt.nHead; h++) {
            assertThat(trace.scores[h]).hasSize(INPUT.length);
            assertThat(trace.weights[h]).hasSize(INPUT.length);
            double sum = 0;
            for (double w : trace.weights[h]) {
                assertThat(w).isBetween(0.0, 1.0);
                sum += w;
            }
            assertThat(sum).isCloseTo(1.0, within(1e-12));
        }
    }

    @Test
    @DisplayName("가중치는 점수의 softmax다: 점수가 큰 위치가 큰 가중치를 받고, 비율은 exp(점수 차이)다")
    void weightsAreSoftmaxOfScores() {
        Trace trace = Trace.capture(newModel(), INPUT);

        for (int h = 0; h < Gpt.nHead; h++) {
            double[] s = trace.scores[h];
            double[] w = trace.weights[h];
            for (int t = 1; t < s.length; t++) {
                assertThat(w[t] / w[0]).isCloseTo(Math.exp(s[t] - s[0]), within(1e-9));
            }
        }
    }

    @Test
    @DisplayName("어텐션 출력은 캐시에 쌓인 밸류를 가중치로 섞은 가중합이다")
    void attentionOutputIsWeightedSumOfCachedValues() {
        Trace trace = Trace.capture(newModel(), INPUT);
        List<Value[]> values = trace.cache.values.get(0);

        assertThat(values).hasSize(INPUT.length);
        for (int h = 0; h < Gpt.nHead; h++) {
            for (int d = 0; d < Gpt.headDim; d++) {
                double expected = 0;
                for (int t = 0; t < INPUT.length; t++) {
                    expected += trace.weights[h][t] * values.get(t)[h * Gpt.headDim + d].data;
                }
                assertThat(trace.attnOut[h][d]).isCloseTo(expected, within(1e-12));
            }
        }
    }

    @Test
    @DisplayName("기록을 켜도 로짓은 기록 없이 계산한 값과 같고, 끝나면 기록이 꺼진다")
    void recordingDoesNotChangeTheComputation() {
        Gpt model = newModel();
        Trace trace = Trace.capture(model, INPUT);

        assertThat(model.trace).isNull();

        Gpt.KvCache cache = model.newCache();
        Value[] plain = null;
        for (int pos = 0; pos < INPUT.length; pos++) {
            plain = model.forward(INPUT[pos], pos, cache);
        }
        assertThat(trace.logits).hasSize(VOCAB);
        for (int i = 0; i < VOCAB; i++) {
            assertThat(trace.logits[i].data).isEqualTo(plain[i].data);
        }
    }
}
