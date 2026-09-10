import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Gpt는 토큰 하나를 받아 다음 토큰의 로짓을 내놓는 모델이다. 앞 위치의 키와 밸류는
 * KvCache에 쌓이므로, forward를 한 번 부를 때마다 캐시가 한 위치씩 자란다.
 *
 * 가중치는 무작위로 초기화되지만 시드를 고정했으므로 이 테스트의 값들은 재현된다.
 * 다만 "무작위 가중치의 출력"이 어떤 값인지는 시드가 정하는 우연이라서,
 * 여기서는 특정 로짓 값이 아니라 구조적 성질(크기, 합, 의존 관계, 손실의 변화 방향)만 단정한다.
 */
class GptTest {

    private static final int VOCAB = 27;

    private Gpt newModel() {
        return new Gpt(VOCAB, new Random(42));
    }

    @Test
    @DisplayName("파라미터 수는 5장에서 말한 4,192개다")
    void parameterCountMatchesTheBook() {
        assertThat(newModel().params).hasSize(4192);
    }

    @Test
    @DisplayName("forward는 어휘 크기만큼의 로짓을 내놓고, softmax는 그것을 합이 1인 확률로 바꾼다")
    void forwardProducesLogitsOverTheVocabulary() {
        Gpt model = newModel();
        Value[] logits = model.forward(0, 0, model.newCache());

        assertThat(logits).hasSize(VOCAB);

        Value[] probs = Gpt.softmax(logits);
        double sum = 0;
        for (Value p : probs) {
            assertThat(p.data).isBetween(0.0, 1.0);
            sum += p.data;
        }
        assertThat(sum).isCloseTo(1.0, within(1e-12));
    }

    @Test
    @DisplayName("softmax는 로짓의 순서를 보존하고 최댓값을 빼도 결과가 같다")
    void softmaxPreservesOrderAndIsShiftInvariant() {
        Value[] small = {new Value(2.0), new Value(1.0), new Value(0.1)};
        Value[] shifted = {new Value(1002.0), new Value(1001.0), new Value(1000.1)};

        Value[] p = Gpt.softmax(small);
        Value[] q = Gpt.softmax(shifted);

        assertThat(p[0].data).isGreaterThan(p[1].data).isGreaterThan(p[2].data);
        assertThat(p[0].data).as("7장의 계산 예시").isCloseTo(0.66, within(5e-3));
        for (int i = 0; i < 3; i++) {
            assertThat(q[i].data).isCloseTo(p[i].data, within(1e-12));
        }
    }

    @Test
    @DisplayName("forward를 한 번 부를 때마다 KvCache에 위치 하나의 키와 밸류가 쌓인다")
    void cacheGrowsOnePositionPerForward() {
        Gpt model = newModel();
        Gpt.KvCache cache = model.newCache();

        model.forward(1, 0, cache);
        model.forward(2, 1, cache);
        model.forward(3, 2, cache);

        assertThat(cache.keys).hasSize(Gpt.nLayer);
        assertThat(cache.keys.get(0)).hasSize(3);
        assertThat(cache.values.get(0)).hasSize(3);
        assertThat(cache.keys.get(0).get(0)).hasSize(Gpt.nEmbd);
    }

    @Test
    @DisplayName("현재 위치의 출력은 캐시에 쌓인 앞 토큰들에 의존한다")
    void outputDependsOnPreviousTokensThroughTheCache() {
        Gpt model = newModel();

        Gpt.KvCache afterA = model.newCache();
        model.forward(1, 0, afterA);
        Value[] logitsAfterA = model.forward(5, 1, afterA);

        Gpt.KvCache afterB = model.newCache();
        model.forward(2, 0, afterB);
        Value[] logitsAfterB = model.forward(5, 1, afterB);

        // 같은 토큰 5를 같은 위치 1에 넣었지만 앞 토큰이 다르므로 로짓이 달라야 한다.
        boolean anyDifferent = false;
        for (int i = 0; i < VOCAB; i++) {
            if (Math.abs(logitsAfterA[i].data - logitsAfterB[i].data) > 1e-12) {
                anyDifferent = true;
            }
        }
        assertThat(anyDifferent).isTrue();
    }

    @Test
    @DisplayName("rmsnorm은 벡터의 제곱 평균의 제곱근을 1로 맞춘다")
    void rmsnormScalesVectorToUnitRms() {
        Value[] x = {new Value(3), new Value(-4), new Value(12), new Value(0)};

        Value[] normed = Gpt.rmsnorm(x);

        double meanSquare = 0;
        for (Value v : normed) {
            meanSquare += v.data * v.data;
        }
        meanSquare /= normed.length;
        // 0으로 나누는 것을 막는 작은 값(1e-5) 때문에 정확히 1은 아니다.
        assertThat(Math.sqrt(meanSquare)).isCloseTo(1.0, within(1e-4));
    }

    @Test
    @DisplayName("학습 전 손실은 어휘 27개를 균등하게 찍는 수준인 ln(27) 근처다")
    void initialLossIsNearUniformGuess() {
        // 가중치를 표준편차 0.02의 작은 값으로 초기화하므로 로짓이 거의 같고 분포가 균등에 가깝다.
        // 정확한 값은 시드가 정하는 우연이라 단정할 수 없고, ln(27) ≈ 3.30 근처라는 성질만 확인한다.
        Gpt model = newModel();
        int[] seq = {26, 4, 12, 12, 0, 26}; // BOS, e, m, m, a, BOS

        assertThat(sequenceLoss(model, seq)).isCloseTo(Math.log(27), within(0.1));
    }

    @Test
    @DisplayName("경사 하강 한 걸음은 같은 시퀀스의 손실을 줄인다")
    void oneGradientStepReducesLossOnTheSameSequence() {
        // 학습이 얼마나 빨리 되는지는 데이터와 시드에 달린 문제라 단정할 수 없다.
        // 여기서는 "기울기의 반대 방향으로 조금 움직이면 손실이 줄어든다"는 방향만 확인한다.
        Gpt model = newModel();
        int[] seq = {26, 4, 12, 12, 0, 26};

        double before = sequenceLoss(model, seq);
        Value loss = sequenceLossNode(model, seq);
        for (Value p : model.params) {
            p.grad = 0;
        }
        loss.backward();
        for (Value p : model.params) {
            p.data -= 0.1 * p.grad;
        }
        double after = sequenceLoss(model, seq);

        assertThat(after).isLessThan(before);
    }

    private static double sequenceLoss(Gpt model, int[] seq) {
        return sequenceLossNode(model, seq).data;
    }

    // 5장 학습 루프의 손실 계산과 같다. 각 위치에서 정답 토큰 확률의 -log를 평균한다.
    private static Value sequenceLossNode(Gpt model, int[] seq) {
        Gpt.KvCache cache = model.newCache();
        Value loss = new Value(0);
        int n = seq.length - 1;
        for (int pos = 0; pos < n; pos++) {
            Value[] probs = Gpt.softmax(model.forward(seq[pos], pos, cache));
            loss = loss.add(probs[seq[pos + 1]].log().neg());
        }
        return loss.mul(new Value(1.0 / n));
    }
}
