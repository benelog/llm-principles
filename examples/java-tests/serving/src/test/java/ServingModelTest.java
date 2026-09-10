import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * serving 예제의 Model은 프리필(병렬)과 디코드(순차)가 같은 계산을 다른 방식으로 실행한다는 것을 보인다.
 *
 * 6장이 측정한 "병렬 프리필이 몇 배 빠르다", "캐시 없는 생성이 161배 느리다" 같은 시간 배율은
 * CPU 코어 수와 JIT 상태에 따라 달라지는 값이라 테스트로 단정할 수 없다.
 * 여기서는 시간을 재지 않고, 두 경로가 같은 결과를 낸다는 사실과 KV 캐시 크기 공식만 확인한다.
 */
class ServingModelTest {

    private final Main.Model model = new Main.Model(new Random(42));

    private static int[] prompt(int length, long seed) {
        Random rng = new Random(seed);
        int[] tokens = new int[length];
        for (int i = 0; i < length; i++) {
            tokens[i] = rng.nextInt(Main.vocabSize);
        }
        return tokens;
    }

    @Test
    @DisplayName("병렬 프리필은 한 토큰씩 순차 처리한 것과 같은 은닉 벡터와 캐시를 만든다")
    void prefillMatchesSequentialDecode() {
        int[] tokens = prompt(32, 1);

        Main.Model.KvCache seqCache = model.newCache();
        double[] hSeq = null;
        for (int pos = 0; pos < tokens.length; pos++) {
            hSeq = model.decodeStep(tokens[pos], pos, seqCache);
        }
        Main.PrefillResult result = model.prefill(tokens);

        assertThat(result.hidden()).containsExactly(hSeq);
        assertThat(result.cache().keys.get(0)).hasSize(tokens.length);
        for (int l = 0; l < Main.nLayer; l++) {
            for (int pos = 0; pos < tokens.length; pos++) {
                assertThat(result.cache().keys.get(l).get(pos))
                        .containsExactly(seqCache.keys.get(l).get(pos));
            }
        }
    }

    @Test
    @DisplayName("KV 캐시를 재사용한 생성과 매번 다시 계산한 생성은 같은 토큰 열을 낸다")
    void cachedAndUncachedGenerationAgree() {
        int[] tokens = prompt(16, 2);
        Main.PrefillResult result = model.prefill(tokens);

        int[] withCache = model.generate(result.cache(), result.hidden(), tokens.length, 8);
        int[] noCache = model.generateNoCache(tokens, 8);

        assertThat(withCache).containsExactly(noCache);
    }

    @Test
    @DisplayName("decodeStep은 위치마다 모든 층의 캐시에 키와 밸류를 하나씩 쌓는다")
    void decodeStepAppendsToEveryLayer() {
        Main.Model.KvCache cache = model.newCache();

        model.decodeStep(3, 0, cache);
        model.decodeStep(5, 1, cache);

        for (int l = 0; l < Main.nLayer; l++) {
            assertThat(cache.keys.get(l)).hasSize(2);
            assertThat(cache.values.get(l)).hasSize(2);
            assertThat(cache.keys.get(l).get(0)).hasSize(Main.nEmbd);
        }
    }

    @Test
    @DisplayName("키가 하나뿐이면 어텐션 출력은 그 밸류 그대로다 (softmax 가중치가 1)")
    void attentionOverSingleKeyReturnsItsValue() {
        double[] q = new double[Main.nEmbd];
        double[] k = new double[Main.nEmbd];
        double[] v = new double[Main.nEmbd];
        for (int i = 0; i < Main.nEmbd; i++) {
            q[i] = 0.1 * i;
            k[i] = -0.05 * i;
            v[i] = i;
        }

        double[] out = Main.attend(q, List.of(k), List.of(v));

        assertThat(out).containsExactly(v);
    }

    @Test
    @DisplayName("점수가 같은 키들 위의 어텐션 출력은 밸류들의 평균이다")
    void attentionWithEqualScoresAveragesValues() {
        double[] q = new double[Main.nEmbd]; // 0 벡터라 모든 키와의 내적이 0으로 같다
        double[] k = new double[Main.nEmbd];
        double[] v1 = new double[Main.nEmbd];
        double[] v2 = new double[Main.nEmbd];
        for (int i = 0; i < Main.nEmbd; i++) {
            v1[i] = 2;
            v2[i] = 4;
        }

        double[] out = Main.attend(q, List.of(k, k), List.of(v1, v2));

        for (double value : out) {
            assertThat(value).isCloseTo(3.0, within(1e-12));
        }
    }

    @Test
    @DisplayName("KV 캐시 크기 공식: Llama 3 8B가 8,192 토큰이면 1 GiB, 128K 토큰이면 16 GiB")
    void kvCacheBytesMatchesChapter6() {
        long gib = 1L << 30;

        assertThat(Main.kvCacheBytes(32, 8, 128, 8192, 2)).isEqualTo(gib);
        assertThat(Main.kvCacheBytes(32, 8, 128, 131072, 2)).isEqualTo(16 * gib);
        assertThat(Main.kvCacheBytes(32, 32, 128, 131072, 2)).as("GQA 없이 KV 헤드 32개").isEqualTo(64 * gib);
    }
}
