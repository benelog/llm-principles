import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * VectorIndex는 청크를 TF-IDF 희소 벡터로 바꿔 두고 코사인 유사도로 검색한다.
 * 벡터의 길이를 1로 정규화해 두므로 내적이 곧 코사인 유사도다.
 */
class VectorIndexTest {

    private final List<Chunk> chunks = List.of(
            new Chunk("doc", 0, "공용 와이파이에서는 VPN을 켠다"),
            new Chunk("doc", 1, "연차는 전자결재로 신청한다"),
            new Chunk("doc", 2, "와이파이 비밀번호는 매달 바뀐다"));
    private final VectorIndex index = new VectorIndex(chunks);

    @Test
    @DisplayName("embed는 길이가 1인 벡터를 만든다")
    void embeddingsAreUnitLength() {
        Map<String, Double> vec = index.embed(Map.of("와이", 2, "이파", 1));

        double norm = vec.values().stream().mapToDouble(v -> v * v).sum();
        assertThat(Math.sqrt(norm)).isCloseTo(1.0, within(1e-12));
    }

    @Test
    @DisplayName("청크 본문을 그대로 질의하면 코사인 유사도가 1이다")
    void identicalTextScoresOne() {
        List<SearchResult> results = index.search("연차는 전자결재로 신청한다", 1);

        assertThat(results.get(0).chunk().seq()).isEqualTo(1);
        assertThat(results.get(0).score()).isCloseTo(1.0, within(1e-12));
    }

    @Test
    @DisplayName("공통 토큰이 없는 청크는 유사도 0이라 결과에서 빠진다")
    void unrelatedChunksAreExcluded() {
        List<SearchResult> results = index.search("와이파이", 10);

        assertThat(results).extracting(r -> r.chunk().seq()).containsExactlyInAnyOrder(0, 2);
    }

    @Test
    @DisplayName("흔한 토큰보다 드문 토큰이 유사도에 더 크게 기여한다")
    void rareTokensDominateSimilarity() {
        // "와이파이"는 두 청크에 나오고 "VPN"은 하나에만 나온다.
        List<SearchResult> results = index.search("VPN 와이파이", 10);

        assertThat(results.get(0).chunk().seq()).isEqualTo(0);
        assertThat(results.get(0).score()).isGreaterThan(results.get(1).score());
    }
}
