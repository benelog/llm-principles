import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Bm25Index는 키워드 검색의 역색인이다. 질의 토큰이 희귀할수록(IDF), 청크에 자주 나올수록(TF)
 * 점수가 높고, 빈도의 기여는 포화하며, 긴 청크는 점수를 깎는다.
 */
class Bm25IndexTest {

    private static Chunk chunk(int seq, String text) {
        return new Chunk("doc", seq, text);
    }

    private static Chunk repeated(int seq, String word, int times, String filler) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) {
            sb.append(word).append(' ');
        }
        sb.append(filler);
        return chunk(seq, sb.toString());
    }

    @Test
    @DisplayName("질의 토큰이 들어 있는 청크만 결과에 오르고, 없는 토큰은 무시된다")
    void onlyMatchingChunksAreReturned() {
        Bm25Index index = new Bm25Index(List.of(
                chunk(0, "공용 와이파이에서는 VPN을 켠다"),
                chunk(1, "연차는 전자결재로 신청한다")));

        List<SearchResult> results = index.search("와이파이 VPN", 10);

        assertThat(results).extracting(r -> r.chunk().seq()).containsExactly(0);
        assertThat(index.search("존재하지않는단어", 10)).isEmpty();
    }

    @Test
    @DisplayName("희귀한 토큰이 흔한 토큰보다 점수에 더 크게 기여한다 (IDF)")
    void rareTokensWeighMore() {
        // "회사"는 모든 청크에 나오고 "보안"은 하나에만 나온다.
        Bm25Index index = new Bm25Index(List.of(
                chunk(0, "회사 규정"), chunk(1, "회사 식당"), chunk(2, "회사 보안")));

        double common = index.search("회사", 10).get(0).score();
        double rare = index.search("보안", 10).get(0).score();

        assertThat(rare).isGreaterThan(common);
    }

    @Test
    @DisplayName("같은 단어가 두 배 나와도 점수는 두 배가 되지 않는다 (빈도 포화)")
    void termFrequencySaturates() {
        // 길이 보정의 영향을 없애기 위해 두 청크를 토큰 20개로 같게 맞춘다.
        List<Chunk> chunks = List.of(
                repeated(0, "보안", 10, "지침 ".repeat(10).strip()),
                repeated(1, "보안", 20, ""));
        Bm25Index index = new Bm25Index(chunks);

        List<SearchResult> results = index.search("보안", 10);
        double score10 = results.stream().filter(r -> r.chunk().seq() == 0).findFirst().orElseThrow().score();
        double score20 = results.stream().filter(r -> r.chunk().seq() == 1).findFirst().orElseThrow().score();

        assertThat(score20).isGreaterThan(score10);
        assertThat(score20).isLessThan(2 * score10);
    }

    @Test
    @DisplayName("같은 빈도라면 평균보다 긴 청크가 낮은 점수를 받는다 (길이 보정)")
    void longerChunksArePenalized() {
        Bm25Index index = new Bm25Index(List.of(
                chunk(0, "보안 지침"),
                chunk(1, "보안 지침 그리고 아주 길게 이어지는 다른 내용들이 잔뜩 들어 있는 청크")));

        List<SearchResult> results = index.search("보안", 10);

        assertThat(results).extracting(r -> r.chunk().seq()).containsExactly(0, 1);
        assertThat(results.get(0).score()).isGreaterThan(results.get(1).score());
    }

    @Test
    @DisplayName("결과는 점수 내림차순이고 topK로 잘린다")
    void resultsAreSortedAndLimited() {
        Bm25Index index = new Bm25Index(List.of(
                chunk(0, "보안"), chunk(1, "보안 보안"), chunk(2, "보안 보안 보안"), chunk(3, "다른 내용")));

        List<SearchResult> results = index.search("보안", 2);

        assertThat(results).hasSize(2);
        assertThat(results.get(0).score()).isGreaterThanOrEqualTo(results.get(1).score());
    }
}
