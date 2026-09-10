import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** SearchResult.topK는 점수가 0보다 큰 청크만 골라 내림차순으로 상위 몇 개를 돌려준다. */
class SearchResultTest {

    private final List<Chunk> chunks = List.of(
            new Chunk("doc", 0, "a"), new Chunk("doc", 1, "b"), new Chunk("doc", 2, "c"), new Chunk("doc", 3, "d"));

    @Test
    @DisplayName("점수 0인 청크는 제외하고 내림차순으로 정렬한다")
    void dropsZeroScoresAndSortsDescending() {
        List<SearchResult> results = SearchResult.topK(chunks, new double[] {0.2, 0.0, 0.9, 0.5}, 10);

        assertThat(results).extracting(r -> r.chunk().seq()).containsExactly(2, 3, 0);
    }

    @Test
    @DisplayName("점수가 같으면 원래 순서를 유지한다 (안정 정렬)")
    void keepsOriginalOrderForTies() {
        List<SearchResult> results = SearchResult.topK(chunks, new double[] {0.5, 0.5, 0.9, 0.5}, 10);

        assertThat(results).extracting(r -> r.chunk().seq()).containsExactly(2, 0, 1, 3);
    }

    @Test
    @DisplayName("topK보다 결과가 많으면 앞에서 자른다")
    void truncatesToTopK() {
        List<SearchResult> results = SearchResult.topK(chunks, new double[] {0.2, 0.1, 0.9, 0.5}, 2);

        assertThat(results).extracting(r -> r.chunk().seq()).containsExactly(2, 3);
    }
}
