import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Rrf.fuse는 점수 단위가 다른 두 검색 결과를 순위만으로 합친다.
 * 각 목록에서 순위 r(1부터)인 문서에 1/(60 + r)을 주고 더한다.
 */
class RrfTest {

    private static final Chunk A = new Chunk("doc", 0, "a");
    private static final Chunk B = new Chunk("doc", 1, "b");
    private static final Chunk C = new Chunk("doc", 2, "c");

    @Test
    @DisplayName("두 목록 모두에 오른 문서가 한쪽에만 오른 문서보다 앞선다")
    void documentsInBothListsRankFirst() {
        List<SearchResult> lexical = List.of(new SearchResult(A, 9.0), new SearchResult(B, 3.0));
        List<SearchResult> vector = List.of(new SearchResult(C, 0.9), new SearchResult(B, 0.5));

        List<SearchResult> fused = Rrf.fuse(List.of(lexical, vector), 10);

        assertThat(fused).extracting(r -> r.chunk().seq()).containsExactly(1, 0, 2);
        assertThat(fused.get(0).score()).as("B: 1/62 + 1/62").isCloseTo(2.0 / 62, within(1e-12));
        assertThat(fused.get(1).score()).as("A: 1/61").isCloseTo(1.0 / 61, within(1e-12));
    }

    @Test
    @DisplayName("원래 점수의 크기는 무시되고 순위만 쓰인다")
    void originalScoresAreIgnored() {
        List<SearchResult> big = List.of(new SearchResult(A, 1000.0));
        List<SearchResult> small = List.of(new SearchResult(B, 0.001));

        List<SearchResult> fused = Rrf.fuse(List.of(big, small), 10);

        assertThat(fused.get(0).score()).isCloseTo(fused.get(1).score(), within(1e-12));
    }

    @Test
    @DisplayName("topK로 결과 수를 제한한다")
    void limitsToTopK() {
        List<SearchResult> lexical = List.of(new SearchResult(A, 1), new SearchResult(B, 1), new SearchResult(C, 1));

        assertThat(Rrf.fuse(List.of(lexical), 2)).hasSize(2);
    }
}
