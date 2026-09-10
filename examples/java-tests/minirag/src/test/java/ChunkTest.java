import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Chunk는 검색과 인용의 단위다. 문장 경계를 먼저 찾고, 문장을 자르지 않은 채
 * 최대 길이 이하가 되도록 이어 붙인다.
 */
class ChunkTest {

    @Test
    @DisplayName("마침표 뒤의 공백이나 줄바꿈을 문장 경계로 본다")
    void splitsAtSentenceBoundaries() {
        List<String> sentences = Chunk.splitSentences("첫 문장이다. 둘째 문장이다!\n셋째 문장인가? 버전 1.2는 문장이 아니다.");

        assertThat(sentences).containsExactly("첫 문장이다.", "둘째 문장이다!", "셋째 문장인가?", "버전 1.2는 문장이 아니다.");
    }

    @Test
    @DisplayName("청크는 최대 길이를 넘지 않는 범위에서 문장을 이어 붙이고 문장 중간을 자르지 않는다")
    void chunksRespectMaxLengthWithoutCuttingSentences() {
        String text = "가나다라. 마바사아. 자차카타. 파하.";

        List<Chunk> chunks = Chunk.chunkDocument("doc", text, 12);

        assertThat(chunks).extracting(Chunk::text)
                .containsExactly("가나다라. 마바사아.", "자차카타. 파하.");
        for (Chunk chunk : chunks) {
            assertThat(chunk.text().codePointCount(0, chunk.text().length())).isLessThanOrEqualTo(12);
        }
    }

    @Test
    @DisplayName("청크는 문서 이름과 순번을 출처로 담는다")
    void chunksCarryProvenance() {
        List<Chunk> chunks = Chunk.chunkDocument("보안지침", "하나. 둘. 셋.", 3);

        assertThat(chunks).extracting(Chunk::docId).containsOnly("보안지침");
        assertThat(chunks).extracting(Chunk::seq).containsExactly(0, 1, 2);
    }

    @Test
    @DisplayName("최대 길이보다 긴 문장은 잘리지 않고 혼자서 청크 하나가 된다")
    void overlongSentenceBecomesItsOwnChunk() {
        List<Chunk> chunks = Chunk.chunkDocument("doc", "짧다. 이 문장은 최대 길이보다 훨씬 길다. 끝.", 8);

        assertThat(chunks).extracting(Chunk::text)
                .containsExactly("짧다.", "이 문장은 최대 길이보다 훨씬 길다.", "끝.");
    }
}
