import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * BPE는 가장 자주 나오는 인접 토큰 쌍을 병합하는 학습(train)과,
 * 학습 때와 같은 순서로 병합을 재생하는 인코딩(encode)으로 이루어진다.
 * 빈도가 같으면 사전순으로 고르므로 결과가 결정적이고, 모든 검사를 정확히 단정할 수 있다.
 */
class BpeTest {

    @Test
    @DisplayName("toChars는 문자 단위로 나누고 공백을 ▁로 바꿔 단어 경계도 토큰으로 다룬다")
    void toCharsMarksWordBoundaries() {
        assertThat(Main.toChars("은행 좋다")).containsExactly("은", "행", "▁", "좋", "다");
    }

    @Test
    @DisplayName("mostFrequentPair는 가장 자주 나오는 인접 쌍을 찾고, 빈도가 같으면 사전순으로 고른다")
    void mostFrequentPairBreaksTiesLexicographically() {
        Map.Entry<Main.Pair, Integer> best = Main.mostFrequentPair(List.of("a", "b", "a", "b", "c", "d", "c", "d"));

        // ("a","b")와 ("c","d")가 두 번씩이므로 사전순으로 앞서는 ("a","b")를 고른다.
        assertThat(best.getKey()).isEqualTo(new Main.Pair("a", "b"));
        assertThat(best.getValue()).isEqualTo(2);
    }

    @Test
    @DisplayName("applyMerge는 쌍이 나타나는 자리마다 왼쪽부터 겹치지 않게 병합한다")
    void applyMergeMergesNonOverlapping() {
        List<String> merged = Main.applyMerge(List.of("a", "a", "a", "b"), new Main.Pair("a", "a"));

        assertThat(merged).containsExactly("aa", "a", "b");
    }

    @Test
    @DisplayName("train은 가장 흔한 쌍부터 순서대로 병합 규칙을 만든다")
    void trainLearnsMergesInFrequencyOrder() {
        String corpus = "은행이 좋다 은행은 크다 은행에서 만나자";

        List<Main.Pair> merges = Main.train(corpus, 3);

        // "은행"이 세 번으로 가장 흔하다. 병합 뒤에는 "은행" + 다음 글자 쌍들이 한 번씩이라
        // 두 번 이상 나오는 쌍 가운데 사전순으로 고른다.
        assertThat(merges.get(0)).isEqualTo(new Main.Pair("은", "행"));
        assertThat(merges).hasSize(3);
    }

    @Test
    @DisplayName("두 번 이상 나오는 쌍이 없으면 요청한 횟수 전에 병합을 멈춘다")
    void trainStopsWhenNoPairRepeats() {
        List<Main.Pair> merges = Main.train("abc", 10);

        assertThat(merges).isEmpty();
    }

    @Test
    @DisplayName("encode는 학습된 병합을 같은 순서로 적용하고, 못 본 문자는 낱글자로 남긴다")
    void encodeReplaysMergesAndKeepsUnknownCharacters() {
        List<Main.Pair> merges = Main.train("은행이 좋다 은행은 크다 은행에서 만나자 학교가 좋다 학교는 크다 학교에서 만나자", 12);

        List<String> tokens = Main.encode("은행나무", merges);

        assertThat(tokens.get(0)).isEqualTo("은행");
        assertThat(tokens).containsExactly("은행", "나", "무");
    }

    @Test
    @DisplayName("병합 규칙이 없으면 인코딩은 문자 단위 토큰화와 같다")
    void encodeWithoutMergesIsCharacterLevel() {
        assertThat(Main.encode("ab c", List.of())).containsExactly("a", "b", "▁", "c");
    }
}
