import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tokenizer는 문자 단위 토크나이저다. 데이터셋의 고유 문자마다 정수 ID를 배정하고,
 * 마지막 ID 하나를 시퀀스의 시작과 끝을 뜻하는 BOS 토큰으로 쓴다.
 */
class TokenizerTest {

    private final Tokenizer tok = new Tokenizer(List.of("emma", "olivia"));

    @Test
    @DisplayName("어휘 크기는 고유 문자 수에 BOS 하나를 더한 값이다")
    void vocabularyIsUniqueCharactersPlusBos() {
        // e, m, a, o, l, i, v: 7자
        assertThat(tok.vocabSize()).isEqualTo(8);
        assertThat(tok.bos).as("BOS는 마지막 ID").isEqualTo(7);
    }

    @Test
    @DisplayName("문자 ID는 정렬 순서로 배정되어 실행할 때마다 같다")
    void idsFollowSortedOrder() {
        // 정렬하면 a, e, i, l, m, o, v
        assertThat(tok.encode("a")).containsExactly(0);
        assertThat(tok.encode("v")).containsExactly(6);
        assertThat(tok.encode("emma")).containsExactly(1, 4, 4, 0);
    }

    @Test
    @DisplayName("encode와 decode는 서로 역연산이다")
    void encodeAndDecodeRoundTrip() {
        int[] ids = tok.encode("olivia");

        assertThat(tok.decode(toList(ids))).isEqualTo("olivia");
    }

    @Test
    @DisplayName("decode는 BOS 토큰을 건너뛴다")
    void decodeSkipsBos() {
        List<Integer> withBos = List.of(tok.bos, 1, 4, 4, 0, tok.bos);

        assertThat(tok.decode(withBos)).isEqualTo("emma");
    }

    private static List<Integer> toList(int[] ids) {
        return java.util.Arrays.stream(ids).boxed().toList();
    }
}
