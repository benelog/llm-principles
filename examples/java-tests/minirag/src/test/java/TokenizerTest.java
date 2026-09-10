import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * minirag의 Tokenizer는 문자 바이그램 토크나이저다. 조사가 붙어 형태가 바뀌는
 * 한국어 단어에서 공통 토큰이 남도록 두 글자씩 겹쳐 자른다.
 */
class TokenizerTest {

    @Test
    @DisplayName("조사가 달라도 어간의 바이그램은 같게 남는다")
    void bigramsSurviveParticles() {
        assertThat(Tokenizer.tokenize("은행이")).containsExactly("은행", "행이");
        assertThat(Tokenizer.tokenize("은행에서")).containsExactly("은행", "행에", "에서");
    }

    @Test
    @DisplayName("공백으로 단어를 나누고 기호는 제거하며 영문은 소문자로 바꾼다")
    void splitsOnWhitespaceAndDropsSymbols() {
        assertThat(Tokenizer.tokenize("VPN을 켜세요! (필수)")).containsExactly("vp", "pn", "n을", "켜세", "세요", "필수");
    }

    @Test
    @DisplayName("한 글자 단어는 그대로 토큰이 되고 빈 입력은 빈 목록이다")
    void singleCharactersAndEmptyInput() {
        assertThat(Tokenizer.tokenize("나 는")).containsExactly("나", "는");
        assertThat(Tokenizer.tokenize("   ")).isEmpty();
    }
}
