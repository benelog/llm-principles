import java.util.ArrayList;
import java.util.List;

// tag::tokenize[]
// Tokenizer는 텍스트를 문자 2-gram(바이그램)으로 나눈다.
// 한국어는 "은행이", "은행에서"처럼 조사가 붙어 단어 형태가 계속 변하므로
// 공백 단위로 자르면 같은 단어를 다른 토큰으로 취급하게 된다.
// 형태소 분석기가 없는 환경에서는 문자 바이그램이 실용적인 차선책이다.
// "은행이" -> ["은행", "행이"] 가 되어 "은행"이라는 공통 토큰이 남는다.
final class Tokenizer {
    private Tokenizer() {
    }

    static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        for (String word : text.toLowerCase().trim().split("\\s+")) {
            // 기호를 제거하고 문자만 남긴다.
            int[] filtered = word.codePoints().filter(Character::isLetterOrDigit).toArray();
            if (filtered.length == 0) {
                continue;
            }
            if (filtered.length == 1) {
                tokens.add(new String(filtered, 0, 1));
                continue;
            }
            for (int i = 0; i + 1 < filtered.length; i++) {
                tokens.add(new String(filtered, i, 2));
            }
        }
        return tokens;
    }
}
// end::tokenize[]
