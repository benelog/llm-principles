import java.util.ArrayList;
import java.util.List;

// tag::chunking[]
// Chunk는 검색과 인용의 기본 단위다.
// 어느 문서의 몇 번째 조각인지 추적할 수 있도록 출처를 함께 담는다.
record Chunk(String docId, // 원본 문서 이름
             int seq,      // 문서 안에서의 순서
             String text) {

    // splitSentences는 한국어 문장의 끝을 찾아 텍스트를 문장 단위로 나눈다.
    // "다.", "요.", "함." 같은 종결 어미 뒤의 마침표와 줄바꿈을 경계로 삼는다.
    static List<String> splitSentences(String text) {
        List<String> sentences = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        int[] cps = text.codePoints().toArray();
        for (int i = 0; i < cps.length; i++) {
            int cp = cps[i];
            current.appendCodePoint(cp);
            boolean endOfSentence = cp == '\n';
            if ((cp == '.' || cp == '!' || cp == '?')
                    && (i + 1 == cps.length || cps[i + 1] == ' ' || cps[i + 1] == '\n')) {
                endOfSentence = true;
            }
            if (endOfSentence) {
                String s = current.toString().strip();
                if (!s.isEmpty()) {
                    sentences.add(s);
                }
                current.setLength(0);
            }
        }
        String s = current.toString().strip();
        if (!s.isEmpty()) {
            sentences.add(s);
        }
        return sentences;
    }

    // chunkDocument는 문장을 존중하면서 최대 길이 이하의 청크로 묶는다.
    // 문장 중간을 자르는 고정 길이 방식과 달리 의미 단위가 보존된다.
    // 길이는 코드 포인트(문자) 수로 센다.
    static List<Chunk> chunkDocument(String docId, String text, int maxChars) {
        List<Chunk> chunks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (String sentence : splitSentences(text)) {
            // 이 문장을 더하면 최대 길이를 넘는 경우 지금까지 모은 청크를 마감한다.
            if (current.length() > 0
                    && current.codePointCount(0, current.length()) + sentence.codePointCount(0, sentence.length()) > maxChars) {
                flush(docId, current, chunks);
            }
            if (current.length() > 0) {
                current.append(' ');
            }
            current.append(sentence);
        }
        flush(docId, current, chunks);
        return chunks;
    }

    private static void flush(String docId, StringBuilder current, List<Chunk> chunks) {
        String s = current.toString().strip();
        if (!s.isEmpty()) {
            chunks.add(new Chunk(docId, chunks.size(), s));
        }
        current.setLength(0);
    }
}
// end::chunking[]
