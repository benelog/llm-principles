import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

// tag::tokenizer[]
// Tokenizer는 문자 단위 토크나이저다.
// 데이터셋에 나오는 고유 문자마다 정수 ID를 하나씩 배정한다.
// names.txt는 소문자 알파벳 26자로 이루어져 있으므로
// 어휘 크기는 26 + BOS 특수 토큰 1개 = 27이 된다.
public class Tokenizer {
    private final Map<Character, Integer> stoi = new HashMap<>(); // 문자 -> 토큰 ID
    private final char[] itos;                                    // 토큰 ID -> 문자
    final int bos; // 시퀀스의 시작과 끝을 나타내는 특수 토큰

    // 데이터셋 전체에서 고유 문자를 수집해 어휘를 만든다.
    public Tokenizer(List<String> docs) {
        // 실행할 때마다 같은 ID가 나오도록 정렬된 집합에 문자를 모은다
        TreeSet<Character> seen = new TreeSet<>();
        for (String doc : docs) {
            for (char ch : doc.toCharArray()) {
                seen.add(ch);
            }
        }
        itos = new char[seen.size()];
        int id = 0;
        for (char ch : seen) {
            itos[id] = ch;
            stoi.put(ch, id);
            id++;
        }
        bos = itos.length; // 마지막 ID를 BOS 토큰으로 사용
    }

    // 고유 문자 수에 BOS 토큰 1개를 더한 값이다.
    public int vocabSize() { return itos.length + 1; }

    // 문자열을 토큰 ID 목록으로 바꾼다.
    public int[] encode(String s) {
        int[] ids = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            ids[i] = stoi.get(s.charAt(i));
        }
        return ids;
    }

    // 토큰 ID 목록을 문자열로 되돌린다. BOS는 건너뛴다.
    public String decode(List<Integer> ids) {
        StringBuilder out = new StringBuilder();
        for (int id : ids) {
            if (id == bos) {
                continue;
            }
            out.append(itos[id]);
        }
        return out.toString();
    }
}
// end::tokenizer[]
