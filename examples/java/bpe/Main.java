// BPE(Byte Pair Encoding) 토크나이저의 최소 구현.
// 가장 자주 등장하는 인접 토큰 쌍을 반복해서 병합하는 학습과,
// 학습된 병합 규칙을 같은 순서로 재생하는 인코딩으로 이루어진다.

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {

    // tag::train[]
    // Pair는 병합 대상인 인접 토큰 쌍이다.
    record Pair(String left, String right) {}

    // train은 코퍼스에서 가장 자주 등장하는 인접 쌍을 numMerges번 병합하고,
    // 병합 규칙을 적용한 순서대로 반환한다. 이 목록이 곧 토크나이저의 사전이다.
    static List<Pair> train(String corpus, int numMerges) {
        List<String> tokens = toChars(corpus);
        List<Pair> merges = new ArrayList<>();
        while (merges.size() < numMerges) {
            Map.Entry<Pair, Integer> best = mostFrequentPair(tokens);
            if (best == null || best.getValue() < 2) { // 두 번 이상 나온 쌍이 없으면 더 병합할 것이 없다
                break;
            }
            merges.add(best.getKey());
            tokens = applyMerge(tokens, best.getKey());
        }
        return merges;
    }

    // mostFrequentPair는 토큰 열에서 가장 자주 등장하는 인접 쌍과 그 횟수를 찾는다.
    // 빈도가 같으면 사전순으로 앞서는 쌍을 골라 결과를 결정적으로 만든다.
    static Map.Entry<Pair, Integer> mostFrequentPair(List<String> tokens) {
        Map<Pair, Integer> counts = new HashMap<>();
        for (int i = 0; i + 1 < tokens.size(); i++) {
            counts.merge(new Pair(tokens.get(i), tokens.get(i + 1)), 1, Integer::sum);
        }
        Map.Entry<Pair, Integer> best = null;
        for (Map.Entry<Pair, Integer> e : counts.entrySet()) {
            if (best == null || e.getValue() > best.getValue()
                    || (e.getValue().equals(best.getValue()) && PAIR_ORDER.compare(e.getKey(), best.getKey()) < 0)) {
                best = e;
            }
        }
        return best;
    }
    // end::train[]

    // tag::encode[]
    // encode는 텍스트를 문자 단위로 쪼갠 뒤, 학습 때와 같은 순서로
    // 병합 규칙을 적용해서 토큰 열로 바꾼다.
    static List<String> encode(String text, List<Pair> merges) {
        List<String> tokens = toChars(text);
        for (Pair m : merges) {
            tokens = applyMerge(tokens, m);
        }
        return tokens;
    }

    // applyMerge는 토큰 열에서 쌍 p가 나타나는 자리마다 병합한 새 토큰 열을 만든다.
    static List<String> applyMerge(List<String> tokens, Pair p) {
        List<String> out = new ArrayList<>(tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            if (i + 1 < tokens.size() && tokens.get(i).equals(p.left()) && tokens.get(i + 1).equals(p.right())) {
                out.add(p.left() + p.right());
                i++;
            } else {
                out.add(tokens.get(i));
            }
        }
        return out;
    }
    // end::encode[]

    // toChars는 텍스트를 문자(코드 포인트) 단위 토큰으로 나눈다. 공백은 ▁로 바꿔서
    // 단어 경계도 병합에 참여하는 문자로 다룬다. SentencePiece와 같은 방식이다.
    static List<String> toChars(String text) {
        List<String> out = new ArrayList<>();
        text.replace(" ", "▁").codePoints().forEach(cp -> out.add(Character.toString(cp)));
        return out;
    }

    // 쌍의 사전순. 왼쪽 토큰을 먼저 비교하고 같으면 오른쪽 토큰을 비교한다.
    static final Comparator<Pair> PAIR_ORDER =
            Comparator.comparing(Pair::left).thenComparing(Pair::right);

    static String quote(List<String> tokens) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < tokens.size(); i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append('"').append(tokens.get(i)).append('"');
        }
        return sb.append(']').toString();
    }

    public static void main(String[] args) {
        String corpus = "은행이 좋다 은행은 크다 은행에서 만나자 학교가 좋다 학교는 크다 학교에서 만나자";
        List<Pair> merges = train(corpus, 12);

        System.out.println("== 학습된 병합 규칙 (적용 순서대로) ==");
        for (int i = 0; i < merges.size(); i++) {
            Pair m = merges.get(i);
            System.out.printf("%2d: \"%s\" + \"%s\" -> \"%s\"%n", i + 1, m.left(), m.right(), m.left() + m.right());
        }

        System.out.println();
        System.out.println("== 인코딩 ==");
        for (String text : new String[] {"은행에서 보자", "은행나무"}) {
            System.out.printf("\"%s\" -> %s%n", text, quote(encode(text, merges)));
        }
    }
}
