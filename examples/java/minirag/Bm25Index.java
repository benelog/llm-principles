import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// tag::bm25[]
// Bm25Index는 키워드(lexical) 검색을 위한 역색인이다.
// "어떤 토큰이 어떤 청크에 몇 번 나오는가"를 미리 정리해 둔다.
final class Bm25Index {
    private final List<Chunk> chunks;
    private final Map<String, Integer> docFreq = new HashMap<>();      // 토큰이 등장한 청크 수
    private final List<Map<String, Integer>> termFreq = new ArrayList<>(); // 청크별 토큰 등장 횟수
    private final List<Integer> docLen = new ArrayList<>();              // 청크별 토큰 수
    private final double avgDocLen;

    Bm25Index(List<Chunk> chunks) {
        this.chunks = chunks;
        int totalLen = 0;
        for (Chunk chunk : chunks) {
            Map<String, Integer> tf = new HashMap<>();
            List<String> tokens = Tokenizer.tokenize(chunk.text());
            for (String t : tokens) {
                tf.merge(t, 1, Integer::sum);
            }
            for (String t : tf.keySet()) {
                docFreq.merge(t, 1, Integer::sum);
            }
            termFreq.add(tf);
            docLen.add(tokens.size());
            totalLen += tokens.size();
        }
        this.avgDocLen = (double) totalLen / chunks.size();
    }

    // search는 BM25 점수로 청크의 순위를 매긴다.
    // 희귀한 토큰일수록(IDF), 그 청크에 자주 나올수록(TF) 점수가 높고,
    // 청크가 평균보다 길면 점수를 깎아 길이에 따른 유리함을 보정한다.
    List<SearchResult> search(String query, int topK) {
        final double k1 = 1.2, b = 0.75;
        double n = chunks.size();
        double[] scores = new double[chunks.size()];
        for (String token : Tokenizer.tokenize(query)) {
            int df = docFreq.getOrDefault(token, 0);
            if (df == 0) {
                continue;
            }
            double idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
            for (int i = 0; i < chunks.size(); i++) {
                double tf = termFreq.get(i).getOrDefault(token, 0);
                if (tf == 0) {
                    continue;
                }
                double lenNorm = 1 - b + b * docLen.get(i) / avgDocLen;
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * lenNorm);
            }
        }
        return SearchResult.topK(chunks, scores, topK);
    }
}
// end::bm25[]
