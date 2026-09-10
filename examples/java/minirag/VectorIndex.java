import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// tag::vector[]
// VectorIndex는 청크를 벡터로 바꿔 두고 코사인 유사도로 검색한다.
// 이 예제의 벡터는 토큰 빈도(TF-IDF)로 만든 희소 벡터다.
// 실제 RAG 시스템은 학습된 임베딩 모델이 만든 밀집 벡터를 쓰지만,
// "벡터로 바꾼 뒤 기하학적 거리를 잰다"는 검색 절차 자체는 같다.
// Vector DB가 하는 일도 이 코사인 계산을 대규모로, 근사적으로 빠르게 하는 것이다.
final class VectorIndex {
    private final List<Chunk> chunks;
    private final List<Map<String, Double>> vectors = new ArrayList<>(); // 토큰 -> 가중치 희소 벡터
    private final Map<String, Integer> docFreq = new HashMap<>();
    private final int total;

    VectorIndex(List<Chunk> chunks) {
        this.chunks = chunks;
        this.total = chunks.size();
        List<Map<String, Integer>> tfs = new ArrayList<>();
        for (Chunk chunk : chunks) {
            Map<String, Integer> tf = new HashMap<>();
            for (String t : Tokenizer.tokenize(chunk.text())) {
                tf.merge(t, 1, Integer::sum);
            }
            for (String t : tf.keySet()) {
                docFreq.merge(t, 1, Integer::sum);
            }
            tfs.add(tf);
        }
        for (Map<String, Integer> tf : tfs) {
            vectors.add(embed(tf));
        }
    }

    // embed는 토큰 빈도를 TF-IDF 가중치 벡터로 바꾸고 길이를 1로 정규화한다.
    // 정규화해 두면 내적이 곧 코사인 유사도가 된다.
    Map<String, Double> embed(Map<String, Integer> tf) {
        Map<String, Double> vec = new HashMap<>();
        double norm = 0;
        for (Map.Entry<String, Integer> e : tf.entrySet()) {
            int df = docFreq.getOrDefault(e.getKey(), 0);
            if (df == 0) {
                df = 1;
            }
            double w = e.getValue() * Math.log(1 + (double) total / df);
            vec.put(e.getKey(), w);
            norm += w * w;
        }
        norm = Math.sqrt(norm);
        for (Map.Entry<String, Double> e : vec.entrySet()) {
            e.setValue(e.getValue() / norm);
        }
        return vec;
    }

    // search는 질의를 같은 방식으로 벡터화한 뒤
    // 모든 청크 벡터와의 코사인 유사도를 계산한다(brute-force 최근접 이웃 검색).
    List<SearchResult> search(String query, int topK) {
        Map<String, Integer> tf = new HashMap<>();
        for (String t : Tokenizer.tokenize(query)) {
            tf.merge(t, 1, Integer::sum);
        }
        Map<String, Double> queryVec = embed(tf);

        double[] scores = new double[chunks.size()];
        for (int i = 0; i < vectors.size(); i++) {
            Map<String, Double> vec = vectors.get(i);
            double dot = 0;
            for (Map.Entry<String, Double> e : queryVec.entrySet()) {
                dot += e.getValue() * vec.getOrDefault(e.getKey(), 0.0);
            }
            scores[i] = dot;
        }
        return SearchResult.topK(chunks, scores, topK);
    }
}
// end::vector[]
