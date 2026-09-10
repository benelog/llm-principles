import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

// tag::rrf[]
// Rrf.fuse는 두 검색 결과를 Reciprocal Rank Fusion으로 합친다.
// 점수의 단위가 서로 다른 검색기(BM25 점수와 코사인 유사도)를 합칠 때는
// 점수 대신 "순위"를 쓰는 편이 안정적이다.
// 각 목록에서 순위 r인 문서에 1/(k+r) 점을 주고 합산한다.
final class Rrf {
    private Rrf() {
    }

    static List<SearchResult> fuse(List<List<SearchResult>> lists, int topK) {
        final double k = 60.0;
        Map<String, Double> scores = new LinkedHashMap<>();
        Map<String, Chunk> chunkByKey = new LinkedHashMap<>();
        for (List<SearchResult> list : lists) {
            for (int rank = 0; rank < list.size(); rank++) {
                Chunk chunk = list.get(rank).chunk();
                String key = chunk.docId() + "#" + chunk.seq();
                scores.merge(key, 1.0 / (k + rank + 1), Double::sum);
                chunkByKey.put(key, chunk);
            }
        }
        List<Chunk> chunks = new ArrayList<>();
        double[] flat = new double[scores.size()];
        int i = 0;
        for (Map.Entry<String, Double> e : scores.entrySet()) {
            chunks.add(chunkByKey.get(e.getKey()));
            flat[i++] = e.getValue();
        }
        return SearchResult.topK(chunks, flat, topK);
    }
}
// end::rrf[]
