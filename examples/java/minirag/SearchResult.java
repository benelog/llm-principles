import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

// SearchResult는 검색 방식과 무관하게 공통으로 쓰는 결과 형식이다.
record SearchResult(Chunk chunk, double score) {

    // topK는 점수가 0보다 큰 청크만 골라 점수 내림차순으로 상위 topK개를 돌려준다.
    // 안정 정렬이므로 점수가 같으면 원래 순서가 유지된다.
    static List<SearchResult> topK(List<Chunk> chunks, double[] scores, int topK) {
        List<SearchResult> results = new ArrayList<>();
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > 0) {
                results.add(new SearchResult(chunks.get(i), scores[i]));
            }
        }
        results.sort(Comparator.comparingDouble(SearchResult::score).reversed());
        if (results.size() > topK) {
            results = new ArrayList<>(results.subList(0, topK));
        }
        return results;
    }
}
