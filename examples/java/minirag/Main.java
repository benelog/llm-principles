import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

// minirag: RAG 파이프라인의 검색 부분을 JDK 표준 라이브러리만으로 구현한 예제.
// 청킹 -> 색인(BM25, 벡터) -> 하이브리드 검색 -> 프롬프트 조립 순서로 실행된다.
public class Main {

    // 가상의 사내 문서. 실제 시스템이라면 파일이나 DB에서 읽어 온다.
    static final Map<String, String> DOCUMENTS = new LinkedHashMap<>();

    static {
        DOCUMENTS.put("휴가규정.txt", """
                연차 휴가는 입사일 기준으로 1년마다 15일이 부여된다. 3년 이상 근속하면 2년마다 1일씩 가산되어 최대 25일까지 늘어난다.
                휴가 신청은 인사 시스템에서 최소 3일 전에 등록해야 한다. 긴급한 사유가 있으면 부서장 승인으로 당일 신청도 가능하다.
                반차는 오전과 오후로 나누어 사용할 수 있으며 0.5일로 계산된다. 사용하지 않은 연차는 다음 해 3월까지 이월된다.""");

        DOCUMENTS.put("보안지침.txt", """
                사내 문서는 보안 등급에 따라 일반, 대외비, 극비로 분류한다. 대외비 이상의 문서는 외부 클라우드 서비스에 올릴 수 없다.
                업무용 노트북에는 반드시 디스크 암호화를 적용해야 한다. 퇴근 시에는 화면 잠금을 설정하고 자리를 비울 때도 잠금을 유지한다.
                외부에서 사내망에 접속할 때는 VPN을 사용해야 하며, 공용 와이파이에서는 VPN 없이 사내 시스템에 접속하면 안 된다.""");

        DOCUMENTS.put("장비신청.txt", """
                개발 장비는 입사 시 표준 사양의 노트북이 지급된다. 메모리 증설이나 모니터 추가는 장비 관리 시스템에서 신청한다.
                장비 신청은 팀장 승인 후 총무팀에서 처리하며 보통 5영업일이 걸린다. 고사양 장비가 필요한 경우 사유서를 첨부해야 한다.
                노트북 교체 주기는 4년이며, 고장이 잦은 경우 조기 교체를 신청할 수 있다.""");
    }

    // tag::pipeline[]
    public static void main(String[] args) {
        // 1단계: 청킹. 문서를 검색 단위인 청크로 나눈다.
        List<Chunk> chunks = new ArrayList<>();
        for (Map.Entry<String, String> doc : DOCUMENTS.entrySet()) {
            chunks.addAll(Chunk.chunkDocument(doc.getKey(), doc.getValue(), 120));
        }
        System.out.printf("문서 %d개를 청크 %d개로 나누었다.%n%n", DOCUMENTS.size(), chunks.size());

        // 2단계: 색인. 같은 청크를 두 방식으로 색인한다.
        Bm25Index bm25 = new Bm25Index(chunks);        // 키워드 검색용 역색인
        VectorIndex vectors = new VectorIndex(chunks); // 벡터 검색용 인덱스

        // 3단계: 질의. 두 검색 결과를 RRF로 합쳐 최종 순위를 만든다.
        List<String> queries = List.of(
                "연차는 며칠까지 이월할 수 있나요?",
                "카페 와이파이에서 사내 시스템에 접속해도 되나요?");
        for (String query : queries) {
            System.out.printf("== 질의: %s%n", query);
            List<SearchResult> bm25Results = bm25.search(query, 5);
            List<SearchResult> vectorResults = vectors.search(query, 5);
            List<SearchResult> hybrid = Rrf.fuse(List.of(bm25Results, vectorResults), 3);

            printResults("BM25(키워드)", bm25Results.subList(0, Math.min(2, bm25Results.size())));
            printResults("벡터(코사인)", vectorResults.subList(0, Math.min(2, vectorResults.size())));
            printResults("하이브리드(RRF)", hybrid);

            // 4단계: 프롬프트 조립. 검색된 청크가 LLM의 컨텍스트가 된다.
            System.out.println(buildPrompt(query, hybrid));
        }
    }
    // end::pipeline[]

    // tag::prompt[]
    // buildPrompt는 검색된 청크를 컨텍스트로 묶어 LLM에 보낼 프롬프트를 만든다.
    // "주어진 문맥에만 근거해서 답하라"는 지시와 출처 표기가 환각을 줄인다.
    static String buildPrompt(String query, List<SearchResult> results) {
        StringBuilder b = new StringBuilder();
        b.append("-- LLM에 전달할 프롬프트 --\n");
        b.append("아래 문맥에만 근거해서 질문에 답하세요. 답변에 출처 문서명을 표시하세요.\n");
        b.append("문맥에 없는 내용은 모른다고 답하세요.\n\n");
        for (int i = 0; i < results.size(); i++) {
            SearchResult r = results.get(i);
            b.append(String.format("[문맥 %d | 출처: %s]%n%s%n%n", i + 1, r.chunk().docId(), r.chunk().text()));
        }
        b.append(String.format("질문: %s%n", query));
        return b.toString();
    }
    // end::prompt[]

    static void printResults(String label, List<SearchResult> results) {
        System.out.printf("[%s]%n", label);
        for (int i = 0; i < results.size(); i++) {
            SearchResult r = results.get(i);
            String text = r.chunk().text();
            int end = text.offsetByCodePoints(0, Math.min(40, text.codePointCount(0, text.length())));
            String preview = text.substring(0, end);
            System.out.printf("  %d위 %.4f %s#%d: %s...%n",
                    i + 1, r.score(), r.chunk().docId(), r.chunk().seq(), preview);
        }
    }
}
