package main

import "math"

// tag::vector[]
// VectorIndex는 청크를 벡터로 바꿔 두고 코사인 유사도로 검색한다.
// 이 예제의 벡터는 토큰 빈도(TF-IDF)로 만든 희소 벡터다.
// 실제 RAG 시스템은 학습된 임베딩 모델이 만든 밀집 벡터를 쓰지만,
// "벡터로 바꾼 뒤 기하학적 거리를 잰다"는 검색 절차 자체는 같다.
// Vector DB가 하는 일도 이 코사인 계산을 대규모로, 근사적으로 빠르게 하는 것이다.
type VectorIndex struct {
	chunks  []Chunk
	vectors []map[string]float64 // 토큰 -> 가중치 희소 벡터
	docFreq map[string]int
	total   int
}

func NewVectorIndex(chunks []Chunk) *VectorIndex {
	idx := &VectorIndex{chunks: chunks, docFreq: map[string]int{}, total: len(chunks)}
	tfs := make([]map[string]int, len(chunks))
	for i, chunk := range chunks {
		tf := map[string]int{}
		for _, t := range tokenize(chunk.Text) {
			tf[t]++
		}
		for t := range tf {
			idx.docFreq[t]++
		}
		tfs[i] = tf
	}
	for _, tf := range tfs {
		idx.vectors = append(idx.vectors, idx.embed(tf))
	}
	return idx
}

// embed는 토큰 빈도를 TF-IDF 가중치 벡터로 바꾸고 길이를 1로 정규화한다.
// 정규화해 두면 내적이 곧 코사인 유사도가 된다.
func (idx *VectorIndex) embed(tf map[string]int) map[string]float64 {
	vec := map[string]float64{}
	var norm float64
	for token, count := range tf {
		df := idx.docFreq[token]
		if df == 0 {
			df = 1
		}
		w := float64(count) * math.Log(1+float64(idx.total)/float64(df))
		vec[token] = w
		norm += w * w
	}
	norm = math.Sqrt(norm)
	for token := range vec {
		vec[token] /= norm
	}
	return vec
}

// Search는 질의를 같은 방식으로 벡터화한 뒤
// 모든 청크 벡터와의 코사인 유사도를 계산한다(brute-force 최근접 이웃 검색).
func (idx *VectorIndex) Search(query string, topK int) []SearchResult {
	tf := map[string]int{}
	for _, t := range tokenize(query) {
		tf[t]++
	}
	queryVec := idx.embed(tf)

	scores := make([]float64, len(idx.chunks))
	for i, vec := range idx.vectors {
		var dot float64
		for token, w := range queryVec {
			dot += w * vec[token]
		}
		scores[i] = dot
	}
	return topKResults(idx.chunks, scores, topK)
}
// end::vector[]

// tag::rrf[]
// fuseRRF는 두 검색 결과를 Reciprocal Rank Fusion으로 합친다.
// 점수의 단위가 서로 다른 검색기(BM25 점수와 코사인 유사도)를 합칠 때는
// 점수 대신 "순위"를 쓰는 편이 안정적이다.
// 각 목록에서 순위 r인 문서에 1/(k+r) 점을 주고 합산한다.
func fuseRRF(lists [][]SearchResult, topK int) []SearchResult {
	const k = 60.0
	scores := map[string]float64{}
	chunkByKey := map[string]Chunk{}
	for _, list := range lists {
		for rank, result := range list {
			key := result.Chunk.DocID + "#" + string(rune('0'+result.Chunk.Seq))
			scores[key] += 1.0 / (k + float64(rank+1))
			chunkByKey[key] = result.Chunk
		}
	}
	var chunks []Chunk
	var flat []float64
	for key, score := range scores {
		chunks = append(chunks, chunkByKey[key])
		flat = append(flat, score)
	}
	return topKResults(chunks, flat, topK)
}
// end::rrf[]
