package main

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// tag::tokenize[]
// tokenize는 텍스트를 문자 2-gram(바이그램)으로 나눈다.
// 한국어는 "은행이", "은행에서"처럼 조사가 붙어 단어 형태가 계속 변하므로
// 공백 단위로 자르면 같은 단어를 다른 토큰으로 취급하게 된다.
// 형태소 분석기가 없는 환경에서는 문자 바이그램이 실용적인 차선책이다.
// "은행이" -> ["은행", "행이"] 가 되어 "은행"이라는 공통 토큰이 남는다.
func tokenize(text string) []string {
	var tokens []string
	for _, word := range strings.Fields(strings.ToLower(text)) {
		runes := []rune(word)
		// 기호를 제거하고 문자만 남긴다.
		filtered := runes[:0]
		for _, r := range runes {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				filtered = append(filtered, r)
			}
		}
		if len(filtered) == 1 {
			tokens = append(tokens, string(filtered))
			continue
		}
		for i := 0; i+1 < len(filtered); i++ {
			tokens = append(tokens, string(filtered[i:i+2]))
		}
	}
	return tokens
}
// end::tokenize[]

// tag::bm25[]
// BM25Index는 키워드(lexical) 검색을 위한 역색인이다.
// "어떤 토큰이 어떤 청크에 몇 번 나오는가"를 미리 정리해 둔다.
type BM25Index struct {
	chunks    []Chunk
	docFreq   map[string]int   // 토큰이 등장한 청크 수
	termFreq  []map[string]int // 청크별 토큰 등장 횟수
	docLen    []int            // 청크별 토큰 수
	avgDocLen float64
}

func NewBM25Index(chunks []Chunk) *BM25Index {
	idx := &BM25Index{
		chunks:  chunks,
		docFreq: map[string]int{},
	}
	totalLen := 0
	for _, chunk := range chunks {
		tf := map[string]int{}
		tokens := tokenize(chunk.Text)
		for _, t := range tokens {
			tf[t]++
		}
		for t := range tf {
			idx.docFreq[t]++
		}
		idx.termFreq = append(idx.termFreq, tf)
		idx.docLen = append(idx.docLen, len(tokens))
		totalLen += len(tokens)
	}
	idx.avgDocLen = float64(totalLen) / float64(len(chunks))
	return idx
}

// Search는 BM25 점수로 청크의 순위를 매긴다.
// 희귀한 토큰일수록(IDF), 그 청크에 자주 나올수록(TF) 점수가 높고,
// 청크가 평균보다 길면 점수를 깎아 길이에 따른 유리함을 보정한다.
func (idx *BM25Index) Search(query string, topK int) []SearchResult {
	const k1, b = 1.2, 0.75
	n := float64(len(idx.chunks))
	scores := make([]float64, len(idx.chunks))
	for _, token := range tokenize(query) {
		df := idx.docFreq[token]
		if df == 0 {
			continue
		}
		idf := math.Log(1 + (n-float64(df)+0.5)/(float64(df)+0.5))
		for i := range idx.chunks {
			tf := float64(idx.termFreq[i][token])
			if tf == 0 {
				continue
			}
			lenNorm := 1 - b + b*float64(idx.docLen[i])/idx.avgDocLen
			scores[i] += idf * tf * (k1 + 1) / (tf + k1*lenNorm)
		}
	}
	return topKResults(idx.chunks, scores, topK)
}
// end::bm25[]

// SearchResult는 검색 방식과 무관하게 공통으로 쓰는 결과 형식이다.
type SearchResult struct {
	Chunk Chunk
	Score float64
}

func topKResults(chunks []Chunk, scores []float64, topK int) []SearchResult {
	var results []SearchResult
	for i, s := range scores {
		if s > 0 {
			results = append(results, SearchResult{Chunk: chunks[i], Score: s})
		}
	}
	sort.SliceStable(results, func(a, b int) bool { return results[a].Score > results[b].Score })
	if len(results) > topK {
		results = results[:topK]
	}
	return results
}
