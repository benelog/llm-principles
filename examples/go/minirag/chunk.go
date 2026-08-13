package main

import "strings"

// tag::chunking[]
// Chunk는 검색과 인용의 기본 단위다.
// 어느 문서의 몇 번째 조각인지 추적할 수 있도록 출처를 함께 담는다.
type Chunk struct {
	DocID string // 원본 문서 이름
	Seq   int    // 문서 안에서의 순서
	Text  string
}

// splitSentences는 한국어 문장의 끝을 찾아 텍스트를 문장 단위로 나눈다.
// "다.", "요.", "함." 같은 종결 어미 뒤의 마침표와 줄바꿈을 경계로 삼는다.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder
	runes := []rune(text)
	for i, r := range runes {
		current.WriteRune(r)
		endOfSentence := false
		if r == '\n' {
			endOfSentence = true
		}
		if (r == '.' || r == '!' || r == '?') &&
			(i+1 == len(runes) || runes[i+1] == ' ' || runes[i+1] == '\n') {
			endOfSentence = true
		}
		if endOfSentence {
			if s := strings.TrimSpace(current.String()); s != "" {
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}
	if s := strings.TrimSpace(current.String()); s != "" {
		sentences = append(sentences, s)
	}
	return sentences
}

// chunkDocument는 문장을 존중하면서 최대 길이 이하의 청크로 묶는다.
// 문장 중간을 자르는 고정 길이 방식과 달리 의미 단위가 보존된다.
func chunkDocument(docID, text string, maxRunes int) []Chunk {
	var chunks []Chunk
	var current strings.Builder
	flush := func() {
		if s := strings.TrimSpace(current.String()); s != "" {
			chunks = append(chunks, Chunk{DocID: docID, Seq: len(chunks), Text: s})
		}
		current.Reset()
	}
	for _, sentence := range splitSentences(text) {
		// 이 문장을 더하면 최대 길이를 넘는 경우 지금까지 모은 청크를 마감한다.
		if current.Len() > 0 && len([]rune(current.String()))+len([]rune(sentence)) > maxRunes {
			flush()
		}
		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(sentence)
	}
	flush()
	return chunks
}
// end::chunking[]
