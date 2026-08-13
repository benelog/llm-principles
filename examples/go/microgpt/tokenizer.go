package main

import "sort"

// tag::tokenizer[]
// Tokenizer는 문자 단위 토크나이저다.
// 데이터셋에 나오는 고유 문자마다 정수 ID를 하나씩 배정한다.
// names.txt는 소문자 알파벳 26자로 이루어져 있으므로
// 어휘 크기는 26 + BOS 특수 토큰 1개 = 27이 된다.
type Tokenizer struct {
	stoi map[rune]int // 문자 -> 토큰 ID
	itos map[int]rune // 토큰 ID -> 문자
	BOS  int          // 시퀀스의 시작과 끝을 나타내는 특수 토큰
}

// NewTokenizer는 데이터셋 전체에서 고유 문자를 수집해 어휘를 만든다.
func NewTokenizer(docs []string) *Tokenizer {
	seen := map[rune]bool{}
	for _, doc := range docs {
		for _, ch := range doc {
			seen[ch] = true
		}
	}
	// 실행할 때마다 같은 ID가 나오도록 문자를 정렬해 둔다
	chars := make([]rune, 0, len(seen))
	for ch := range seen {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	t := &Tokenizer{stoi: map[rune]int{}, itos: map[int]rune{}}
	for i, ch := range chars {
		t.stoi[ch] = i
		t.itos[i] = ch
	}
	t.BOS = len(chars) // 마지막 ID를 BOS 토큰으로 사용
	return t
}

// VocabSize는 고유 문자 수에 BOS 토큰 1개를 더한 값이다.
func (t *Tokenizer) VocabSize() int { return len(t.stoi) + 1 }

// Encode는 문자열을 토큰 ID 목록으로 바꾼다.
func (t *Tokenizer) Encode(s string) []int {
	ids := make([]int, 0, len(s))
	for _, ch := range s {
		ids = append(ids, t.stoi[ch])
	}
	return ids
}

// Decode는 토큰 ID 목록을 문자열로 되돌린다. BOS는 건너뛴다.
func (t *Tokenizer) Decode(ids []int) string {
	out := make([]rune, 0, len(ids))
	for _, id := range ids {
		if id == t.BOS {
			continue
		}
		out = append(out, t.itos[id])
	}
	return string(out)
}

// end::tokenizer[]
