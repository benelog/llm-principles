// BPE(Byte Pair Encoding) 토크나이저의 최소 구현.
// 가장 자주 등장하는 인접 토큰 쌍을 반복해서 병합하는 학습과,
// 학습된 병합 규칙을 같은 순서로 재생하는 인코딩으로 이루어진다.
package main

import (
	"fmt"
	"strings"
)

// tag::train[]
// Pair는 병합 대상인 인접 토큰 쌍이다.
type Pair struct{ Left, Right string }

// Train은 코퍼스에서 가장 자주 등장하는 인접 쌍을 numMerges번 병합하고,
// 병합 규칙을 적용한 순서대로 반환한다. 이 목록이 곧 토크나이저의 사전이다.
func Train(corpus string, numMerges int) []Pair {
	tokens := toChars(corpus)
	var merges []Pair
	for len(merges) < numMerges {
		best, count := mostFrequentPair(tokens)
		if count < 2 { // 두 번 이상 나온 쌍이 없으면 더 병합할 것이 없다
			break
		}
		merges = append(merges, best)
		tokens = applyMerge(tokens, best)
	}
	return merges
}

// mostFrequentPair는 토큰 열에서 가장 자주 등장하는 인접 쌍을 찾는다.
// 빈도가 같으면 사전순으로 앞서는 쌍을 골라 결과를 결정적으로 만든다.
func mostFrequentPair(tokens []string) (Pair, int) {
	counts := map[Pair]int{}
	for i := 0; i+1 < len(tokens); i++ {
		counts[Pair{tokens[i], tokens[i+1]}]++
	}
	var best Pair
	bestCount := 0
	for p, c := range counts {
		if c > bestCount || (c == bestCount && pairLess(p, best)) {
			best, bestCount = p, c
		}
	}
	return best, bestCount
}

// end::train[]

// tag::encode[]
// Encode는 텍스트를 문자 단위로 쪼갠 뒤, 학습 때와 같은 순서로
// 병합 규칙을 적용해서 토큰 열로 바꾼다.
func Encode(text string, merges []Pair) []string {
	tokens := toChars(text)
	for _, m := range merges {
		tokens = applyMerge(tokens, m)
	}
	return tokens
}

// applyMerge는 토큰 열에서 쌍 p가 나타나는 자리마다 병합한 새 토큰 열을 만든다.
func applyMerge(tokens []string, p Pair) []string {
	out := make([]string, 0, len(tokens))
	for i := 0; i < len(tokens); i++ {
		if i+1 < len(tokens) && tokens[i] == p.Left && tokens[i+1] == p.Right {
			out = append(out, p.Left+p.Right)
			i++
		} else {
			out = append(out, tokens[i])
		}
	}
	return out
}

// end::encode[]

// toChars는 텍스트를 문자(룬) 단위 토큰으로 나눈다. 공백은 ▁로 바꿔서
// 단어 경계도 병합에 참여하는 문자로 다룬다. SentencePiece와 같은 방식이다.
func toChars(text string) []string {
	var out []string
	for _, r := range strings.ReplaceAll(text, " ", "▁") {
		out = append(out, string(r))
	}
	return out
}

func pairLess(a, b Pair) bool {
	if a.Left != b.Left {
		return a.Left < b.Left
	}
	return a.Right < b.Right
}

func main() {
	corpus := "은행이 좋다 은행은 크다 은행에서 만나자 학교가 좋다 학교는 크다 학교에서 만나자"
	merges := Train(corpus, 12)

	fmt.Println("== 학습된 병합 규칙 (적용 순서대로) ==")
	for i, m := range merges {
		fmt.Printf("%2d: %q + %q -> %q\n", i+1, m.Left, m.Right, m.Left+m.Right)
	}

	fmt.Println()
	fmt.Println("== 인코딩 ==")
	for _, text := range []string{"은행에서 보자", "은행나무"} {
		fmt.Printf("%q -> %q\n", text, Encode(text, merges))
	}
}
