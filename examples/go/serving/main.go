// KV 캐시와 프리필·디코드가 만드는 추론 비용 구조를 측정하는 예제.
// 학습 없이 무작위 가중치를 쓴다. 계산량과 메모리 크기는 가중치 값과
// 무관하므로 시간과 용량을 재는 데는 지장이 없다.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// 시간을 잴 수 있도록 microGPT보다 키운 장난감 트랜스포머 크기.
const (
	vocabSize = 256 // 어휘 크기
	nEmbd     = 64  // 임베딩 차원
	nHead     = 8   // 어텐션 헤드 수
	headDim   = nEmbd / nHead
	nLayer    = 4   // 트랜스포머 블록 수
	maxCtx    = 512 // 최대 문맥 길이
)

// Layer는 트랜스포머 블록 하나의 가중치 묶음이다.
type Layer struct {
	wq, wk, wv, wo [][]float64 // 어텐션 Q, K, V, 출력 투영
	wfc, wproj     [][]float64 // MLP 확장과 축소
}

// Model은 임베딩, 블록들, 최종 로짓 투영으로 이루어진다.
type Model struct {
	wte    [][]float64 // 토큰 임베딩 (vocabSize x nEmbd)
	wpe    [][]float64 // 위치 임베딩 (maxCtx x nEmbd)
	layers []*Layer
	wout   [][]float64 // 최종 로짓 투영 (vocabSize x nEmbd)
}

func newMatrix(rows, cols int, rng *rand.Rand) [][]float64 {
	w := make([][]float64, rows)
	for i := range w {
		w[i] = make([]float64, cols)
		for j := range w[i] {
			w[i][j] = rng.NormFloat64() * 0.02
		}
	}
	return w
}

func NewModel(rng *rand.Rand) *Model {
	m := &Model{
		wte:  newMatrix(vocabSize, nEmbd, rng),
		wpe:  newMatrix(maxCtx, nEmbd, rng),
		wout: newMatrix(vocabSize, nEmbd, rng),
	}
	for l := 0; l < nLayer; l++ {
		m.layers = append(m.layers, &Layer{
			wq: newMatrix(nEmbd, nEmbd, rng), wk: newMatrix(nEmbd, nEmbd, rng),
			wv: newMatrix(nEmbd, nEmbd, rng), wo: newMatrix(nEmbd, nEmbd, rng),
			wfc: newMatrix(4*nEmbd, nEmbd, rng), wproj: newMatrix(nEmbd, 4*nEmbd, rng),
		})
	}
	return m
}

// matvec은 행렬 w와 벡터 x의 곱이다.
func matvec(w [][]float64, x []float64) []float64 {
	out := make([]float64, len(w))
	for i, row := range w {
		sum := 0.0
		for j, v := range x {
			sum += row[j] * v
		}
		out[i] = sum
	}
	return out
}

// rmsnorm은 벡터를 제곱 평균의 제곱근으로 나눠 크기를 일정하게 유지한다.
func rmsnorm(x []float64) []float64 {
	ss := 0.0
	for _, v := range x {
		ss += v * v
	}
	scale := 1 / math.Sqrt(ss/float64(len(x))+1e-5)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * scale
	}
	return out
}

func add(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func tanhVec(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Tanh(v)
	}
	return out
}

// embed는 토큰 임베딩과 위치 임베딩을 더한 입력 벡터다.
func (m *Model) embed(tokenID, pos int) []float64 {
	return add(m.wte[tokenID], m.wpe[pos])
}

// attend는 질의 벡터 하나가 캐시된 키·밸류 전체를 참조하는 멀티헤드 어텐션이다.
func attend(q []float64, keys, values [][]float64) []float64 {
	out := make([]float64, nEmbd)
	invSqrt := 1 / math.Sqrt(float64(headDim))
	for h := 0; h < nHead; h++ {
		lo := h * headDim
		// 질의와 각 키의 내적으로 점수를 매긴다
		scores := make([]float64, len(keys))
		maxScore := math.Inf(-1)
		for t, k := range keys {
			dot := 0.0
			for d := 0; d < headDim; d++ {
				dot += q[lo+d] * k[lo+d]
			}
			scores[t] = dot * invSqrt
			if scores[t] > maxScore {
				maxScore = scores[t]
			}
		}
		// softmax 가중치로 밸류들을 섞는다
		sum := 0.0
		for t := range scores {
			scores[t] = math.Exp(scores[t] - maxScore)
			sum += scores[t]
		}
		for t, v := range values {
			w := scores[t] / sum
			for d := 0; d < headDim; d++ {
				out[lo+d] += w * v[lo+d]
			}
		}
	}
	return out
}

// tag::kvcache[]
// KVCache는 처리를 마친 위치들의 키와 밸류를 층별로 보관한다.
type KVCache struct {
	keys   [][][]float64 // [층][위치][차원]
	values [][][]float64
}

func (m *Model) NewCache() *KVCache {
	return &KVCache{
		keys:   make([][][]float64, nLayer),
		values: make([][][]float64, nLayer),
	}
}

// DecodeStep은 토큰 하나를 처리해 캐시에 K, V를 쌓고 마지막 은닉 벡터를
// 돌려준다. 2부 microGPT의 Forward와 같은 구조이며, 학습이 필요 없으므로
// 자동 미분 없이 float64로만 계산한다.
func (m *Model) DecodeStep(tokenID, pos int, cache *KVCache) []float64 {
	x := m.embed(tokenID, pos)
	for l, layer := range m.layers {
		// 어텐션: 현재 토큰의 K, V는 캐시에 추가하고, Q는 캐시 전체를 참조한다
		xn := rmsnorm(x)
		q := matvec(layer.wq, xn)
		cache.keys[l] = append(cache.keys[l], matvec(layer.wk, xn))
		cache.values[l] = append(cache.values[l], matvec(layer.wv, xn))
		x = add(x, matvec(layer.wo, attend(q, cache.keys[l], cache.values[l])))
		// MLP
		xn = rmsnorm(x)
		x = add(x, matvec(layer.wproj, tanhVec(matvec(layer.wfc, xn))))
	}
	return rmsnorm(x)
}

// end::kvcache[]

// parallelFor는 f(0)부터 f(n-1)까지를 CPU 코어들에 나눠 실행한다.
func parallelFor(n int, f func(i int)) {
	workers := runtime.NumCPU()
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w; i < n; i += workers {
				f(i)
			}
		}(w)
	}
	wg.Wait()
}

// tag::prefill[]
// Prefill은 프롬프트의 모든 위치를 층 단위로 한꺼번에 처리한다.
// 프롬프트 토큰은 전부 미리 알고 있으므로 위치별 계산이 서로 독립이고,
// 병렬로 실행할 수 있다. 디코드는 직전 토큰이 나와야 다음 토큰을 시작할
// 수 있으므로 이런 병렬화가 불가능하다.
func (m *Model) Prefill(tokens []int) (*KVCache, []float64) {
	n := len(tokens)
	cache := m.NewCache()
	xs := make([][]float64, n)
	parallelFor(n, func(i int) { xs[i] = m.embed(tokens[i], i) })
	for l, layer := range m.layers {
		qs := make([][]float64, n)
		ks := make([][]float64, n)
		vs := make([][]float64, n)
		// 1) 모든 위치의 Q, K, V를 병렬로 계산해 캐시를 채운다
		parallelFor(n, func(i int) {
			xn := rmsnorm(xs[i])
			qs[i] = matvec(layer.wq, xn)
			ks[i] = matvec(layer.wk, xn)
			vs[i] = matvec(layer.wv, xn)
		})
		cache.keys[l], cache.values[l] = ks, vs
		// 2) 어텐션과 MLP도 병렬. 위치 i는 0..i의 키·밸류만 참조한다(causal)
		parallelFor(n, func(i int) {
			x := add(xs[i], matvec(layer.wo, attend(qs[i], ks[:i+1], vs[:i+1])))
			xn := rmsnorm(x)
			xs[i] = add(x, matvec(layer.wproj, tanhVec(matvec(layer.wfc, xn))))
		})
	}
	return cache, rmsnorm(xs[n-1])
}

// end::prefill[]

// nextToken은 로짓이 가장 큰 토큰을 고른다(greedy 디코딩). 캐시 유무를
// 비교할 때 두 경로가 같은 토큰 열을 내는지 확인하기 위해 무작위성을 뺐다.
func (m *Model) nextToken(h []float64) int {
	logits := matvec(m.wout, h)
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return best
}

// tag::nocache[]
// GenerateNoCache는 캐시 없이 토큰을 생성한다. 새 토큰을 하나 만들 때마다
// 빈 캐시에서 출발해 시퀀스 전체를 처음부터 다시 계산한다.
func (m *Model) GenerateNoCache(prompt []int, numTokens int) []int {
	seq := append([]int{}, prompt...)
	for len(seq) < len(prompt)+numTokens {
		cache := m.NewCache()
		var h []float64
		for pos, t := range seq { // 앞 토큰 전체를 다시 계산한다
			h = m.DecodeStep(t, pos, cache)
		}
		seq = append(seq, m.nextToken(h))
	}
	return seq[len(prompt):]
}

// Generate는 프리필이 채운 캐시를 이어받아, 새 토큰마다 DecodeStep 한 번만
// 실행한다. 앞 토큰들의 K, V는 캐시에서 그대로 재사용된다.
func (m *Model) Generate(cache *KVCache, h []float64, startPos, numTokens int) []int {
	out := []int{}
	for pos := startPos; len(out) < numTokens; pos++ {
		next := m.nextToken(h)
		out = append(out, next)
		h = m.DecodeStep(next, pos, cache)
	}
	return out
}

// end::nocache[]

// tag::memory[]
// KVCacheBytes는 컨텍스트를 가득 채웠을 때 KV 캐시가 차지하는 크기다.
// K와 V 두 벌 × 층 수 × KV 헤드 수 × 헤드 차원 × 컨텍스트 길이 × 원소 크기.
func KVCacheBytes(layers, kvHeads, headDim, ctxLen, bytesPerElem int) int {
	return 2 * layers * kvHeads * headDim * ctxLen * bytesPerElem
}

// end::memory[]

func formatBytes(b int) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(b)/(1<<20))
	default:
		return fmt.Sprintf("%.1f KiB", float64(b)/(1<<10))
	}
}

func sameVec(a, b []float64) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return len(a) == len(b)
}

func sameTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func main() {
	rng := rand.New(rand.NewSource(42))
	model := NewModel(rng)

	const promptLen, genLen = 256, 64
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = rng.Intn(vocabSize)
	}

	fmt.Printf("모델: %d층, 임베딩 %d, 헤드 %d개 | 프롬프트 %d토큰, 생성 %d토큰, CPU 코어 %d개\n\n",
		nLayer, nEmbd, nHead, promptLen, genLen, runtime.NumCPU())

	// 1) 프리필: 순차 처리와 병렬 처리
	t0 := time.Now()
	seqCache := model.NewCache()
	var hSeq []float64
	for pos, t := range prompt {
		hSeq = model.DecodeStep(t, pos, seqCache)
	}
	seqPrefill := time.Since(t0)

	t0 = time.Now()
	cache, h := model.Prefill(prompt)
	parPrefill := time.Since(t0)

	fmt.Println("== 프리필: 프롬프트 처리 ==")
	fmt.Printf("한 토큰씩 순차 처리:  %8s\n", seqPrefill.Round(time.Millisecond/10))
	fmt.Printf("위치들을 병렬 처리:   %8s (%.1f배)\n", parPrefill.Round(time.Millisecond/10),
		float64(seqPrefill)/float64(parPrefill))
	fmt.Printf("두 방식의 마지막 은닉 벡터 일치: %v\n\n", sameVec(hSeq, h))

	// 2) 디코드: 캐시 사용과 캐시 없음
	t0 = time.Now()
	withCache := model.Generate(cache, h, promptLen, genLen)
	cachedTime := time.Since(t0)

	t0 = time.Now()
	noCache := model.GenerateNoCache(prompt, genLen)
	noCacheTime := time.Since(t0)

	fmt.Println("== 디코드: 토큰 생성 ==")
	fmt.Printf("KV 캐시 사용:  %8s (토큰당 %s)\n", cachedTime.Round(time.Millisecond/10),
		(cachedTime / genLen).Round(time.Microsecond*10))
	fmt.Printf("캐시 없음:     %8s (토큰당 %s, %.0f배)\n", noCacheTime.Round(time.Millisecond/10),
		(noCacheTime / genLen).Round(time.Microsecond*10), float64(noCacheTime)/float64(cachedTime))
	fmt.Printf("두 방식의 생성 결과 일치: %v\n\n", sameTokens(withCache, noCache))

	// 3) KV 캐시 메모리 크기
	fmt.Println("== KV 캐시 크기 ==")
	fmt.Printf("이 예제 모델 (%d층, 헤드 %d개, 컨텍스트 %d, float64): %s\n",
		nLayer, nHead, maxCtx, formatBytes(KVCacheBytes(nLayer, nHead, headDim, maxCtx, 8)))
	// Llama 3 8B: 32층, KV 헤드 8개(GQA), 헤드 차원 128, FP16(2바이트)
	fmt.Println("Llama 3 8B (32층, KV 헤드 8개, 헤드 차원 128, FP16):")
	for _, ctx := range []int{8192, 131072} {
		fmt.Printf("  컨텍스트 %7d 토큰: %s\n", ctx, formatBytes(KVCacheBytes(32, 8, 128, ctx, 2)))
	}
	fmt.Printf("GQA 없이 KV 헤드 32개라면, 컨텍스트 %d 토큰: %s\n",
		131072, formatBytes(KVCacheBytes(32, 32, 128, 131072, 2)))
}
