package main

import (
	"math"
	"math/rand"
)

// 모델 크기 설정. 파라미터 수천 개 수준의 아주 작은 GPT다.
const (
	nEmbd     = 16            // 임베딩 차원
	nHead     = 2             // 어텐션 헤드 수
	headDim   = nEmbd / nHead // 헤드 하나가 담당하는 차원
	nLayer    = 1             // 트랜스포머 블록 수
	blockSize = 16            // 최대 문맥 길이
)

// Block은 트랜스포머 블록 하나의 가중치 묶음이다.
type Block struct {
	attnQ, attnK, attnV [][]*Value // 어텐션 Q, K, V 투영 (nEmbd x nEmbd)
	attnProj            [][]*Value // 어텐션 출력 투영 (nEmbd x nEmbd)
	mlpFC               [][]*Value // MLP 확장 (4*nEmbd x nEmbd)
	mlpProj             [][]*Value // MLP 축소 (nEmbd x 4*nEmbd)
}

// GPT는 토큰 임베딩, 위치 임베딩, 트랜스포머 블록, 최종 lm_head로 이루어진다.
type GPT struct {
	vocabSize int
	wte       [][]*Value // 토큰 임베딩 (vocabSize x nEmbd)
	wpe       [][]*Value // 위치 임베딩 (blockSize x nEmbd)
	blocks    []*Block
	lmHead    [][]*Value // 최종 로짓 투영 (vocabSize x nEmbd)
	Params    []*Value   // 옵티마이저가 갱신할 전체 파라미터 목록
}

// NewGPT는 작은 GPT를 만들고 가중치를 정규분포(표준편차 0.02)로 초기화한다.
func NewGPT(vocabSize int, rng *rand.Rand) *GPT {
	m := &GPT{vocabSize: vocabSize}
	newMatrix := func(rows, cols int) [][]*Value {
		w := make([][]*Value, rows)
		for i := range w {
			w[i] = make([]*Value, cols)
			for j := range w[i] {
				w[i][j] = NewValue(rng.NormFloat64() * 0.02)
				m.Params = append(m.Params, w[i][j])
			}
		}
		return w
	}
	m.wte = newMatrix(vocabSize, nEmbd)
	m.wpe = newMatrix(blockSize, nEmbd)
	for l := 0; l < nLayer; l++ {
		m.blocks = append(m.blocks, &Block{
			attnQ:    newMatrix(nEmbd, nEmbd),
			attnK:    newMatrix(nEmbd, nEmbd),
			attnV:    newMatrix(nEmbd, nEmbd),
			attnProj: newMatrix(nEmbd, nEmbd),
			mlpFC:    newMatrix(4*nEmbd, nEmbd),
			mlpProj:  newMatrix(nEmbd, 4*nEmbd),
		})
	}
	m.lmHead = newMatrix(vocabSize, nEmbd)
	return m
}

// KVCache는 이미 처리한 위치들의 키와 값을 층별로 보관한다.
// 학습과 샘플링 모두 이 캐시에 한 위치씩 쌓아 가며 진행한다.
type KVCache struct {
	keys   [][][]*Value // [층][위치][차원]
	values [][][]*Value
}

// NewCache는 시퀀스 하나를 처리할 빈 캐시를 만든다.
func (m *GPT) NewCache() *KVCache {
	return &KVCache{
		keys:   make([][][]*Value, nLayer),
		values: make([][][]*Value, nLayer),
	}
}

// matvec은 행렬 w와 벡터 x의 곱을 계산한다.
func matvec(w [][]*Value, x []*Value) []*Value {
	out := make([]*Value, len(w))
	for i, row := range w {
		sum := row[0].Mul(x[0])
		for j := 1; j < len(x); j++ {
			sum = sum.Add(row[j].Mul(x[j]))
		}
		out[i] = sum
	}
	return out
}

// rmsnorm은 벡터를 제곱 평균의 제곱근으로 나눠 크기를 일정하게 유지한다.
func rmsnorm(x []*Value) []*Value {
	ss := x[0].Mul(x[0])
	for i := 1; i < len(x); i++ {
		ss = ss.Add(x[i].Mul(x[i]))
	}
	// 작은 값을 더해 0으로 나누는 상황을 막는다
	scale := ss.Mul(NewValue(1 / float64(len(x)))).Add(NewValue(1e-5)).Pow(-0.5)
	out := make([]*Value, len(x))
	for i := range x {
		out[i] = x[i].Mul(scale)
	}
	return out
}

// Softmax는 로짓을 확률 분포로 바꾼다.
func Softmax(logits []*Value) []*Value {
	// 수치 안정성을 위해 최댓값을 빼고 지수를 취한다
	maxVal := logits[0].Data
	for _, l := range logits {
		if l.Data > maxVal {
			maxVal = l.Data
		}
	}
	shift := NewValue(maxVal)
	exps := make([]*Value, len(logits))
	sum := logits[0].Sub(shift).Exp()
	exps[0] = sum
	for i := 1; i < len(logits); i++ {
		exps[i] = logits[i].Sub(shift).Exp()
		sum = sum.Add(exps[i])
	}
	inv := sum.Pow(-1)
	out := make([]*Value, len(logits))
	for i := range exps {
		out[i] = exps[i].Mul(inv)
	}
	return out
}

// tag::forward[]
// Forward는 토큰 하나를 받아 다음 토큰의 로짓을 계산한다.
// 이전 위치의 K, V는 캐시에 쌓여 있으므로 현재 토큰은
// 과거 위치만 참조하게 되고, causal 마스크가 따로 필요 없다.
func (m *GPT) Forward(tokenID, posID int, cache *KVCache) []*Value {
	// 토큰 임베딩과 위치 임베딩을 더해 입력 벡터를 만든다
	x := make([]*Value, nEmbd)
	for i := 0; i < nEmbd; i++ {
		x[i] = m.wte[tokenID][i].Add(m.wpe[posID][i])
	}

	for l, blk := range m.blocks {
		// tag::attention[]
		// 1) 멀티헤드 어텐션: 현재 토큰이 이전 토큰들의 정보를 모은다
		xn := rmsnorm(x)
		q := matvec(blk.attnQ, xn)
		k := matvec(blk.attnK, xn)
		v := matvec(blk.attnV, xn)
		cache.keys[l] = append(cache.keys[l], k)
		cache.values[l] = append(cache.values[l], v)

		attnOut := make([]*Value, nEmbd)
		invSqrt := NewValue(1 / math.Sqrt(float64(headDim)))
		for h := 0; h < nHead; h++ {
			lo := h * headDim
			// 현재 질의(q)와 캐시된 키들의 내적을 headDim 제곱근으로 나눈다
			scores := make([]*Value, len(cache.keys[l]))
			for t, kt := range cache.keys[l] {
				dot := q[lo].Mul(kt[lo])
				for d := 1; d < headDim; d++ {
					dot = dot.Add(q[lo+d].Mul(kt[lo+d]))
				}
				scores[t] = dot.Mul(invSqrt)
			}
			// 소프트맥스 가중치로 캐시된 값(v) 벡터들을 섞는다
			weights := Softmax(scores)
			for d := 0; d < headDim; d++ {
				sum := weights[0].Mul(cache.values[l][0][lo+d])
				for t := 1; t < len(weights); t++ {
					sum = sum.Add(weights[t].Mul(cache.values[l][t][lo+d]))
				}
				attnOut[lo+d] = sum
			}
		}
		// 출력 투영을 거쳐 잔차 연결로 원래 벡터에 더한다
		proj := matvec(blk.attnProj, attnOut)
		for i := range x {
			x[i] = x[i].Add(proj[i])
		}
		// end::attention[]

		// 2) MLP 블록: 토큰별 특징을 4배 넓힌 뒤 비선형 변환하고 되돌린다
		xn = rmsnorm(x)
		hidden := matvec(blk.mlpFC, xn)
		for i := range hidden {
			hidden[i] = hidden[i].Tanh()
		}
		proj = matvec(blk.mlpProj, hidden)
		for i := range x {
			x[i] = x[i].Add(proj[i])
		}
	}

	// 최종 정규화 후 어휘 크기의 로짓으로 투영한다
	return matvec(m.lmHead, rmsnorm(x))
}

// end::forward[]
