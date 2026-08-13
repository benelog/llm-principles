package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
)

func main() {
	// 재현 가능하도록 난수 시드를 고정한다
	rng := rand.New(rand.NewSource(42))

	raw, err := os.ReadFile("names.txt")
	if err != nil {
		fmt.Fprintln(os.Stderr, "names.txt를 읽을 수 없습니다:", err)
		os.Exit(1)
	}
	docs := strings.Split(strings.TrimSpace(string(raw)), "\n")

	tok := NewTokenizer(docs)
	model := NewGPT(tok.VocabSize(), rng)
	fmt.Printf("데이터 %d건, 어휘 크기 %d, 파라미터 %d개\n",
		len(docs), tok.VocabSize(), len(model.Params))

	// tag::training[]
	const (
		numSteps = 1500 // 학습 스텝 수
		baseLR   = 0.01 // 기본 학습률
		beta1    = 0.9  // Adam 1차 모멘트 감쇠율
		beta2    = 0.95 // Adam 2차 모멘트 감쇠율
		epsAdam  = 1e-8
	)
	mBuf := make([]float64, len(model.Params)) // Adam 1차 모멘트(기울기 이동 평균)
	vBuf := make([]float64, len(model.Params)) // Adam 2차 모멘트(기울기 제곱 이동 평균)

	lossSum, lossCount := 0.0, 0
	for step := 1; step <= numSteps; step++ {
		// 이름 하나를 골라 시퀀스를 만든다. 입력과 목표는 한 칸씩 어긋난다.
		// 예: seq = [BOS, e, m, m, a, BOS] 이면
		//     입력 BOS -> 목표 e, 입력 e -> 목표 m, ... 입력 a -> 목표 BOS
		doc := docs[rng.Intn(len(docs))]
		seq := append([]int{tok.BOS}, tok.Encode(doc)...)
		seq = append(seq, tok.BOS)
		n := len(seq) - 1
		if n > blockSize {
			n = blockSize
		}

		// 순전파: 위치마다 다음 토큰의 확률을 구하고 교차 엔트로피를 쌓는다
		cache := model.NewCache()
		loss := NewValue(0)
		for pos := 0; pos < n; pos++ {
			logits := model.Forward(seq[pos], pos, cache)
			probs := Softmax(logits)
			// 정답 토큰 확률의 로그에 음수를 취한 값이 손실이다
			loss = loss.Add(probs[seq[pos+1]].Log().Neg())
		}
		loss = loss.Mul(NewValue(1 / float64(n)))

		// 역전파: 기울기를 0으로 되돌린 뒤 손실에서 출발해 전파한다
		for _, p := range model.Params {
			p.Grad = 0
		}
		loss.Backward()

		// Adam 갱신. 학습률은 마지막 스텝에서 0이 되도록 선형 감소시킨다.
		lr := baseLR * (1 - float64(step-1)/float64(numSteps))
		for i, p := range model.Params {
			mBuf[i] = beta1*mBuf[i] + (1-beta1)*p.Grad
			vBuf[i] = beta2*vBuf[i] + (1-beta2)*p.Grad*p.Grad
			mHat := mBuf[i] / (1 - math.Pow(beta1, float64(step)))
			vHat := vBuf[i] / (1 - math.Pow(beta2, float64(step)))
			p.Data -= lr * mHat / (math.Sqrt(vHat) + epsAdam)
		}

		lossSum += loss.Data
		lossCount++
		if step == 1 || step%100 == 0 {
			fmt.Printf("스텝 %4d | 평균 손실 %.4f\n", step, lossSum/float64(lossCount))
			lossSum, lossCount = 0, 0
		}
	}
	// end::training[]

	// tag::sampling[]
	// 샘플링: BOS 토큰으로 시작해 다음 토큰을 확률에 따라 뽑는다.
	// temperature가 낮을수록 확률이 높은 토큰에 집중한다.
	const temperature = 0.8
	fmt.Println("\n생성된 이름 20개:")
	for i := 0; i < 20; i++ {
		cache := model.NewCache()
		tokenID := tok.BOS
		out := []int{}
		for pos := 0; pos < blockSize; pos++ {
			logits := model.Forward(tokenID, pos, cache)
			probs := softmaxData(logits, temperature)
			next := sampleFrom(probs, rng)
			if next == tok.BOS {
				break // BOS가 나오면 이름이 끝난 것으로 본다
			}
			out = append(out, next)
			tokenID = next
		}
		fmt.Println(tok.Decode(out))
	}
	// end::sampling[]
}

// softmaxData는 역전파 그래프 없이 로짓 값만으로 확률을 계산한다.
func softmaxData(logits []*Value, temperature float64) []float64 {
	maxVal := logits[0].Data
	for _, l := range logits {
		if l.Data > maxVal {
			maxVal = l.Data
		}
	}
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, l := range logits {
		probs[i] = math.Exp((l.Data - maxVal) / temperature)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

// sampleFrom은 누적 확률을 따라 토큰 하나를 뽑는다.
func sampleFrom(probs []float64, rng *rand.Rand) int {
	r := rng.Float64()
	acc := 0.0
	for i, p := range probs {
		acc += p
		if r < acc {
			return i
		}
	}
	return len(probs) - 1
}
