// 양자화 알고리즘의 원리를 측정으로 확인하는 예제.
// 선형 양자화, 블록 단위 배율, 정규 분포 분위수 격자(NF4의 아이디어)를
// 외부 라이브러리 없이 표준 라이브러리만으로 구현한다.
package main

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// tag::absmax[]
// quantizeAbsmax는 가장 단순한 선형(대칭) 양자화다.
// 배열의 최대 절댓값이 정수 범위의 끝(maxQ)에 오도록 배율(scale)을 정하고,
// 각 값을 배율로 나눠 가장 가까운 정수로 반올림한다.
// INT8이면 maxQ=127, 4비트면 maxQ=7이다.
func quantizeAbsmax(w []float64, maxQ int) (q []int, scale float64) {
	absMax := 0.0
	for _, v := range w {
		absMax = math.Max(absMax, math.Abs(v))
	}
	scale = absMax / float64(maxQ)
	q = make([]int, len(w))
	for i, v := range w {
		q[i] = int(math.Round(v / scale))
	}
	return q, scale
}

// dequantize는 정수에 배율을 도로 곱해 근사값을 복원한다.
// 반올림에서 잃은 정보는 돌아오지 않는다. 그 차이가 양자화 오차다.
func dequantize(q []int, scale float64) []float64 {
	w := make([]float64, len(q))
	for i, v := range q {
		w[i] = float64(v) * scale
	}
	return w
}

// end::absmax[]

// tag::blocks[]
// quantizeBlocks는 배열을 blockSize개씩 잘라 블록마다 따로 양자화한다.
// 블록마다 배율 하나를 추가로 저장하는 대가로,
// 큰 값 하나가 배열 전체의 격자를 성기게 만드는 것을 막는다.
func quantizeBlocks(w []float64, maxQ, blockSize int) []float64 {
	out := make([]float64, 0, len(w))
	for start := 0; start < len(w); start += blockSize {
		end := min(start+blockSize, len(w))
		q, scale := quantizeAbsmax(w[start:end], maxQ)
		out = append(out, dequantize(q, scale)...)
	}
	return out
}

// end::blocks[]

// tag::normal-grid[]
// normalGrid는 표준 정규 분포를 n개의 등확률 구간으로 나누고
// 각 구간의 중앙 분위수를 격자점으로 삼는다. 값이 밀집한 0 근처에는
// 격자점이 촘촘히, 값이 드문 꼬리 쪽에는 성기게 배치된다.
// QLoRA의 NF4가 이 아이디어를 다듬은 4비트(16개 점) 격자다.
func normalGrid(n int) []float64 {
	grid := make([]float64, n)
	for i := range grid {
		p := (float64(i) + 0.5) / float64(n)
		grid[i] = math.Sqrt2 * math.Erfinv(2*p-1) // 표준 정규 분포 누적 함수의 역함수
	}
	last := grid[n-1]
	for i := range grid {
		grid[i] /= last // [-1, 1] 범위로 정규화
	}
	return grid
}

// uniformGrid는 [-1, 1]을 같은 간격으로 나눈 격자다.
func uniformGrid(n int) []float64 {
	grid := make([]float64, n)
	for i := range grid {
		grid[i] = -1 + 2*float64(i)/float64(n-1)
	}
	return grid
}

// quantizeGrid는 각 값을 최대 절댓값으로 정규화한 뒤
// 격자에서 가장 가까운 점으로 바꾼다.
func quantizeGrid(w []float64, grid []float64) []float64 {
	absMax := 0.0
	for _, v := range w {
		absMax = math.Max(absMax, math.Abs(v))
	}
	out := make([]float64, len(w))
	for i, v := range w {
		best := grid[0]
		for _, g := range grid[1:] {
			if math.Abs(v/absMax-g) < math.Abs(v/absMax-best) {
				best = g
			}
		}
		out[i] = best * absMax
	}
	return out
}

// end::normal-grid[]

// rmsError는 원본과 복원값 사이의 평균 제곱근 오차를 잰다.
func rmsError(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(a)))
}

func main() {
	// 학습된 가중치의 전형인 평균 0의 정규 분포에서 4,096개를 뽑는다.
	// 시드를 고정했으므로 실행할 때마다 같은 결과가 나온다.
	rng := rand.New(rand.NewPCG(7, 7))
	w := make([]float64, 4096)
	for i := range w {
		w[i] = rng.NormFloat64()
	}

	fmt.Println("== 선형 양자화 (표준편차 1.0인 가중치 4,096개) ==")
	q8, s8 := quantizeAbsmax(w, 127)
	fmt.Printf("INT8 (격자 255개):  RMS 오차 %.4f\n", rmsError(w, dequantize(q8, s8)))
	q4, s4 := quantizeAbsmax(w, 7)
	fmt.Printf("4비트 (격자 15개):  RMS 오차 %.4f\n", rmsError(w, dequantize(q4, s4)))

	fmt.Println("\n== 아웃라이어 하나가 끼면 (w[0]을 20.0으로) ==")
	wo := append([]float64(nil), w...)
	wo[0] = 20.0
	qo, so := quantizeAbsmax(wo, 7)
	fmt.Printf("4비트, 텐서 전체에 배율 하나:  RMS 오차 %.4f\n", rmsError(wo, dequantize(qo, so)))
	fmt.Printf("4비트, 64개 블록마다 배율:     RMS 오차 %.4f\n", rmsError(wo, quantizeBlocks(wo, 7, 64)))

	fmt.Println("\n== 격자를 어디에 둘 것인가 (아웃라이어 없는 원본, 격자 16개) ==")
	fmt.Printf("균등 간격 격자:         RMS 오차 %.4f\n", rmsError(w, quantizeGrid(w, uniformGrid(16))))
	fmt.Printf("정규 분포 분위수 격자:  RMS 오차 %.4f\n", rmsError(w, quantizeGrid(w, normalGrid(16))))
}
