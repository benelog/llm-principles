package main

import (
	"math"
	"math/rand/v2"
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

// 부록 "평균, 분산, 정규화"
func moments(c *checker) {
	c.section("평균, 분산, 정규화")

	x := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	c.near("평균", 5, stat.Mean(x, nil), 1e-12)
	// 표본 분산은 n-1로 나누므로 모분산 4와는 값이 다르다.
	c.near("표본 분산", 32.0/7, stat.Variance(x, nil), 1e-12)
	c.near("표준편차는 분산의 제곱근", math.Sqrt(stat.Variance(x, nil)), stat.StdDev(x, nil), 1e-12)

	// 부록: 분산은 자기 자신과의 공분산이다.
	c.near("분산 = 자기 자신과의 공분산", stat.Variance(x, nil), stat.Covariance(x, x, nil), 1e-12)

	// 부록: 평균 0, 분산 1인 값으로 이루어진 d차원 벡터 두 개를 내적하면
	// 결과의 분산이 d가 되고 표준편차는 sqrt(d)가 된다.
	src := rand.New(rand.NewPCG(7, 8))
	const d, trials = 64, 20000
	dots := make([]float64, trials)
	for i := range dots {
		dots[i] = floats.Dot(randSlice(src, d), randSlice(src, d))
	}
	c.near("d차원 내적의 분산 (d=64)", 64, stat.Variance(dots, nil), 3)
	c.near("d차원 내적의 표준편차", 8, stat.StdDev(dots, nil), 0.3)

	// 부록: 내적을 sqrt(d)로 나누면 차원이 몇이든 점수의 분산이 1 근처로 유지된다.
	scaled := make([]float64, trials)
	copy(scaled, dots)
	floats.Scale(1/math.Sqrt(d), scaled)
	c.near("sqrt(d)로 나눈 뒤의 분산", 1, stat.Variance(scaled, nil), 0.05)

	// 차원을 256으로 바꿔도 같은 결론이 나오는지 확인한다.
	const d2 = 256
	dots2 := make([]float64, trials)
	for i := range dots2 {
		dots2[i] = floats.Dot(randSlice(src, d2), randSlice(src, d2)) / math.Sqrt(d2)
	}
	c.near("차원을 256으로 바꿔도 분산 1", 1, stat.Variance(dots2, nil), 0.05)

	// 부록: 5장의 RMSNorm은 벡터를 자기 크기로 나눠 규모를 되돌린다.
	v := []float64{3, -4, 12, 0.5}
	rms := math.Sqrt(floats.Dot(v, v) / float64(len(v)))
	normed := make([]float64, len(v))
	floats.ScaleTo(normed, 1/rms, v)
	newRMS := math.Sqrt(floats.Dot(normed, normed) / float64(len(normed)))
	c.near("RMSNorm 뒤의 RMS", 1, newRMS, 1e-12)
}

// 부록 "정규 분포: 무작위가 합쳐질 때 나타나는 모양"
func normalDistribution(c *checker) {
	c.section("정규 분포: 무작위가 합쳐질 때 나타나는 모양")

	n := distuv.Normal{Mu: 0, Sigma: 1}

	// 부록: 값의 약 68%가 평균 ±1 표준편차 안에, 약 95%가 ±2 표준편차 안에 들어온다.
	within1 := n.CDF(1) - n.CDF(-1)
	within2 := n.CDF(2) - n.CDF(-2)
	c.near("±1 표준편차 안의 비율", 0.68, within1, 0.005)
	c.near("±2 표준편차 안의 비율", 0.95, within2, 0.005)
	c.note("정확한 값은 %.4f와 %.4f다", within1, within2)

	// 부록: 평균을 중심으로 좌우 대칭이다.
	c.near("좌우 대칭 (CDF(0) = 0.5)", 0.5, n.CDF(0), 1e-12)
	c.near("대칭성 P(X<-1.5) = P(X>1.5)", n.CDF(-1.5), 1-n.CDF(1.5), 1e-12)

	// 부록: 중심 극한 정리. 개별 값이 어떤 분포를 따르든 여러 개를 더하면
	// 그 합은 정규 분포에 가까워진다. 균등 분포 12개의 합으로 확인한다.
	src := rand.New(rand.NewPCG(9, 10))
	const trials = 200000
	uniformSum := func(terms int) []float64 {
		out := make([]float64, trials)
		for i := range out {
			s := 0.0
			for j := 0; j < terms; j++ {
				s += src.Float64() - 0.5 // 평균 0, 분산 1/12
			}
			out[i] = s
		}
		return out
	}
	sums := uniformSum(12) // 항이 12개이므로 분산은 1이 된다
	c.near("균등 분포 12개 합의 평균", 0, stat.Mean(sums, nil), 0.02)
	c.near("균등 분포 12개 합의 분산", 1, stat.Variance(sums, nil), 0.03)

	// 왜도는 좌우 대칭이면 0이다. 균등 분포의 합은 대칭이므로 항이 몇 개든 0이다.
	c.near("합의 왜도 (정규 분포는 0)", 0, stat.Skew(sums, nil), 0.03)

	// 초과 첨도는 정규 분포에서 0이지만, 유한한 항의 합은 아직 거기에 닿지 않는다.
	// 균등 분포 n개 합의 이론값은 -6/(5n)이므로 12개면 -0.1이다. 정규 분포에
	// "가까워진다"는 것이지 같아진다는 뜻이 아니라는 사실이 이 값에 드러난다.
	k12 := stat.ExKurtosis(sums, nil)
	c.near("12개 합의 초과 첨도 (이론값 -6/60)", -0.1, k12, 0.02)

	// 항을 늘리면 그 값이 0으로 다가간다. 이것이 중심 극한 정리의 내용이다.
	k48 := stat.ExKurtosis(uniformSum(48), nil)
	c.ok("항을 늘리면 정규 분포에 더 가까워짐", math.Abs(k48) < math.Abs(k12),
		"12개 "+fmtNum(k12)+" → 48개 "+fmtNum(k48))

	// 부록: 7장 NF4는 균등 간격 대신 정규 분포의 분위수에 격자를 배치해서
	// 같은 비트 수로 오차를 줄인다. 2비트(격자 4개)로 직접 비교한다.
	sample := make([]float64, 20000)
	for i := range sample {
		sample[i] = src.NormFloat64()
	}
	uniformGrid := []float64{-3, -1, 1, 3}
	quantileGrid := make([]float64, 4)
	for i := range quantileGrid {
		// 구간을 넷으로 나눈 각 구간의 가운데 분위수를 격자로 쓴다.
		quantileGrid[i] = n.Quantile((float64(i) + 0.5) / 4)
	}
	errUniform := quantizeError(sample, uniformGrid)
	errQuantile := quantizeError(sample, quantileGrid)
	c.ok("분위수 격자가 균등 격자보다 오차가 작다", errQuantile < errUniform,
		fmtNum(errQuantile)+" < "+fmtNum(errUniform))
	c.note("분위수 격자: [%.3f %.3f %.3f %.3f]",
		quantileGrid[0], quantileGrid[1], quantileGrid[2], quantileGrid[3])
}

// 부록 "공분산과 상관: 함께 움직이는 정도"
func covarianceAndCorr(c *checker) {
	c.section("공분산과 상관: 함께 움직이는 정도")

	src := rand.New(rand.NewPCG(11, 12))
	const n = 500
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = src.NormFloat64()
		y[i] = 2*x[i] + 0.8*src.NormFloat64() // 같은 방향으로 함께 움직인다
	}

	cov := stat.Covariance(x, y, nil)
	corr := stat.Correlation(x, y, nil)
	c.ok("함께 커지면 공분산이 양수", cov > 0, "공분산 "+fmtNum(cov))
	c.ok("상관계수는 -1에서 1 사이", corr > 0 && corr < 1, "상관계수 "+fmtNum(corr))

	// 부록: 공분산을 두 표준편차로 나눈 것이 상관계수다.
	c.near("상관계수 = 공분산 / (표준편차 곱)",
		cov/(stat.StdDev(x, nil)*stat.StdDev(y, nil)), corr, 1e-12)

	// 부록의 핵심 주장: 상관계수는 "평균을 뺀 두 값 배열의 코사인 유사도"와
	// 정확히 같은 계산이다. 10장의 벡터 검색에 쓴 그 코사인이다.
	cx := centered(x)
	cy := centered(y)
	cosine := floats.Dot(cx, cy) / (floats.Norm(cx, 2) * floats.Norm(cy, 2))
	c.near("상관계수 = 중심화한 벡터의 코사인", corr, cosine, 1e-12)

	// 부록: 반대로 움직이면 음수, 무관하면 0 근처다.
	neg := make([]float64, n)
	for i := range neg {
		neg[i] = -y[i]
	}
	c.ok("반대로 움직이면 음수", stat.Correlation(x, neg, nil) < 0,
		fmtNum(stat.Correlation(x, neg, nil)))
	indep := randSlice(src, n)
	c.near("무관하면 0 근처", 0, stat.Correlation(x, indep, nil), 0.1)

	// 부록: 변수가 여러 개면 공분산 행렬이 되고, 주성분은 이 행렬에서 계산된다.
	// 분산이 가장 큰 방향은 공분산 행렬의 최대 고유벡터다.
	data := mat.NewDense(n, 2, nil)
	for i := 0; i < n; i++ {
		data.Set(i, 0, x[i])
		data.Set(i, 1, y[i])
	}
	var covMat mat.SymDense
	stat.CovarianceMatrix(&covMat, data, nil)
	c.near("공분산 행렬의 (0,1) 성분", cov, covMat.At(0, 1), 1e-12)

	var eig mat.EigenSym
	if !eig.Factorize(&covMat, true) {
		c.ok("공분산 행렬의 고유분해", false, "Factorize 실패")
		return
	}
	var eigVecs mat.Dense
	eig.VectorsTo(&eigVecs)
	eigVals := eig.Values(nil)
	// Gonum은 고유값을 오름차순으로 돌려주므로 마지막 열이 최대 고유벡터다.
	top := mat.Col(nil, len(eigVals)-1, &eigVecs)

	var pc stat.PC
	pc.PrincipalComponents(data, nil)
	var pcVecs mat.Dense
	pc.VectorsTo(&pcVecs)
	first := mat.Col(nil, 0, &pcVecs)

	// 고유벡터의 부호는 정해지지 않으므로 방향의 절대 일치만 본다.
	align := math.Abs(floats.Dot(top, first))
	c.near("제1주성분 = 공분산 행렬의 최대 고유벡터", 1, align, 1e-8)
}

// 부록 "k-평균 군집화: 데이터에서 대표점 찾기"
func kmeansSection(c *checker) {
	c.section("k-평균 군집화: 데이터에서 대표점 찾기")

	// 부록: 대표점 k개를 놓고 "가장 가까운 대표점에 배정한다, 대표점을 배정된
	// 데이터의 평균으로 옮긴다"를 반복하면 대표점이 밀집한 곳으로 이동한다.
	// Gonum에는 k-평균이 없으므로 Lloyd 알고리즘을 그대로 구현한다.
	src := rand.New(rand.NewPCG(13, 14))
	trueCenters := []float64{-5, 0, 7}
	var data []float64
	for _, m := range trueCenters {
		for i := 0; i < 400; i++ {
			data = append(data, m+0.6*src.NormFloat64())
		}
	}

	centers, iters := lloyd(data, []float64{-1, 0, 1}, 100)
	sort.Float64s(centers)
	maxGap := 0.0
	for i, want := range trueCenters {
		maxGap = math.Max(maxGap, math.Abs(centers[i]-want))
	}
	c.ok("대표점이 밀집한 곳으로 이동", maxGap < 0.15,
		"참값과 최대 차이 "+fmtNum(maxGap))
	c.note("수렴까지 %d회 반복, 찾은 대표점 [%.3f %.3f %.3f]",
		iters, centers[0], centers[1], centers[2])

	// 부록: 반복할수록 목적함수(각 점과 배정된 대표점 거리의 제곱합)가 줄어든다.
	prev := math.Inf(1)
	monotone := true
	cur := []float64{-1, 0, 1}
	for i := 0; i < 15; i++ {
		cur, _ = lloyd(data, cur, 1)
		d := inertia(data, cur)
		if d > prev+1e-9 {
			monotone = false
		}
		prev = d
	}
	c.ok("반복할 때마다 목적함수가 줄어든다", monotone, "Lloyd 알고리즘의 성질")

	// 부록: 분포를 모른 채 데이터에서 직접 격자를 찾는 일반해가 1차원 k-평균이고,
	// 값이 밀집한 곳에 대표점이 몰리는 최적의 양자화 격자가 같은 절차로 나온다.
	// 7장의 NF4 자리에서 정규 분포 샘플에 대해 균등 격자와 비교한다.
	sample := make([]float64, 20000)
	for i := range sample {
		sample[i] = src.NormFloat64()
	}
	uniformGrid := []float64{-3, -1, 1, 3}
	learned, _ := lloyd(sample, []float64{-2, -0.5, 0.5, 2}, 200)
	sort.Float64s(learned)

	errUniform := quantizeError(sample, uniformGrid)
	errLearned := quantizeError(sample, learned)
	c.ok("k-평균 격자가 균등 격자보다 오차가 작다", errLearned < errUniform,
		fmtNum(errLearned)+" < "+fmtNum(errUniform))
	c.note("k-평균이 찾은 격자: [%.3f %.3f %.3f %.3f]",
		learned[0], learned[1], learned[2], learned[3])

	// 값이 밀집한 0 근처에 대표점이 몰렸는지 확인한다.
	inner := learned[2] - learned[1]
	outer := learned[3] - learned[2]
	c.ok("밀집한 곳의 격자 간격이 더 좁다", inner < outer,
		"가운데 간격 "+fmtNum(inner)+" < 바깥 간격 "+fmtNum(outer))
}

// lloyd는 1차원 k-평균의 반복을 수행하고 대표점과 반복 횟수를 돌려준다.
func lloyd(data, init []float64, maxIter int) ([]float64, int) {
	centers := append([]float64(nil), init...)
	k := len(centers)
	for it := 1; it <= maxIter; it++ {
		sums := make([]float64, k)
		counts := make([]int, k)
		for _, v := range data {
			j := nearest(centers, v)
			sums[j] += v
			counts[j]++
		}
		moved := 0.0
		for j := range centers {
			if counts[j] == 0 {
				continue
			}
			next := sums[j] / float64(counts[j])
			moved = math.Max(moved, math.Abs(next-centers[j]))
			centers[j] = next
		}
		if moved < 1e-10 {
			return centers, it
		}
	}
	return centers, maxIter
}

func inertia(data, centers []float64) float64 {
	total := 0.0
	for _, v := range data {
		d := v - centers[nearest(centers, v)]
		total += d * d
	}
	return total
}

func nearest(centers []float64, v float64) int {
	best, bestDist := 0, math.Inf(1)
	for j, ctr := range centers {
		if d := math.Abs(ctr - v); d < bestDist {
			best, bestDist = j, d
		}
	}
	return best
}

// quantizeError는 각 값을 가장 가까운 격자점으로 바꿨을 때의 평균 제곱 오차다.
func quantizeError(values, grid []float64) float64 {
	total := 0.0
	for _, v := range values {
		d := v - grid[nearest(grid, v)]
		total += d * d
	}
	return total / float64(len(values))
}

func centered(x []float64) []float64 {
	out := append([]float64(nil), x...)
	floats.AddConst(-stat.Mean(x, nil), out)
	return out
}
