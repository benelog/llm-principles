package main

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// 부록 "벡터와 행렬: 숫자 배열의 기하학"
func vectorsAndMatrices(c *checker) {
	c.section("벡터와 행렬: 숫자 배열의 기하학")

	a := []float64{1, 2}
	b := []float64{3, 4}

	// 부록: a·b = 1*3 + 2*4 = 11
	c.near("내적 a·b", 11, floats.Dot(a, b), 1e-12)

	// 부록: |a| = 2.236, |b| = 5
	c.near("벡터 길이 |a|", 2.236, floats.Norm(a, 2), 5e-4)
	c.near("벡터 길이 |b|", 5, floats.Norm(b, 2), 1e-12)

	// 부록: cos(a, b) = 11 / (2.236 * 5) ≈ 0.98
	cos := floats.Dot(a, b) / (floats.Norm(a, 2) * floats.Norm(b, 2))
	c.near("코사인 유사도 cos(a,b)", 0.98, cos, 5e-3)

	// 부록: 코사인 유사도는 항상 -1에서 1 사이에 들어온다.
	src := rand.New(rand.NewPCG(1, 2))
	inRange := true
	for i := 0; i < 1000; i++ {
		u, v := randVec(src, 32), randVec(src, 32)
		s := floats.Dot(u, v) / (floats.Norm(u, 2) * floats.Norm(v, 2))
		if s < -1-1e-12 || s > 1+1e-12 {
			inRange = false
		}
	}
	c.ok("코사인 값의 범위", inRange, "무작위 32차원 벡터 1000쌍이 모두 [-1, 1]")

	// 부록: 방향이 같으면 내적이 크고, 수직이면 0, 반대 방향이면 음수다.
	c.near("수직 벡터의 내적", 0, floats.Dot([]float64{1, 0}, []float64{0, 1}), 1e-12)
	c.ok("반대 방향의 내적", floats.Dot(a, []float64{-1, -2}) < 0, "a·(-a) < 0")

	// 부록: lmHead는 임베딩 차원의 벡터를 어휘 크기의 로짓으로 바꾸는 행렬 곱 하나다.
	// 임베딩 4차원, 어휘 3개인 lmHead를 그대로 재현한다.
	x := mat.NewVecDense(4, []float64{0.5, -1.0, 0.25, 2.0})
	lmHead := mat.NewDense(3, 4, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	})
	var logits mat.VecDense
	logits.MulVec(lmHead, x)
	c.ok("lmHead 투영의 출력 길이", logits.Len() == 3,
		"임베딩 4차원 → 로짓 3개")
	// 세 번째 행은 모든 성분의 절반을 더한 것이므로 (0.5-1+0.25+2)/2 = 0.875다.
	c.near("행렬 곱이 섞은 결과", 0.875, logits.AtVec(2), 1e-12)
}

// 부록 "랭크와 주성분 분석: 행렬을 적은 숫자로 근사하기"
func rankAndPCA(c *checker) {
	c.section("랭크와 주성분 분석: 행렬을 적은 숫자로 근사하기")

	src := rand.New(rand.NewPCG(3, 4))
	const d, r = 6, 2

	// 부록: 랭크가 r이면 d×r 행렬과 r×d 행렬의 곱으로 정확히 표현할 수 있다.
	// LoRA의 ΔW = B×A가 정확히 이 형태다.
	bMat := mat.NewDense(d, r, randSlice(src, d*r))
	aMat := mat.NewDense(r, d, randSlice(src, r*d))
	var deltaW mat.Dense
	deltaW.Mul(bMat, aMat)

	var svd mat.SVD
	svd.Factorize(&deltaW, mat.SVDThin)
	sv := svd.Values(nil)
	rank := 0
	for _, s := range sv {
		if s > 1e-10*sv[0] {
			rank++
		}
	}
	c.ok("B×A로 만든 행렬의 랭크", rank == r, "6×2 곱하기 2×6의 랭크는 2")

	// 부록: 저장할 숫자가 d²개에서 2dr개로 줄어든다.
	c.near("저장할 숫자 d²", 36, float64(d*d), 0)
	c.near("저장할 숫자 2dr", 24, float64(2*d*r), 0)

	// 부록: 랭크가 정확히 낮지 않아도 "거의 낮은" 경우에는 근사할 수 있다.
	// 저랭크 행렬에 작은 잡음을 더한 뒤 상위 2개 특이값만 남겨 복원한다.
	noisy := mat.DenseCopyOf(&deltaW)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			noisy.Set(i, j, noisy.At(i, j)+0.01*src.NormFloat64())
		}
	}
	approx := truncatedSVD(noisy, r)
	c.ok("상위 2개 특이값만 남긴 근사", relErr(noisy, approx) < 0.05,
		fmtNum(relErr(noisy, approx)*100)+"% 상대 오차")

	// 부록: 주성분 분석은 분산이 가장 큰 방향을 차례로 찾는다.
	// x축 방향으로 길게 퍼진 데이터를 만들면 제1주성분이 x축과 나란해야 한다.
	const n = 400
	data := mat.NewDense(n, 2, nil)
	for i := 0; i < n; i++ {
		data.Set(i, 0, 5*src.NormFloat64())
		data.Set(i, 1, 0.5*src.NormFloat64())
	}
	var pc stat.PC
	if !pc.PrincipalComponents(data, nil) {
		c.ok("주성분 분석 수행", false, "PrincipalComponents 실패")
		return
	}
	vars := pc.VarsTo(nil)
	var vecs mat.Dense
	pc.VectorsTo(&vecs)

	c.ok("제1주성분의 분산이 더 크다", vars[0] > vars[1],
		fmtNum(vars[0])+" > "+fmtNum(vars[1]))
	c.near("제1주성분이 x축과 나란함", 1, math.Abs(vecs.At(0, 0)), 0.02)

	ratio := vars[0] / floats.Sum(vars)
	c.ok("제1주성분의 설명 비율", ratio > 0.9,
		fmtNum(ratio*100)+"%를 방향 하나로 설명")
}

// 부록 "판별 분석: 집단을 구분하는 방향"
func discriminant(c *checker) {
	c.section("판별 분석: 집단을 구분하는 방향")

	// 3장 activation steering의 설정을 모사한다. 정직한 답변일 때의 활성값과
	// 그렇지 않을 때의 활성값을 각각 모아 두 집단으로 둔다.
	src := rand.New(rand.NewPCG(5, 6))
	const n, dim = 300, 2
	honest := make([][]float64, n)
	other := make([][]float64, n)
	for i := 0; i < n; i++ {
		// 두 집단은 첫 번째 축으로 조금 떨어져 있고, 두 번째 축으로 크게 퍼져 있다.
		honest[i] = []float64{1 + 0.5*src.NormFloat64(), 4 * src.NormFloat64()}
		other[i] = []float64{-1 + 0.5*src.NormFloat64(), 4 * src.NormFloat64()}
	}

	// 부록: 가장 단순한 형태는 두 집단의 평균 벡터를 빼서 그 차이를 방향으로 삼는 것이다.
	meanDiff := make([]float64, dim)
	for j := 0; j < dim; j++ {
		meanDiff[j] = stat.Mean(column(honest, j), nil) - stat.Mean(column(other, j), nil)
	}
	floats.Scale(1/floats.Norm(meanDiff, 2), meanDiff)
	c.near("평균 차이 방향의 첫 성분", 1, math.Abs(meanDiff[0]), 0.1)

	// 부록: Fisher의 선형 판별 분석은 집단 안의 흩어짐까지 고려한다.
	// 집단 안 공분산의 역행렬을 평균 차이에 곱하면 그 방향이 나온다.
	fisher := fisherDirection(honest, other)

	sepMean := separation(honest, other, meanDiff)
	sepFisher := separation(honest, other, fisher)
	c.ok("Fisher 방향이 더 잘 분리한다", sepFisher >= sepMean-1e-9,
		"F비 "+fmtNum(sepFisher)+" ≥ "+fmtNum(sepMean))

	// 두 축의 흩어짐이 같다면 두 방향이 일치해야 한다. 위 데이터는 두 번째 축의
	// 흩어짐이 훨씬 크므로, Fisher 방향은 그 축의 기여를 더 깎아 낸다.
	c.ok("Fisher 방향이 넓게 퍼진 축을 덜 쓴다",
		math.Abs(fisher[1]) <= math.Abs(meanDiff[1])+1e-9,
		"두 번째 성분 "+fmtNum(math.Abs(fisher[1]))+" ≤ "+fmtNum(math.Abs(meanDiff[1])))
}

func truncatedSVD(a mat.Matrix, k int) *mat.Dense {
	var svd mat.SVD
	svd.Factorize(a, mat.SVDThin)
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	s := svd.Values(nil)

	r, cN := a.Dims()
	out := mat.NewDense(r, cN, nil)
	for i := 0; i < k; i++ {
		var outer mat.Dense
		outer.Mul(u.ColView(i), v.ColView(i).T())
		outer.Scale(s[i], &outer)
		out.Add(out, &outer)
	}
	return out
}

func relErr(a, b mat.Matrix) float64 {
	var diff mat.Dense
	diff.Sub(a, b)
	return mat.Norm(&diff, 2) / mat.Norm(a, 2)
}

// fisherDirection은 집단 안 공분산의 역행렬에 평균 차이를 곱한 방향을 돌려준다.
func fisherDirection(g1, g2 [][]float64) []float64 {
	dim := len(g1[0])
	sw := mat.NewSymDense(dim, nil)
	for _, g := range [][][]float64{g1, g2} {
		var cov mat.SymDense
		stat.CovarianceMatrix(&cov, toDense(g), nil)
		sw.AddSym(sw, &cov)
	}
	diff := mat.NewVecDense(dim, nil)
	for j := 0; j < dim; j++ {
		diff.SetVec(j, stat.Mean(column(g1, j), nil)-stat.Mean(column(g2, j), nil))
	}
	var w mat.VecDense
	if err := w.SolveVec(sw, diff); err != nil {
		return nil
	}
	out := make([]float64, dim)
	for j := 0; j < dim; j++ {
		out[j] = w.AtVec(j)
	}
	floats.Scale(1/floats.Norm(out, 2), out)
	return out
}

// separation은 주어진 방향으로 사영했을 때 집단 사이 분산과 집단 안 분산의 비율을 잰다.
func separation(g1, g2 [][]float64, dir []float64) float64 {
	p1, p2 := project(g1, dir), project(g2, dir)
	m1, m2 := stat.Mean(p1, nil), stat.Mean(p2, nil)
	within := stat.Variance(p1, nil) + stat.Variance(p2, nil)
	return (m1 - m2) * (m1 - m2) / within
}

func project(g [][]float64, dir []float64) []float64 {
	out := make([]float64, len(g))
	for i, row := range g {
		out[i] = floats.Dot(row, dir)
	}
	return out
}

func column(rows [][]float64, j int) []float64 {
	out := make([]float64, len(rows))
	for i, row := range rows {
		out[i] = row[j]
	}
	return out
}

func toDense(rows [][]float64) *mat.Dense {
	n, d := len(rows), len(rows[0])
	m := mat.NewDense(n, d, nil)
	for i, row := range rows {
		m.SetRow(i, row)
	}
	return m
}

func randVec(src *rand.Rand, n int) []float64 {
	return randSlice(src, n)
}

func randSlice(src *rand.Rand, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = src.NormFloat64()
	}
	return out
}
