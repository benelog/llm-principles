package main

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// 부록 "선형 회귀와 로지스틱 회귀: 가장 작은 신경망"
func regression(c *checker) {
	c.section("선형 회귀와 로지스틱 회귀: 가장 작은 신경망")

	// 부록: 선형 회귀는 데이터에 가장 잘 맞는 직선 y = ax + b를 찾는다.
	// 기울기 2.5, 절편 -1인 직선에 잡음을 얹고 계수를 되찾아 본다.
	src := rand.New(rand.NewPCG(15, 16))
	const n = 300
	xs := make([]float64, n)
	ys := make([]float64, n)
	for i := 0; i < n; i++ {
		xs[i] = src.Float64() * 10
		ys[i] = 2.5*xs[i] - 1 + 0.3*src.NormFloat64()
	}
	intercept, slope := stat.LinearRegression(xs, ys, nil, false)
	c.near("회귀가 되찾은 기울기 a", 2.5, slope, 0.05)
	c.near("회귀가 되찾은 절편 b", -1, intercept, 0.1)

	// 부록: "가장 잘 맞는"의 기준은 오차 제곱합을 최소로 만드는 것이다(최소제곱법).
	// 회귀가 준 계수 주변을 흔들어 보면 오차 제곱합이 늘어나야 한다.
	base := sse(xs, ys, slope, intercept)
	worse := true
	for _, d := range []float64{-0.05, -0.01, 0.01, 0.05} {
		if sse(xs, ys, slope+d, intercept) < base {
			worse = false
		}
	}
	c.ok("계수를 흔들면 오차 제곱합이 늘어남", worse, "최소제곱 해가 맞음")

	// 부록: 시그모이드는 선택지가 두 개일 때의 softmax와 같은 함수다.
	z := 1.3
	sig := 1 / (1 + math.Exp(-z))
	two := softmax([]float64{z, 0}, 1.0)
	c.near("시그모이드 = 2항 softmax", sig, two[0], 1e-12)

	// 부록: 로지스틱 회귀의 손실(로그 손실)은 교차 엔트로피와 같은 식이다.
	label := 1.0
	logLoss := -(label*math.Log(sig) + (1-label)*math.Log(1-sig))
	c.near("로그 손실 = 교차 엔트로피",
		stat.CrossEntropy([]float64{1, 0}, []float64{sig, 1 - sig}), logLoss, 1e-12)

	// 부록: RLHF의 보상 모델은 "A가 B보다 선호될 확률 = sigmoid(A점수 - B점수)"라는
	// Bradley-Terry 모형으로 학습된다. 점수 차이가 클수록 확률이 1에 가까워진다.
	pref := func(diff float64) float64 { return 1 / (1 + math.Exp(-diff)) }
	c.near("점수가 같으면 선호 확률 0.5", 0.5, pref(0), 1e-12)
	c.ok("점수 차이가 커지면 확률이 1에 접근",
		pref(5) > 0.99 && pref(1) > pref(0.5),
		"diff=5에서 "+fmtNum(pref(5)))

	// 부록: 다중공선성은 입력 변수들끼리 강하게 상관되어 있을 때,
	// 예측은 멀쩡한데 개별 계수의 추정이 불안정해지는 현상이다.
	// 거의 같은 두 변수로 회귀를 돌려 계수와 예측을 각각 흔들어 본다.
	x1 := make([]float64, n)
	x2 := make([]float64, n)
	target := make([]float64, n)
	for i := 0; i < n; i++ {
		x1[i] = src.NormFloat64()
		x2[i] = x1[i] + 0.15*src.NormFloat64() // x1과 거의 같은 변수
		target[i] = 3*x1[i] + 2*x2[i] + 0.1*src.NormFloat64()
	}
	corr := stat.Correlation(x1, x2, nil)
	c.ok("두 입력이 강하게 상관됨", corr > 0.98, "상관계수 "+fmtNum(corr))

	coefA := fitTwo(x1, x2, target)
	// 표본의 절반만 써서 다시 적합하면 계수가 크게 흔들린다.
	coefB := fitTwo(x1[:n/2], x2[:n/2], target[:n/2])
	coefShift := math.Abs(coefA[0]-coefB[0]) + math.Abs(coefA[1]-coefB[1])

	// 예측은 두 계수의 합이 결정하므로 훨씬 안정적이다.
	sumShift := math.Abs((coefA[0] + coefA[1]) - (coefB[0] + coefB[1]))
	c.ok("계수는 흔들려도 예측은 안정적", sumShift < coefShift,
		"계수 변화 "+fmtNum(coefShift)+" vs 계수 합 변화 "+fmtNum(sumShift))
	c.note("표본을 바꾸면 계수가 [%.2f %.2f] → [%.2f %.2f]로 이동한다",
		coefA[0], coefA[1], coefB[0], coefB[1])

	// 분산 팽창 계수(VIF)는 이 불안정성을 재는 표준 지표다.
	// 한 변수를 나머지로 회귀했을 때의 결정계수 R²로 1/(1-R²)을 구한다.
	r2 := corr * corr
	vif := 1 / (1 - r2)
	c.ok("VIF가 경험적 기준 10을 넘음", vif > 10, "VIF "+fmtNum(vif))
}

// 부록 "마르코프 체인: 상태가 미래를 결정한다"
func markovChain(c *checker) {
	c.section("마르코프 체인: 상태가 미래를 결정한다")

	// 상태 세 개의 전이 확률표를 만든다. n-gram 언어 모델이 말뭉치에서 센 표와
	// 같은 물건이다. 각 행의 합은 1이어야 한다.
	p := mat.NewDense(3, 3, []float64{
		0.7, 0.2, 0.1,
		0.3, 0.4, 0.3,
		0.2, 0.3, 0.5,
	})
	rowsSumToOne := true
	for i := 0; i < 3; i++ {
		if math.Abs(floats.Sum(mat.Row(nil, i, p))-1) > 1e-12 {
			rowsSumToOne = false
		}
	}
	c.ok("전이 확률표의 각 행 합이 1", rowsSumToOne, "확률 분포의 조건")

	// 부록: 확률표만 있으면 과정 전체를 전개할 수 있다.
	// 분포에 전이 행렬을 거듭 곱하면 정상 분포로 수렴한다.
	dist := mat.NewVecDense(3, []float64{1, 0, 0})
	for i := 0; i < 200; i++ {
		var next mat.VecDense
		next.MulVec(p.T(), dist)
		dist = &next
	}
	stationary := []float64{dist.AtVec(0), dist.AtVec(1), dist.AtVec(2)}
	c.near("정상 분포의 합", 1, floats.Sum(stationary), 1e-9)

	// 정상 분포는 자기 자신으로 되돌아온다. πP = π다.
	var again mat.VecDense
	again.MulVec(p.T(), dist)
	same := math.Abs(again.AtVec(0)-dist.AtVec(0)) +
		math.Abs(again.AtVec(1)-dist.AtVec(1)) +
		math.Abs(again.AtVec(2)-dist.AtVec(2))
	c.ok("정상 분포는 전이해도 그대로", same < 1e-9, "πP = π")
	c.note("정상 분포 [%.4f %.4f %.4f]", stationary[0], stationary[1], stationary[2])

	// 부록: 다음에 일어날 일이 현재 상태에만 의존하고 거기까지 온 경로에는
	// 의존하지 않는다. 서로 다른 출발점에서 시작해도 같은 정상 분포로 간다.
	other := mat.NewVecDense(3, []float64{0, 0, 1})
	for i := 0; i < 200; i++ {
		var next mat.VecDense
		next.MulVec(p.T(), other)
		other = &next
	}
	gap := math.Abs(other.AtVec(0)-dist.AtVec(0)) +
		math.Abs(other.AtVec(1)-dist.AtVec(1)) +
		math.Abs(other.AtVec(2)-dist.AtVec(2))
	c.ok("출발점이 달라도 같은 곳으로 수렴", gap < 1e-9,
		"경로가 아니라 상태만 남는다")

	// 6장의 KV 캐시가 이 체인의 상태에 해당한다는 관찰을 행렬 거듭제곱으로 확인한다.
	// n단계 전이 확률은 전이 행렬의 n제곱이며, 중간 경로를 기억할 필요가 없다.
	var p2, pTwice mat.Dense
	p2.Pow(p, 2)
	pTwice.Mul(p, p)
	c.near("2단계 전이 = 전이 행렬의 제곱",
		0, mat.Norm(matSub(&p2, &pTwice), 2), 1e-12)
}

// 부록 "벨만 방정식: 강화학습의 뼈대"
func bellman(c *checker) {
	c.section("벨만 방정식: 강화학습의 뼈대")

	// 부록: 매 걸음 보상 1을 받고 할인율이 γ면, 가치는 등비급수의 합 1/(1-γ)다.
	const gamma = 0.9
	c.near("등비급수의 합 1/(1-γ)", 10, 1/(1-gamma), 1e-12)

	// 유한 항까지 더해 가며 그 값으로 수렴하는지 본다.
	partial := 0.0
	for k := 0; k < 300; k++ {
		partial += math.Pow(gamma, float64(k))
	}
	c.near("보상 1이 계속될 때의 가치", 1/(1-gamma), partial, 1e-9)

	// 부록: 할인율은 무한히 이어지는 보상의 합이 발산하지 않게 만드는 장치다.
	// γ가 1이면 같은 합이 발산한다.
	noDiscount := 0.0
	for k := 0; k < 300; k++ {
		noDiscount += 1.0
	}
	c.ok("γ=1이면 합이 커지기만 함", noDiscount == 300, "300걸음에서 이미 300")

	// 부록: V(s) = E[r + γ * V(s')]. 상태 세 개짜리 마르코프 보상 과정에서
	// 가치 반복으로 구한 해가 선형 방정식의 해와 같은지 확인한다.
	trans := mat.NewDense(3, 3, []float64{
		0.5, 0.5, 0.0,
		0.0, 0.5, 0.5,
		0.5, 0.0, 0.5,
	})
	rewards := []float64{1, 0, 2}

	// 가치 반복: V ← r + γPV를 반복한다.
	v := mat.NewVecDense(3, []float64{0, 0, 0})
	for i := 0; i < 500; i++ {
		var next mat.VecDense
		next.MulVec(trans, v)
		next.ScaleVec(gamma, &next)
		next.AddVec(&next, mat.NewVecDense(3, rewards))
		v = &next
	}

	// 닫힌 해: (I - γP)V = r
	eye := mat.NewDiagDense(3, []float64{1, 1, 1})
	var m mat.Dense
	m.Scale(gamma, trans)
	m.Sub(eye, &m)
	var solved mat.VecDense
	if err := solved.SolveVec(&m, mat.NewVecDense(3, rewards)); err != nil {
		c.ok("벨만 방정식 풀이", false, err.Error())
		return
	}
	diff := 0.0
	for i := 0; i < 3; i++ {
		diff += math.Abs(v.AtVec(i) - solved.AtVec(i))
	}
	c.ok("가치 반복의 해 = 선형 방정식의 해", diff < 1e-9,
		"두 방법의 차이 "+fmtNum(diff))
	c.note("가치 함수 V = [%.3f %.3f %.3f]", v.AtVec(0), v.AtVec(1), v.AtVec(2))

	// 벨만 방정식이 실제로 성립하는지 한 상태에서 직접 확인한다.
	var expected mat.VecDense
	expected.MulVec(trans, v)
	lhs := v.AtVec(0)
	rhs := rewards[0] + gamma*expected.AtVec(0)
	c.near("V(s) = r + γ·E[V(s')]", lhs, rhs, 1e-9)

	// 부록: 먼 미래의 보상은 γ, γ², γ³처럼 할인된다.
	c.near("10걸음 뒤 보상의 할인율", math.Pow(gamma, 10), 0.34867844, 1e-6)
}

// 부록 "스케일링 법칙: 손실을 예측하는 멱법칙"
func scalingLaw(c *checker) {
	c.section("스케일링 법칙: 손실을 예측하는 멱법칙")

	// 부록: 멱법칙 y = a * x^(-b)는 양변에 로그를 취하면
	// log y = log a - b * log x가 되어 로그-로그 그래프에서 직선이 된다.
	const a, b = 12.0, 0.35
	src := rand.New(rand.NewPCG(17, 18))

	var logX, logY []float64
	for _, x := range []float64{1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9} {
		y := a * math.Pow(x, -b) * (1 + 0.01*src.NormFloat64())
		logX = append(logX, math.Log(x))
		logY = append(logY, math.Log(y))
	}
	logA, negB := stat.LinearRegression(logX, logY, nil, false)
	c.near("로그-로그 회귀가 되찾은 지수 b", b, -negB, 0.01)
	c.near("로그-로그 회귀가 되찾은 계수 a", a, math.Exp(logA), 0.5)

	// 부록: 작은 모델 여러 개로 계수를 추정하면 훨씬 큰 모델의 손실을 외삽할 수 있다.
	// 위에서 쓴 표본의 최대 크기보다 100배 큰 규모를 예측하고 참값과 비교한다.
	bigN := 1e11
	predicted := math.Exp(logA + negB*math.Log(bigN))
	actual := a * math.Pow(bigN, -b)
	relative := math.Abs(predicted-actual) / actual
	c.ok("100배 큰 규모의 손실 외삽", relative < 0.05,
		"상대 오차 "+fmtNum(relative*100)+"%")
	c.note("예측 %.5f, 참값 %.5f", predicted, actual)

	// 부록: 손실(N, D) = E + A/N^α + B/D^β. E는 아무리 키워도 남는 손실의 바닥이다.
	loss := func(n, d float64) float64 {
		return 1.69 + 406.4/math.Pow(n, 0.34) + 410.7/math.Pow(d, 0.28)
	}
	c.ok("N과 D를 키우면 손실이 줄어듦",
		loss(1e10, 1e11) < loss(1e9, 1e10), "두 항이 모두 감소")
	c.near("N, D를 무한히 키운 극한이 바닥 E", 1.69, loss(1e30, 1e30), 1e-3)
	c.ok("바닥 아래로는 내려가지 않음", loss(1e12, 1e13) > 1.69,
		"손실 "+fmtNum(loss(1e12, 1e13)))

	// 부록: Chinchilla의 결론은 파라미터 1개당 약 20토큰이라는 균형점이었다.
	// 20B 파라미터 모델이면 400B 토큰이라는 계산이 나온다.
	params := 20e9
	c.near("20B 파라미터에 맞는 토큰 수", 400e9, params*20, 1)
	c.note("파라미터 20B면 학습 토큰 400B가 균형점이라는 계산이다")
}

func sse(xs, ys []float64, slope, intercept float64) float64 {
	total := 0.0
	for i := range xs {
		r := ys[i] - (slope*xs[i] + intercept)
		total += r * r
	}
	return total
}

// fitTwo는 절편 없는 2변수 최소제곱 해를 정규 방정식으로 구한다.
func fitTwo(x1, x2, y []float64) []float64 {
	n := len(x1)
	x := mat.NewDense(n, 2, nil)
	for i := 0; i < n; i++ {
		x.Set(i, 0, x1[i])
		x.Set(i, 1, x2[i])
	}
	var qr mat.QR
	qr.Factorize(x)
	var beta mat.Dense
	if err := qr.SolveTo(&beta, false, mat.NewDense(n, 1, y)); err != nil {
		return []float64{math.NaN(), math.NaN()}
	}
	return []float64{beta.At(0, 0), beta.At(1, 0)}
}

func matSub(a, b mat.Matrix) mat.Matrix {
	var out mat.Dense
	out.Sub(a, b)
	return &out
}
