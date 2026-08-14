package main

import (
	"math"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/optimize"
)

// 부록 "미분과 체인 룰: 학습의 방향을 알아내기"
func calculus(c *checker) {
	c.section("미분과 체인 룰: 학습의 방향을 알아내기")

	// 부록: f(x) = x²이면 x = 3에서의 기울기는 6이다.
	square := func(x float64) float64 { return x * x }
	slope := fd.Derivative(square, 3, nil)
	c.near("f(x)=x²의 x=3에서 기울기", 6, slope, 1e-6)

	// 부록: x를 0.001만큼 키우면 f는 약 0.006만큼 커진다.
	c.near("x를 0.001 키웠을 때 변화량", 0.006, square(3.001)-square(3), 1e-5)

	// 부록: 편미분은 다른 입력을 고정하고 하나만 움직여 구한 기울기이고,
	// 그것을 모두 모은 벡터가 그래디언트다.
	// f(x, y) = x²y + 3y이면 ∂f/∂x = 2xy, ∂f/∂y = x² + 3이다.
	f := func(v []float64) float64 { return v[0]*v[0]*v[1] + 3*v[1] }
	grad := fd.Gradient(nil, f, []float64{2, 5}, nil)
	c.near("편미분 ∂f/∂x (x=2, y=5)", 20, grad[0], 1e-5)
	c.near("편미분 ∂f/∂y (x=2, y=5)", 7, grad[1], 1e-5)

	// 부록: 체인 룰은 dy/dx = dy/dg * dg/dx다.
	// g(x) = x², f(u) = sin(u)로 두고 x = 1.3에서 확인한다.
	g := func(x float64) float64 { return x * x }
	outer := math.Sin
	composed := func(x float64) float64 { return outer(g(x)) }

	const x0 = 1.3
	whole := fd.Derivative(composed, x0, nil)   // 합성 함수를 통째로 미분
	local := fd.Derivative(outer, g(x0), nil) * // dy/dg
		fd.Derivative(g, x0, nil) // dg/dx
	c.near("체인 룰: 통째 미분과 로컬 기울기의 곱", whole, local, 1e-6)
	c.note("dy/dx = %.6f, dy/dg × dg/dx = %.6f", whole, local)

	// 부록: 경사 하강법은 "파라미터 = 파라미터 - 학습률 * 기울기"다.
	// 손실 (x-3)²을 직접 이 규칙으로 줄여 본다.
	x := 0.0
	const lr = 0.1
	for i := 0; i < 200; i++ {
		x -= lr * 2 * (x - 3)
	}
	c.near("경사 하강법이 찾은 최소점", 3, x, 1e-6)

	// 부록: 학습률이 너무 크면 최저 부근을 지나쳐 발산한다.
	// (x-3)²의 기울기는 2(x-3)이므로 학습률이 1보다 크면 발산한다.
	diverged := 0.0
	for i := 0; i < 200; i++ {
		diverged -= 1.05 * 2 * (diverged - 3)
	}
	c.ok("학습률이 너무 크면 발산", math.Abs(diverged-3) > 1e6,
		"학습률 1.05에서 |x-3| = "+fmtNum(math.Abs(diverged-3)))

	// 같은 최소화를 Gonum의 최적화기로도 확인한다. 규칙은 같고 보폭 조절만 다르다.
	problem := optimize.Problem{
		Func: func(v []float64) float64 { return (v[0] - 3) * (v[0] - 3) },
		Grad: func(dst, v []float64) { dst[0] = 2 * (v[0] - 3) },
	}
	res, err := optimize.Minimize(problem, []float64{0}, nil, &optimize.GradientDescent{})
	if err != nil {
		c.ok("Gonum 경사 하강법", false, err.Error())
		return
	}
	c.near("Gonum GradientDescent의 해", 3, res.X[0], 1e-6)
}
