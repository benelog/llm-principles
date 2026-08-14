package llmprinciples.mathlab;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.FiniteDifferencesDifferentiator;
import org.apache.commons.math3.analysis.differentiation.UnivariateDifferentiableFunction;

/** 부록 "미분과 체인 룰: 학습의 방향을 알아내기" */
public final class Calculus {

    private Calculus() {
    }

    public static void run(Checker c) {
        c.section("미분과 체인 룰: 학습의 방향을 알아내기");

        // Commons Math의 DerivativeStructure는 4장의 autograd와 같은 자동 미분이다.
        // 값과 함께 도함수를 들고 다니면서 연산할 때마다 체인 룰을 적용한다.
        // 첫 번째 인자는 변수 개수, 두 번째는 미분 차수, 세 번째는 변수 번호, 네 번째는 값이다.
        DerivativeStructure x = new DerivativeStructure(1, 1, 0, 3.0);

        // 부록: f(x) = x²이면 x = 3에서의 기울기는 6이다.
        DerivativeStructure square = x.multiply(x);
        c.near("f(x)=x²의 x=3에서 기울기", 6, square.getPartialDerivative(1), 1e-12);
        c.note("자동 미분이라 수치 미분과 달리 오차 없이 정확히 6이 나온다");

        // 부록: x를 0.001만큼 키우면 f는 약 0.006만큼 커진다.
        c.near("x를 0.001 키웠을 때 변화량", 0.006, 3.001 * 3.001 - 3 * 3, 1e-5);

        // 부록: 편미분은 다른 입력을 고정하고 하나만 움직여 구한 기울기이고,
        // 그것을 모두 모은 벡터가 그래디언트다.
        // f(x, y) = x²y + 3y이면 ∂f/∂x = 2xy, ∂f/∂y = x² + 3이다.
        DerivativeStructure vx = new DerivativeStructure(2, 1, 0, 2.0);
        DerivativeStructure vy = new DerivativeStructure(2, 1, 1, 5.0);
        DerivativeStructure f = vx.multiply(vx).multiply(vy).add(vy.multiply(3));
        c.near("편미분 ∂f/∂x (x=2, y=5)", 20, f.getPartialDerivative(1, 0), 1e-12);
        c.near("편미분 ∂f/∂y (x=2, y=5)", 7, f.getPartialDerivative(0, 1), 1e-12);

        // 부록: 체인 룰은 dy/dx = dy/dg * dg/dx다.
        // g(x) = x², f(u) = sin(u)로 두고 x = 1.3에서 확인한다.
        final double x0 = 1.3;
        DerivativeStructure xc = new DerivativeStructure(1, 1, 0, x0);
        double whole = xc.multiply(xc).sin().getPartialDerivative(1); // 합성 함수를 통째로

        double dyDg = Math.cos(x0 * x0); // sin의 도함수를 g(x)에서 평가
        double dgDx = 2 * x0;            // x²의 도함수
        c.near("체인 룰: 통째 미분과 로컬 기울기의 곱", whole, dyDg * dgDx, 1e-12);
        c.note("dy/dx = %.6f, dy/dg × dg/dx = %.6f", whole, dyDg * dgDx);

        // 자동 미분 없이 수치 미분으로도 같은 값이 나오는지 확인한다.
        // Gonum에는 자동 미분이 없어 이 방식만 쓸 수 있다.
        UnivariateFunction composed = v -> Math.sin(v * v);
        UnivariateDifferentiableFunction numeric =
                new FiniteDifferencesDifferentiator(5, 1e-4).differentiate(composed);
        double approx = numeric.value(new DerivativeStructure(1, 1, 0, x0))
                .getPartialDerivative(1);
        c.near("수치 미분도 같은 값", whole, approx, 1e-6);

        // 부록: 경사 하강법은 "파라미터 = 파라미터 - 학습률 * 기울기"다.
        // 손실 (x-3)²을 직접 이 규칙으로 줄여 본다.
        double p = 0.0;
        final double learningRate = 0.1;
        for (int i = 0; i < 200; i++) {
            p -= learningRate * 2 * (p - 3);
        }
        c.near("경사 하강법이 찾은 최소점", 3, p, 1e-6);

        // 기울기를 손으로 적지 않고 자동 미분으로 받아도 결과가 같다.
        double auto = 0.0;
        for (int i = 0; i < 200; i++) {
            DerivativeStructure param = new DerivativeStructure(1, 1, 0, auto);
            DerivativeStructure loss = param.subtract(3).pow(2);
            auto -= learningRate * loss.getPartialDerivative(1);
        }
        c.near("자동 미분으로 돌린 경사 하강법", 3, auto, 1e-6);

        // 부록: 학습률이 너무 크면 최저 부근을 지나쳐 발산한다.
        // (x-3)²의 기울기는 2(x-3)이므로 학습률이 1보다 크면 발산한다.
        double diverged = 0.0;
        for (int i = 0; i < 200; i++) {
            diverged -= 1.05 * 2 * (diverged - 3);
        }
        c.ok("학습률이 너무 크면 발산", Math.abs(diverged - 3) > 1e6,
                "학습률 1.05에서 |x-3| = " + Checker.num(Math.abs(diverged - 3)));

        // 부록: 너무 작으면 학습이 느리다. 같은 반복 횟수에서 얼마나 갔는지 비교한다.
        double slow = 0.0;
        for (int i = 0; i < 200; i++) {
            slow -= 0.001 * 2 * (slow - 3);
        }
        c.ok("학습률이 너무 작으면 느림", Math.abs(slow - 3) > 1,
                "200걸음 뒤에도 " + Checker.num(slow) + "에 머문다");
    }
}
