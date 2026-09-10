import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Value는 스칼라 자동 미분 엔진이다. 값을 계산하면서(forward) 계산 그래프를 기록하고,
 * backward()가 그 그래프를 역순으로 돌며 체인 룰로 기울기를 채운다.
 * 자동 미분은 부동소수점 연산만으로 결정되므로 모든 검사를 정확한 값으로 단정할 수 있다.
 */
class ValueTest {

    @Test
    @DisplayName("연산은 값을 계산하면서 입력 노드와의 연결을 기록한다")
    void operationsRecordTheGraph() {
        Value a = new Value(2);
        Value b = new Value(3);

        Value sum = a.add(b);
        Value product = a.mul(b);

        assertThat(sum.data).isEqualTo(5);
        assertThat(product.data).isEqualTo(6);
        assertThat(sum.children).containsExactly(a, b);
        assertThat(product.children).containsExactly(a, b);
        assertThat(a.children).as("상수 노드는 입력이 없다").isEmpty();
    }

    @Test
    @DisplayName("backward는 곱셈의 기울기를 상대편 입력 값으로, 덧셈의 기울기를 그대로 전달한다")
    void backwardAppliesLocalDerivatives() {
        Value a = new Value(2);
        Value b = new Value(3);
        Value c = new Value(4);

        // y = a * b + c
        Value y = a.mul(b).add(c);
        y.backward();

        assertThat(y.grad).as("출발점의 기울기는 1").isEqualTo(1);
        assertThat(a.grad).as("dy/da = b").isEqualTo(3);
        assertThat(b.grad).as("dy/db = a").isEqualTo(2);
        assertThat(c.grad).as("dy/dc = 1").isEqualTo(1);
    }

    @Test
    @DisplayName("같은 노드가 두 경로에 쓰이면 기울기가 더해진다 (+=인 이유)")
    void gradientsAccumulateAcrossPaths() {
        Value x = new Value(3);

        // y = x * x + x. dy/dx = 2x + 1 = 7
        Value y = x.mul(x).add(x);
        y.backward();

        assertThat(x.grad).isEqualTo(7);
    }

    @Test
    @DisplayName("7장의 체인 룰 예제: y = (2x + 1)²의 x = 1에서 기울기는 12다")
    void chainRuleExampleFromChapter7() {
        Value x = new Value(1);

        Value y = x.mul(new Value(2)).add(new Value(1)).pow(2);
        y.backward();

        assertThat(y.data).isEqualTo(9);
        assertThat(x.grad).isEqualTo(12);
    }

    @Test
    @DisplayName("exp, log, tanh, pow의 기울기는 수치 미분과 일치한다")
    void elementaryDerivativesMatchNumericDifferentiation() {
        assertGradientMatches(0.7, Value::exp, v -> Math.exp(v));
        assertGradientMatches(0.7, Value::log, v -> Math.log(v));
        assertGradientMatches(0.7, Value::tanh, v -> Math.tanh(v));
        assertGradientMatches(0.7, v -> v.pow(-0.5), v -> Math.pow(v, -0.5));
    }

    private static void assertGradientMatches(double at, java.util.function.UnaryOperator<Value> op,
                                              java.util.function.DoubleUnaryOperator f) {
        Value x = new Value(at);
        op.apply(x).backward();

        double h = 1e-6;
        double numeric = (f.applyAsDouble(at + h) - f.applyAsDouble(at - h)) / (2 * h);
        assertThat(x.grad).isCloseTo(numeric, within(1e-6));
    }

    @Test
    @DisplayName("sub와 div는 기본 연산의 조합이므로 기울기도 그 조합으로 나온다")
    void derivedOperationsReuseBasicOnes() {
        Value a = new Value(6);
        Value b = new Value(3);

        Value quotient = a.div(b); // a * b^-1
        quotient.backward();

        assertThat(quotient.data).isEqualTo(2);
        assertThat(a.grad).as("d(a/b)/da = 1/b").isCloseTo(1.0 / 3, within(1e-12));
        assertThat(b.grad).as("d(a/b)/db = -a/b²").isCloseTo(-6.0 / 9, within(1e-12));
    }

    @Test
    @DisplayName("backward는 그래프의 각 노드를 정확히 한 번씩 역순으로 방문한다")
    void backwardVisitsEachNodeOnceInTopologicalOrder() {
        // 마름모 모양 그래프: x → (u, v) → y. u와 v 모두 x에서 나오고 y는 둘을 더한다.
        Value x = new Value(2);
        Value u = x.mul(new Value(3)); // 3x
        Value v = x.mul(x);            // x²
        Value y = u.add(v);            // 3x + x², dy/dx = 3 + 2x = 7

        y.backward();

        // u.backwardFn과 v.backwardFn이 각각 한 번씩만 실행되어야 7이 나온다.
        // 두 번 실행되면 14, 방문 순서가 틀리면 y의 기울기가 전달되기 전에 계산되어 0이 된다.
        assertThat(x.grad).isEqualTo(7);
    }
}
