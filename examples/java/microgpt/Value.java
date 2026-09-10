import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

// tag::value[]
// Value는 스칼라 자동미분 엔진의 기본 단위다.
// 연산을 할 때마다 새 Value가 만들어지고, 입력 노드와의 연결이 기록된다.
// 이 연결을 따라가면 전체 계산 과정이 하나의 그래프가 된다.
public class Value {
    double data;                // 순전파로 계산한 값
    double grad;                // 역전파로 누적되는 기울기
    final List<Value> children; // 이 값을 만든 입력 노드들
    Runnable backwardFn;        // 국소 미분을 입력 노드에 전달하는 람다

    // 상수 값으로 노드를 만든다.
    public Value(double data) {
        this(data, Collections.emptyList());
    }

    private Value(double data, List<Value> children) {
        this.data = data;
        this.children = children;
    }

    // 덧셈. 기울기는 양쪽 입력에 그대로 전달된다.
    public Value add(Value other) {
        Value out = new Value(data + other.data, List.of(this, other));
        out.backwardFn = () -> {
            this.grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }

    // 곱셈. 각 입력의 기울기는 상대편 입력 값에 비례한다.
    public Value mul(Value other) {
        Value out = new Value(data * other.data, List.of(this, other));
        out.backwardFn = () -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }

    // 거듭제곱. 지수 n은 상수로 취급한다. d/dx x^n = n * x^(n-1)
    public Value pow(double n) {
        Value out = new Value(Math.pow(data, n), List.of(this));
        out.backwardFn = () -> this.grad += n * Math.pow(this.data, n - 1) * out.grad;
        return out;
    }

    // 지수 함수. 미분 결과가 출력 값 자신과 같다.
    public Value exp() {
        Value out = new Value(Math.exp(data), List.of(this));
        out.backwardFn = () -> this.grad += out.data * out.grad;
        return out;
    }

    // 자연로그. 교차 엔트로피 손실 계산에 쓰인다.
    public Value log() {
        Value out = new Value(Math.log(data), List.of(this));
        out.backwardFn = () -> this.grad += out.grad / this.data;
        return out;
    }

    // MLP 블록에서 쓰는 비선형 활성 함수.
    public Value tanh() {
        double t = Math.tanh(data);
        Value out = new Value(t, List.of(this));
        out.backwardFn = () -> this.grad += (1 - t * t) * out.grad;
        return out;
    }

    // 위 연산을 조합한 편의 연산들.
    public Value neg() { return mul(new Value(-1)); }
    public Value sub(Value other) { return add(other.neg()); }
    public Value div(Value other) { return mul(other.pow(-1)); }
    // end::value[]

    // tag::backward[]
    // 연산 그래프를 위상 정렬한 뒤 역순으로 체인 룰을 적용한다.
    // 호출한 노드(보통 손실)를 기준으로 모든 입력 노드의 grad가 채워진다.
    public void backward() {
        // 깊이 우선 탐색으로 위상 정렬. 입력이 먼저, 출력이 나중에 쌓인다.
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        buildTopo(this, visited, topo);

        // 손실 자신의 기울기는 1. 역순으로 각 노드의 국소 미분을 전파한다.
        grad = 1;
        for (int i = topo.size() - 1; i >= 0; i--) {
            if (topo.get(i).backwardFn != null) {
                topo.get(i).backwardFn.run();
            }
        }
    }

    private static void buildTopo(Value node, Set<Value> visited, List<Value> topo) {
        if (!visited.add(node)) {
            return;
        }
        for (Value child : node.children) {
            buildTopo(child, visited, topo);
        }
        topo.add(node);
    }
    // end::backward[]
}
