package main

import "math"

// tag::value[]
// Value는 스칼라 자동미분 엔진의 기본 단위다.
// 연산을 할 때마다 새 Value가 만들어지고, 입력 노드와의 연결이 기록된다.
// 이 연결을 따라가면 전체 계산 과정이 하나의 그래프가 된다.
type Value struct {
	Data     float64  // 순전파로 계산한 값
	Grad     float64  // 역전파로 누적되는 기울기
	children []*Value // 이 값을 만든 입력 노드들
	backward func()   // 국소 미분을 입력 노드에 전달하는 클로저
}

// NewValue는 상수 값으로 노드를 만든다.
func NewValue(data float64) *Value { return &Value{Data: data} }

// Add는 덧셈. 기울기는 양쪽 입력에 그대로 전달된다.
func (v *Value) Add(other *Value) *Value {
	out := &Value{Data: v.Data + other.Data, children: []*Value{v, other}}
	out.backward = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

// Mul은 곱셈. 각 입력의 기울기는 상대편 입력 값에 비례한다.
func (v *Value) Mul(other *Value) *Value {
	out := &Value{Data: v.Data * other.Data, children: []*Value{v, other}}
	out.backward = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

// Pow는 거듭제곱. 지수 n은 상수로 취급한다. d/dx x^n = n * x^(n-1)
func (v *Value) Pow(n float64) *Value {
	out := &Value{Data: math.Pow(v.Data, n), children: []*Value{v}}
	out.backward = func() {
		v.Grad += n * math.Pow(v.Data, n-1) * out.Grad
	}
	return out
}

// Exp는 지수 함수. 미분 결과가 출력 값 자신과 같다.
func (v *Value) Exp() *Value {
	out := &Value{Data: math.Exp(v.Data), children: []*Value{v}}
	out.backward = func() {
		v.Grad += out.Data * out.Grad
	}
	return out
}

// Log는 자연로그. 교차 엔트로피 손실 계산에 쓰인다.
func (v *Value) Log() *Value {
	out := &Value{Data: math.Log(v.Data), children: []*Value{v}}
	out.backward = func() {
		v.Grad += out.Grad / v.Data
	}
	return out
}

// Tanh는 MLP 블록에서 쓰는 비선형 활성 함수.
func (v *Value) Tanh() *Value {
	t := math.Tanh(v.Data)
	out := &Value{Data: t, children: []*Value{v}}
	out.backward = func() {
		v.Grad += (1 - t*t) * out.Grad
	}
	return out
}

// 위 연산을 조합한 편의 연산들.
func (v *Value) Neg() *Value             { return v.Mul(NewValue(-1)) }
func (v *Value) Sub(other *Value) *Value { return v.Add(other.Neg()) }
func (v *Value) Div(other *Value) *Value { return v.Mul(other.Pow(-1)) }

// end::value[]

// tag::backward[]
// Backward는 연산 그래프를 위상 정렬한 뒤 역순으로 체인 룰을 적용한다.
// 호출한 노드(보통 손실)를 기준으로 모든 입력 노드의 Grad가 채워진다.
func (v *Value) Backward() {
	// 깊이 우선 탐색으로 위상 정렬. 입력이 먼저, 출력이 나중에 쌓인다.
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)
	var build func(node *Value)
	build = func(node *Value) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.children {
			build(child)
		}
		topo = append(topo, node)
	}
	build(v)

	// 손실 자신의 기울기는 1. 역순으로 각 노드의 국소 미분을 전파한다.
	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

// end::backward[]
