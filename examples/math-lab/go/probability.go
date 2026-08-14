package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

// 부록 "확률 분포와 조건부 확률"
func probabilityAndBayes(c *checker) {
	c.section("확률 분포와 조건부 확률")

	// 부록: 확률 분포는 모든 경우에 확률을 배정한 것으로 전체 합이 1이다.
	// LLM의 출력이 어휘 크기만큼의 항목을 가진 이산 확률 분포다.
	logits := []float64{2.0, 1.0, 0.1}
	p := softmax(logits, 1.0)
	c.near("softmax 출력의 합", 1, floats.Sum(p), 1e-12)

	// 부록: 곱의 법칙에 따라 문장 전체의 확률은 조건부 확률들의 곱으로 분해된다.
	// P(t1, t2) = P(t1) * P(t2 | t1)을 결합 분포에서 직접 확인한다.
	// 토큰이 두 개(0, 1)뿐인 결합 분포를 쓴다.
	joint := [2][2]float64{
		{0.10, 0.30}, // P(t1=0, t2=0), P(t1=0, t2=1)
		{0.45, 0.15}, // P(t1=1, t2=0), P(t1=1, t2=1)
	}
	pT1 := joint[1][0] + joint[1][1] // P(t1=1)
	pT2GivenT1 := joint[1][1] / pT1  // P(t2=1 | t1=1)
	c.near("곱의 법칙 P(t1,t2) = P(t1)·P(t2|t1)",
		joint[1][1], pT1*pT2GivenT1, 1e-12)

	// 부록: 베이즈 정리는 P(A|B) = P(B|A) * P(A) / P(B)다.
	// 스팸 필터를 예로 든다. 스팸 비율 20%, 스팸에 "무료"가 들어갈 확률 60%,
	// 정상 메일에 들어갈 확률 5%로 두고 "무료"를 본 뒤 스팸일 확률을 구한다.
	pSpam := 0.20
	pWordGivenSpam := 0.60
	pWordGivenHam := 0.05
	pWord := pWordGivenSpam*pSpam + pWordGivenHam*(1-pSpam) // 전체 확률의 법칙
	pSpamGivenWord := pWordGivenSpam * pSpam / pWord

	c.near("전체 확률의 법칙으로 구한 P(단어)", 0.16, pWord, 1e-12)
	c.near("베이즈 정리로 뒤집은 P(스팸|단어)", 0.75, pSpamGivenWord, 1e-12)
	c.ok("증거를 보고 확률이 갱신됨", pSpamGivenWord > pSpam,
		"사전 "+fmtNum(pSpam)+" → 사후 "+fmtNum(pSpamGivenWord))

	// 부록: 나이브 베이즈는 클래스가 주어지면 단어들이 독립이라고 가정해서
	// P(단어들|클래스)를 단어별 확률의 곱, 실제 계산으로는 로그 확률의 합으로 바꾼다.
	spamWordProbs := []float64{0.60, 0.40, 0.30}
	hamWordProbs := []float64{0.05, 0.10, 0.20}

	prodSpam := pSpam
	for _, q := range spamWordProbs {
		prodSpam *= q
	}
	logSpam := math.Log(pSpam)
	for _, q := range spamWordProbs {
		logSpam += math.Log(q)
	}
	c.near("곱과 로그 합이 같은 값", math.Log(prodSpam), logSpam, 1e-12)

	prodHam := 1 - pSpam
	for _, q := range hamWordProbs {
		prodHam *= q
	}
	c.ok("나이브 베이즈가 스팸으로 분류", prodSpam > prodHam,
		"스팸 점수 "+fmtNum(prodSpam)+" > 정상 점수 "+fmtNum(prodHam))
}

// 부록 "로짓, softmax, temperature"
func softmaxAndTemperature(c *checker) {
	c.section("로짓, softmax, temperature")

	logits := []float64{2.0, 1.0, 0.1}

	// 부록: exp는 [7.39, 2.72, 1.11]이고 합은 11.21이다.
	exps := make([]float64, len(logits))
	for i, v := range logits {
		exps[i] = math.Exp(v)
	}
	c.near("exp(2.0)", 7.39, exps[0], 5e-3)
	c.near("exp(1.0)", 2.72, exps[1], 5e-3)
	c.near("exp(0.1)", 1.11, exps[2], 5e-3)
	c.near("exp들의 합", 11.21, floats.Sum(exps), 5e-3)

	// 부록: softmax는 [0.66, 0.24, 0.10]이고 합은 1.0이다.
	p := softmax(logits, 1.0)
	c.near("softmax 1위 확률", 0.66, p[0], 5e-3)
	c.near("softmax 2위 확률", 0.24, p[1], 5e-3)
	c.near("softmax 3위 확률", 0.10, p[2], 5e-3)
	c.near("softmax 합", 1.0, floats.Sum(p), 1e-12)

	// 부록: 점수가 1 크면 확률은 약 2.7배가 된다.
	c.near("로짓 1 차이의 확률 배율", math.E, p[0]/p[1], 1e-9)

	// 부록: T = 0.5면 [0.86, 0.12, 0.02]로 뾰족해진다.
	cold := softmax(logits, 0.5)
	c.near("T=0.5의 1위 확률", 0.86, cold[0], 5e-3)
	c.near("T=0.5의 2위 확률", 0.12, cold[1], 5e-3)
	c.near("T=0.5의 3위 확률", 0.02, cold[2], 5e-3)

	// 부록: T = 2.0이면 [0.50, 0.30, 0.19]로 평평해진다.
	hot := softmax(logits, 2.0)
	c.near("T=2.0의 1위 확률", 0.50, hot[0], 5e-3)
	c.near("T=2.0의 2위 확률", 0.30, hot[1], 5e-3)
	c.near("T=2.0의 3위 확률", 0.19, hot[2], 5e-3)

	// 부록: T가 0에 가까우면 그리디 선택과 같아지고, T가 크면 균등 분포에 가까워진다.
	greedy := softmax(logits, 0.01)
	c.near("T→0이면 1위에 확률이 몰림", 1, greedy[0], 1e-9)
	flat := softmax(logits, 1000)
	c.near("T→∞이면 균등 분포에 가까워짐", 1.0/3, flat[0], 1e-3)

	// 부록: temperature는 분포의 뾰족함을 조절한다. 엔트로피로 재면 단조 증가한다.
	monotone := true
	prev := -1.0
	for _, t := range []float64{0.25, 0.5, 1.0, 2.0, 4.0} {
		h := stat.Entropy(softmax(logits, t))
		if h < prev {
			monotone = false
		}
		prev = h
	}
	c.ok("T가 커지면 엔트로피가 커진다", monotone, "0.25에서 4.0까지 단조 증가")

	// softmax는 로짓 전체에 같은 수를 더해도 결과가 변하지 않는다. 위 로짓에
	// 1000을 더해도 분포는 그대로여야 한다.
	big := []float64{1002.0, 1001.0, 1000.1}
	naive := math.Exp(big[0]) / (math.Exp(big[0]) + math.Exp(big[1]) + math.Exp(big[2]))
	stable := softmax(big, 1.0)
	c.ok("exp를 그대로 쓰면 값이 넘침", math.IsNaN(naive),
		"exp(1002)가 무한대가 되어 NaN")
	c.near("최댓값을 빼면 원래 분포와 같음", 0.66, stable[0], 5e-3)
	c.near("LogSumExp도 같은 값", math.Log(stable[0]),
		big[0]-floats.LogSumExp(big), 1e-12)
}

// 부록 "로그, 교차 엔트로피, 최대 우도"
func crossEntropySection(c *checker) {
	c.section("로그, 교차 엔트로피, 최대 우도")

	// 부록: log(a*b) = log(a) + log(b)
	a, b := 0.3, 0.017
	c.near("log(a*b) = log a + log b", math.Log(a*b), math.Log(a)+math.Log(b), 1e-12)

	// 부록: 손실 = -log(정답 토큰에 배정된 확률)
	c.near("확률 0.9의 손실", 0.11, -math.Log(0.9), 5e-3)
	c.near("확률 0.5의 손실", 0.69, -math.Log(0.5), 5e-3)
	c.near("확률 0.01의 손실", 4.61, -math.Log(0.01), 5e-3)

	// 부록: 정답에 확률 1을 주면 손실이 0이다.
	c.near("확률 1의 손실", 0, -math.Log(1), 1e-12)

	// 부록: 파라미터가 무작위인 모델은 어휘 27개에 균등하게 1/27을 배정하므로
	// 손실이 -log(1/27) = ln(27) ≈ 3.3에서 시작한다. 5장의 학습 전 손실이다.
	c.near("어휘 27개 균등 분포의 손실", 3.3, -math.Log(1.0/27), 5e-3)
	c.near("-log(1/27) = ln(27)", math.Log(27), -math.Log(1.0/27), 1e-12)

	// 부록: 문장의 확률은 조건부 확률 수백 개의 곱이라 부동소수점으로는
	// 금방 0으로 내려앉는다(언더플로). 그래서 로그 확률의 합으로 계산한다.
	prob := 1.0
	logProb := 0.0
	const tokens = 700
	for i := 0; i < tokens; i++ {
		prob *= 0.3
		logProb += math.Log(0.3)
	}
	c.ok("확률 700개를 곱하면 0으로 언더플로", prob == 0, "곱셈 결과가 정확히 0")
	c.ok("로그 합은 유한한 값을 유지", !math.IsInf(logProb, 0),
		"로그 확률 "+fmtNum(logProb))
	// float64가 표현할 수 있는 가장 작은 양수보다 작아지는 순간 0이 된다.
	c.ok("언더플로 직전까지는 표현됨", math.Pow(0.3, 600) > 0,
		"600개까지는 "+fmt.Sprintf("%.3e", math.Pow(0.3, 600)))

	// 부록: 최대 우도는 학습 데이터에 모델이 배정하는 확률을 최대로 만드는
	// 파라미터를 찾는 것이고, 부호를 뒤집으면 최소화 문제(교차 엔트로피)가 된다.
	// 동전 던지기에서 최대 우도 해가 표본 비율과 같은지 확인한다.
	heads, total := 37, 100
	bestQ, bestLoss := 0.0, math.Inf(1)
	for q := 0.001; q < 1; q += 0.001 {
		loss := -float64(heads)*math.Log(q) - float64(total-heads)*math.Log(1-q)
		if loss < bestLoss {
			bestQ, bestLoss = q, loss
		}
	}
	c.near("최대 우도 해 = 표본 비율", 0.37, bestQ, 1e-3)

	// 교차 엔트로피 손실은 정답 분포와 모델 분포 사이의 값이다.
	// 정답이 한 토큰에 확률 1을 주는 경우 -log(그 토큰의 확률)로 줄어든다.
	target := []float64{0, 1, 0}
	model := []float64{0.2, 0.5, 0.3}
	c.near("정답이 확실할 때의 교차 엔트로피",
		-math.Log(0.5), stat.CrossEntropy(target, model), 1e-12)
}

// 부록 "엔트로피와 perplexity"
func entropyAndPerplexity(c *checker) {
	c.section("엔트로피와 perplexity")

	// 부록: 한 토큰이 확률 1로 정해져 있으면 엔트로피는 0이다.
	c.near("확률 1인 분포의 엔트로피", 0, stat.Entropy([]float64{1, 0, 0}), 1e-12)

	// 부록: V개 후보가 균등하면 ln(V)로 최대가 된다.
	uniform := make([]float64, 27)
	for i := range uniform {
		uniform[i] = 1.0 / 27
	}
	c.near("균등 분포 27개의 엔트로피", math.Log(27), stat.Entropy(uniform), 1e-12)

	// 균등 분포가 정말 최대인지 무작위 분포와 비교한다.
	maxIsUniform := true
	for _, dist := range [][]float64{
		{0.5, 0.3, 0.2}, {0.9, 0.05, 0.05}, {0.4, 0.35, 0.25},
	} {
		if stat.Entropy(dist) > math.Log(3)+1e-12 {
			maxIsUniform = false
		}
	}
	c.ok("균등 분포의 엔트로피가 최대", maxIsUniform, "ln(3)을 넘는 분포 없음")

	// 부록: perplexity = exp(평균 손실)
	c.near("평균 손실 3.3의 perplexity", 27, math.Exp(3.2958), 0.1)
	c.near("평균 손실 2.0의 perplexity", 7.4, math.Exp(2.0), 0.05)

	// 부록: 손실 ln(27)이면 perplexity가 정확히 27이다.
	// "27개 후보를 놓고 균등하게 찍는 정도의 불확실성"이라는 해석이 여기서 나온다.
	c.near("ln(27) 손실의 perplexity", 27, math.Exp(math.Log(27)), 1e-9)
	c.near("균등 분포의 perplexity = 후보 수", 27, math.Exp(stat.Entropy(uniform)), 1e-9)
}

// 부록 "KL 발산: 두 분포 사이의 차이 재기"
func klDivergence(c *checker) {
	c.section("KL 발산: 두 분포 사이의 차이 재기")

	p := []float64{0.5, 0.3, 0.2}
	q := []float64{0.4, 0.4, 0.2}

	// 부록: P와 Q가 같으면 0이고, 다를수록 커지며, 음수가 되지 않는다.
	c.near("같은 분포의 KL", 0, stat.KullbackLeibler(p, p), 1e-12)
	c.ok("다른 분포의 KL은 양수", stat.KullbackLeibler(p, q) > 0,
		fmtNum(stat.KullbackLeibler(p, q)))

	// 부록: 순서를 바꾸면 값이 달라진다(비대칭).
	forward := stat.KullbackLeibler(p, q)
	backward := stat.KullbackLeibler(q, p)
	c.ok("KL은 비대칭", math.Abs(forward-backward) > 1e-6,
		"KL(P||Q) "+fmtNum(forward)+" ≠ KL(Q||P) "+fmtNum(backward))

	// 부록의 관계식: 교차 엔트로피(P, Q) = 엔트로피(P) + KL(P || Q)
	c.near("교차 엔트로피 = 엔트로피 + KL",
		stat.CrossEntropy(p, q), stat.Entropy(p)+stat.KullbackLeibler(p, q), 1e-12)

	// 부록: 학습 데이터의 분포 P는 변하지 않으므로 교차 엔트로피를 줄이는 것은
	// 곧 KL을 줄이는 것이다. 모델 분포를 P 쪽으로 옮기며 두 값이 함께 줄어드는지 본다.
	ceDrops, klDrops := true, true
	prevCE, prevKL := math.Inf(1), math.Inf(1)
	for step := 0; step <= 10; step++ {
		t := float64(step) / 10
		model := make([]float64, len(p))
		for i := range model {
			model[i] = (1-t)*q[i] + t*p[i] // Q에서 P로 조금씩 이동
		}
		ce := stat.CrossEntropy(p, model)
		kl := stat.KullbackLeibler(p, model)
		if ce > prevCE+1e-12 {
			ceDrops = false
		}
		if kl > prevKL+1e-12 {
			klDrops = false
		}
		prevCE, prevKL = ce, kl
	}
	c.ok("교차 엔트로피와 KL이 함께 줄어듦", ceDrops && klDrops,
		"모델 분포를 데이터 분포로 옮기는 동안 둘 다 단조 감소")

	// 1장의 지식 증류에서 student가 teacher를 따라 배울 때의 손실이 이 값이다.
	teacher := softmax([]float64{3.0, 1.0, 0.5}, 1.0)
	student := softmax([]float64{2.0, 1.5, 1.0}, 1.0)
	before := stat.KullbackLeibler(teacher, student)
	closer := softmax([]float64{2.8, 1.2, 0.6}, 1.0)
	after := stat.KullbackLeibler(teacher, closer)
	c.ok("student가 teacher에 가까워지면 KL 감소", after < before,
		fmtNum(after)+" < "+fmtNum(before))
}

// softmax는 로짓을 temperature로 나눈 뒤 확률 분포로 바꾼다.
// 최댓값을 빼는 것은 exp가 넘치지 않게 하려는 표준 기법이다.
func softmax(logits []float64, temperature float64) []float64 {
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = v / temperature
	}
	max := floats.Max(scaled)
	out := make([]float64, len(scaled))
	for i, v := range scaled {
		out[i] = math.Exp(v - max)
	}
	floats.Scale(1/floats.Sum(out), out)
	return out
}
