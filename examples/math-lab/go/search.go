package main

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

// 부록 "검색의 통계: TF-IDF와 BM25"
func searchStatistics(c *checker) {
	c.section("검색의 통계: TF-IDF와 BM25")

	// 부록: IDF = log(전체 문서 수 / 그 단어가 나온 문서 수)
	idf := func(total, df float64) float64 { return math.Log(total / df) }

	// 부록: 모든 문서에 나오는 단어는 IDF가 log(1) = 0이 되어 검색에 기여하지 못한다.
	c.near("모든 문서에 나오는 단어의 IDF", 0, idf(1000, 1000), 1e-12)

	// 부록: 드문 단어일수록 값이 커진다.
	rare := idf(1000, 2)
	common := idf(1000, 500)
	c.ok("드문 단어의 IDF가 더 큼", rare > common,
		"df=2에서 "+fmtNum(rare)+", df=500에서 "+fmtNum(common))

	monotone := true
	prev := math.Inf(1)
	for _, df := range []float64{1, 10, 100, 500, 1000} {
		v := idf(1000, df)
		if v > prev {
			monotone = false
		}
		prev = v
	}
	c.ok("df가 커질수록 IDF가 단조 감소", monotone, "흔할수록 벌점")

	// 부록: TF-IDF는 두 빈도의 곱이다.
	tfidf := func(tf, total, df float64) float64 { return tf * idf(total, df) }
	c.near("TF-IDF = TF × IDF", 3*idf(1000, 10), tfidf(3, 1000, 10), 1e-12)
	c.ok("같은 단어라도 많이 나오면 점수가 큼",
		tfidf(5, 1000, 10) > tfidf(2, 1000, 10), "TF에 비례")

	// 부록: BM25는 단어 빈도의 기여에 상한을 두어 포화시키고 문서 길이로 보정한다.
	// 같은 단어가 20번 나온 문서가 10번 나온 문서보다 2배 관련 있다고 보지 않는다.
	const k1, bParam, avgLen = 1.2, 0.75, 100.0
	bm25TF := func(tf, docLen float64) float64 {
		return tf * (k1 + 1) / (tf + k1*(1-bParam+bParam*docLen/avgLen))
	}

	tf10 := bm25TF(10, avgLen)
	tf20 := bm25TF(20, avgLen)
	c.ok("BM25는 빈도가 두 배여도 점수가 두 배가 아님", tf20 < 2*tf10,
		"tf=10에서 "+fmtNum(tf10)+", tf=20에서 "+fmtNum(tf20))
	c.note("TF-IDF의 빈도 항이라면 10에서 20으로 정확히 두 배가 된다")

	// 빈도를 아무리 올려도 k1+1을 넘지 못한다는 것이 포화의 내용이다.
	c.ok("BM25 빈도 항의 상한", bm25TF(1e9, avgLen) < k1+1,
		"상한 "+fmtNum(k1+1)+"에 수렴")

	// 부록: 문서 길이로 점수를 보정한다. 같은 빈도라면 긴 문서가 손해를 본다.
	shortDoc := bm25TF(5, 50)
	longDoc := bm25TF(5, 400)
	c.ok("같은 빈도면 긴 문서의 점수가 낮음", longDoc < shortDoc,
		"길이 50에서 "+fmtNum(shortDoc)+", 길이 400에서 "+fmtNum(longDoc))
}

// 부록 "t-검정과 분산 분석: 지표의 차이가 진짜인지 확인하기"
func significanceTests(c *checker) {
	c.section("t-검정과 분산 분석: 지표의 차이가 진짜인지 확인하기")

	// 부록: 정확도 80% 근처, 문항 100개면 표준 오차가 약 4%p다.
	p, n := 0.8, 100.0
	se := math.Sqrt(p * (1 - p) / n)
	c.near("정확도 80%, 문항 100개의 표준 오차", 0.04, se, 1e-12)

	// Gonum의 StdErr로도 같은 값이 나오는지 확인한다.
	// 0과 1로 된 채점 결과의 표준편차는 sqrt(p(1-p))이다.
	c.near("StdErr로 계산한 표준 오차", se,
		stat.StdErr(math.Sqrt(p*(1-p)), n), 1e-12)

	// 부록: 표준 오차를 절반으로 줄이려면 문항이 400개 필요하다.
	// 정밀도를 두 배로 만들려면 표본이 네 배 필요하다는 제곱근의 법칙이다.
	se400 := math.Sqrt(p * (1 - p) / 400)
	c.near("문항 400개의 표준 오차", se/2, se400, 1e-12)

	// 부록: 79%와 83%의 차이는 우연의 범위를 벗어나지 못한다.
	// 두 비율의 차이에 대한 검정을 직접 수행한다.
	diff := 0.83 - 0.79
	seDiff := math.Sqrt(0.79*0.21/100 + 0.83*0.17/100)
	z := diff / seDiff
	pValue := 2 * (1 - distuv.Normal{Mu: 0, Sigma: 1}.CDF(math.Abs(z)))
	c.ok("79% vs 83%는 유의하지 않음", pValue > 0.05,
		"p값 "+fmtNum(pValue))
	c.note("z = %.3f로, 유의 판정선인 1.96에 못 미친다", z)

	// 부록: 대응표본 t-검정은 질문별 점수 차이를 먼저 구하고 그 평균이
	// 0에서 벗어났는지 본다. 질문 난이도가 만드는 흔들림이 상쇄되므로 더 민감하다.
	src := rand.New(rand.NewPCG(19, 20))
	const items = 100
	before := make([]float64, items)
	after := make([]float64, items)
	for i := 0; i < items; i++ {
		difficulty := 6.0 * src.NormFloat64() // 질문마다 크게 다른 난이도
		before[i] = 70 + difficulty + 0.5*src.NormFloat64()
		after[i] = before[i] + 1.0 + 0.5*src.NormFloat64() // 일정하게 1점 개선
	}

	pairedP := pairedTTest(before, after)
	unpairedP := welchTTest(before, after)
	c.ok("대응표본 검정이 더 민감함", pairedP < unpairedP,
		"대응 p="+fmtNum(pairedP)+" < 독립 p="+fmtNum(unpairedP))
	c.ok("대응표본 검정은 개선을 잡아냄", pairedP < 0.05, "p값 "+fmtNum(pairedP))
	c.ok("독립표본 검정은 같은 개선을 놓침", unpairedP > 0.05, "p값 "+fmtNum(unpairedP))
	c.note("질문 난이도의 흔들림이 두 구성에 똑같이 들어 있어 대응표본에서는 상쇄된다")

	// 부록: 분산 분석은 집단 사이의 분산과 집단 안의 분산의 비율(F비)로
	// "차이가 하나라도 있는가"를 한 번에 검정한다.
	groupA := make([]float64, 40)
	groupB := make([]float64, 40)
	groupC := make([]float64, 40)
	for i := range groupA {
		groupA[i] = 70 + 3*src.NormFloat64()
		groupB[i] = 70 + 3*src.NormFloat64()
		groupC[i] = 76 + 3*src.NormFloat64() // 이 집단만 다르다
	}
	fStat, fp := oneWayANOVA([][]float64{groupA, groupB, groupC})
	c.ok("차이가 있는 세 집단에서 유의한 F비", fp < 0.01,
		"F="+fmtNum(fStat)+", p="+fmtNum(fp))

	// 세 집단이 모두 같은 분포라면 유의 판정이 나오는 비율이 유의수준 5% 근처여야 한다.
	// 한 번만 돌려서는 알 수 없으므로 같은 실험을 여러 번 반복해서 비율을 잰다.
	const anovaRuns = 2000
	falseAlarm := 0
	for r := 0; r < anovaRuns; r++ {
		same := make([][]float64, 3)
		for gi := range same {
			same[gi] = make([]float64, 40)
			for i := range same[gi] {
				same[gi][i] = 70 + 3*src.NormFloat64()
			}
		}
		if _, p := oneWayANOVA(same); p < 0.05 {
			falseAlarm++
		}
	}
	anovaRate := float64(falseAlarm) / anovaRuns
	c.near("차이 없는 세 집단의 위양성률 = 유의수준", 0.05, anovaRate, 0.015)

	// 부록: 쌍마다 t-검정을 반복하면 우연히 유의해 보이는 결과가 나올 확률이 누적된다.
	// 차이가 없는 집단 여러 개로 실험을 반복해 위양성률을 재 본다.
	falsePositives := 0
	const runs = 2000
	for r := 0; r < runs; r++ {
		groups := make([][]float64, 4)
		for gi := range groups {
			groups[gi] = make([]float64, 30)
			for i := range groups[gi] {
				groups[gi][i] = 70 + 3*src.NormFloat64()
			}
		}
		flagged := false
		for i := 0; i < len(groups); i++ {
			for j := i + 1; j < len(groups); j++ {
				if welchTTest(groups[i], groups[j]) < 0.05 {
					flagged = true
				}
			}
		}
		if flagged {
			falsePositives++
		}
	}
	rate := float64(falsePositives) / runs
	c.ok("쌍별 반복 검정의 위양성률이 5%를 넘음", rate > 0.05,
		"집단 4개에서 "+fmtNum(rate*100)+"%")
	c.note("검정 6번을 반복하면 이론값은 1-0.95^6 = %.1f%%다", (1-math.Pow(0.95, 6))*100)
}

// pairedTTest는 대응표본 t-검정의 양측 p값을 돌려준다.
func pairedTTest(a, b []float64) float64 {
	diffs := make([]float64, len(a))
	for i := range a {
		diffs[i] = b[i] - a[i]
	}
	n := float64(len(diffs))
	mean, std := stat.MeanStdDev(diffs, nil)
	t := mean / (std / math.Sqrt(n))
	return twoSided(t, n-1)
}

// welchTTest는 분산이 다를 수 있는 두 독립 표본의 t-검정 p값을 돌려준다.
func welchTTest(a, b []float64) float64 {
	na, nb := float64(len(a)), float64(len(b))
	ma, sa := stat.MeanStdDev(a, nil)
	mb, sb := stat.MeanStdDev(b, nil)
	va, vb := sa*sa/na, sb*sb/nb
	t := (mb - ma) / math.Sqrt(va+vb)
	df := (va + vb) * (va + vb) / (va*va/(na-1) + vb*vb/(nb-1))
	return twoSided(t, df)
}

func twoSided(t, df float64) float64 {
	dist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: df}
	return 2 * (1 - dist.CDF(math.Abs(t)))
}

// oneWayANOVA는 일원 분산 분석의 F비와 p값을 돌려준다.
func oneWayANOVA(groups [][]float64) (float64, float64) {
	var all []float64
	for _, g := range groups {
		all = append(all, g...)
	}
	grand := stat.Mean(all, nil)

	between, within := 0.0, 0.0
	for _, g := range groups {
		m := stat.Mean(g, nil)
		between += float64(len(g)) * (m - grand) * (m - grand)
		for _, v := range g {
			within += (v - m) * (v - m)
		}
	}
	dfBetween := float64(len(groups) - 1)
	dfWithin := float64(len(all) - len(groups))
	f := (between / dfBetween) / (within / dfWithin)
	dist := distuv.F{D1: dfBetween, D2: dfWithin}
	return f, 1 - dist.CDF(f)
}
