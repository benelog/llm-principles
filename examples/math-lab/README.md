# math-lab: 수학·통계 부록 검산 실습

"수학·통계 기초" 부록(`book/appendix-math-stats.adoc`)이 본문에 적어 둔 값들을 수치 계산 라이브러리로 다시 계산해서 맞는지 확인하는 실습 코드다. Go와 Java 두 벌이 있고, 같은 내용을 각 언어의 대표 라이브러리로 구현했다.

| | Go | Java |
|---|---|---|
| 라이브러리 | [Gonum](https://gonum.org) | [Apache Commons Math 3](https://commons.apache.org/proper/commons-math/) |
| 빌드 | Go 모듈 | Gradle |
| 실행 | `cd go && go run .` | `cd java && ./gradlew run` |
| 검산 항목 | 157건 | 159건 |

`examples/go/`의 다른 예제들은 표준 라이브러리만 쓴다는 제약이 있으므로, 외부 라이브러리를 쓰는 이 실습은 별도 디렉터리에 두었다.

## 읽는 방법

부록의 절 순서를 그대로 따른다. 부록을 읽다가 확인하고 싶은 절이 나오면 같은 이름의 함수를 열어 보면 된다.

| 부록 절 | Go | Java |
|---|---|---|
| 벡터와 행렬 | `linalg.go` | `LinearAlgebra.java` |
| 랭크와 주성분 분석 | `linalg.go` | `LinearAlgebra.java` |
| 판별 분석 | `linalg.go` | `LinearAlgebra.java` |
| 미분과 체인 룰 | `calculus.go` | `Calculus.java` |
| 평균, 분산, 정규화 | `stats.go` | `Statistics.java` |
| 정규 분포 | `stats.go` | `Statistics.java` |
| 공분산과 상관 | `stats.go` | `Statistics.java` |
| k-평균 군집화 | `stats.go` | `Statistics.java` |
| 확률 분포와 조건부 확률 | `probability.go` | `Probability.java` |
| 로짓, softmax, temperature | `probability.go` | `Probability.java` |
| 로그, 교차 엔트로피, 최대 우도 | `probability.go` | `Probability.java` |
| 엔트로피와 perplexity | `probability.go` | `Probability.java` |
| KL 발산 | `probability.go` | `Probability.java` |
| 선형 회귀와 로지스틱 회귀 | `models.go` | `Models.java` |
| 마르코프 체인 | `models.go` | `Models.java` |
| 벨만 방정식 | `models.go` | `Models.java` |
| 스케일링 법칙 | `models.go` | `Models.java` |
| 검색의 통계 | `search.go` | `SearchStats.java` |
| t-검정과 분산 분석 | `search.go` | `SearchStats.java` |

출력은 검산 한 줄에 한 항목이다. `부록` 열이 원고에 적힌 값, `계산` 열이 라이브러리가 내놓은 값이다.

```
== 로짓, softmax, temperature
  [OK  ] exp(2.0)                         부록 7.39         계산 7.389056
  [OK  ] softmax 1위 확률                 부록 0.66         계산 0.659001
  [OK  ] T=0.5의 1위 확률                 부록 0.86         계산 0.86018
```

전부 통과하면 종료 코드가 0이고, 하나라도 어긋나면 1이다. 값을 바꿔 가며 결과가 어떻게 달라지는지 보는 것이 이 코드의 사용법이다. 예를 들어 `softmax`에 넘기는 temperature를 바꾸거나, 표본 크기를 줄여 t-검정의 판정이 뒤집히는 지점을 찾아볼 수 있다.

## 두 라이브러리의 차이

같은 부록을 옮기면서 드러난 차이가 몇 군데 있다.

- **자동 미분**: Commons Math의 `DerivativeStructure`는 4장의 autograd와 같은 자동 미분이라 도함수를 오차 없이 정확히 계산한다. Gonum에는 없어서 `diff/fd`의 수치 미분을 쓴다.
- **가설 검정**: Commons Math는 `TTest`, `OneWayAnova`를 제공한다. Gonum에는 분포의 CDF만 있어서 검정통계량과 p값을 직접 조립했다(`search.go`의 `pairedTTest`, `oneWayANOVA`).
- **k-평균**: Commons Math의 `KMeansPlusPlusClusterer`를 쓸 수 있다. Gonum에는 없어서 Lloyd 알고리즘을 직접 구현했다(`stats.go`의 `lloyd`).
- **주성분 분석**: Gonum은 `stat.PC`로 바로 구한다. Commons Math에는 없어서 공분산 행렬의 고유분해로 직접 계산했다.
- **엔트로피와 KL 발산**: Gonum은 `stat.Entropy`, `stat.KullbackLeibler`를 제공한다. Commons Math에는 없어서 정의 그대로 구현했다.

없는 기능을 직접 구현한 자리가 오히려 부록의 정의를 코드로 확인하는 지점이 된다.

## 부록의 주장 가운데 검산으로 확인되는 것들

단순히 숫자를 맞춰 보는 데 그치지 않고, 부록이 문장으로 서술한 관계를 코드로 확인한다.

- 상관계수는 평균을 뺀 두 배열의 코사인 유사도와 정확히 같은 값이다.
- 교차 엔트로피(P, Q) = 엔트로피(P) + KL(P || Q)라는 관계식이 성립한다.
- 시그모이드는 선택지가 두 개일 때의 softmax와 같은 함수이고, 로그 손실은 교차 엔트로피와 같은 식이다.
- 평균 0, 분산 1인 d차원 벡터 두 개의 내적은 분산이 d가 되고, `sqrt(d)`로 나누면 차원과 무관하게 분산이 1 근처로 유지된다.
- 정규 분포의 분위수에 격자를 두면 균등 격자보다 양자화 오차가 작다(7장 NF4의 근거). 분포를 모른 채 데이터에서 격자를 찾는 1차원 k-평균도 같은 결과에 도달한다.
- 정확도 80%, 문항 100개면 표준 오차가 4%p이므로 79%와 83%의 차이는 유의하지 않다. 같은 데이터라도 대응표본 t-검정은 그 개선을 잡아낸다.
- 쌍별로 t-검정을 반복하면 위양성률이 유의수준 5%를 훌쩍 넘는다(다중 비교 문제).

## 준비물

- Go 1.22 이상(`math/rand/v2` 사용). Gonum은 `go run` 시 자동으로 받는다.
- Java 21 이상. Gradle wrapper가 있으므로 Gradle을 따로 설치하지 않아도 된다.
