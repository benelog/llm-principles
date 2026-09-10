# math-lab: 7장 수학·통계 검산 실습

7장 "코드 뒤의 수학과 통계"(`book/ch07-math-stats.adoc`)가 본문에 적어 둔 값들을 수치 계산 라이브러리로 다시 계산해서 맞는지 확인하는 실습 코드다. [Apache Commons Math 3](https://commons.apache.org/proper/commons-math/) 하나로 선형대수, 분포, 회귀, 검정을 모두 다루며, Gradle로 빌드한다. 검산 항목은 190건이다.

```bash
cd java && ./gradlew run
```

`examples/java/`의 다른 예제들은 표준 라이브러리만 쓴다는 제약이 있으므로, 외부 라이브러리를 쓰는 이 실습은 별도 디렉터리에 두었다.

## 읽는 방법

7장의 절 순서를 그대로 따른다. 7장을 읽다가 확인하고 싶은 절이 나오면 같은 이름의 함수를 열어 보면 된다. 주성분 분석은 7장에서 데이터의 구조 절에 있지만, 검산 코드는 저랭크 근사와 같은 행렬 연산을 쓰므로 랭크 검산과 한 함수에 두었다.

| 7장 절 | 검산 함수 | 파일 |
|---|---|---|
| 벡터와 행렬 (벡터와 내적, 코사인 유사도, 행렬, 랭크와 저랭크 근사) | `vectorsAndMatrices`, `rankAndPca` | `LinearAlgebra.java` |
| 확률 (확률 분포와 조건부 확률, 베이즈 정리, softmax와 temperature, 샘플링) | `probabilityAndBayes`, `softmaxAndTemperature` | `Probability.java` |
| 로그와 정보량 (로그, 최대 우도와 교차 엔트로피, 엔트로피와 perplexity, KL 발산) | `crossEntropy`, `entropyAndPerplexity`, `klDivergence` | `Probability.java` |
| 미분 (도함수와 그래디언트, 체인 룰, 경사 하강법) | `run` | `Calculus.java` |
| 평균과 분산 (평균·분산·표준편차, 정규화와 sqrt(d), 정규 분포) | `moments`, `normalDistribution` | `Statistics.java` |
| 데이터의 구조 (공분산과 상관계수, 주성분 분석, 판별 분석, k-평균) | `covarianceAndCorrelation`, `rankAndPca`, `discriminant`, `kmeans` | `Statistics.java`, `LinearAlgebra.java` |
| 회귀 (선형 회귀, 로지스틱 회귀) | `regression` | `Models.java` |
| 확률 과정과 결정 (마르코프 체인, 벨만 방정식) | `markovChain`, `bellman` | `Models.java` |
| 규모와 측정 (스케일링 법칙, TF-IDF와 BM25, t-검정과 분산 분석) | `scalingLaw`, `searchStatistics`, `significanceTests` | `Models.java`, `SearchStats.java` |

출력은 검산 한 줄에 한 항목이다. `원고` 열이 7장에 적힌 값, `계산` 열이 라이브러리가 내놓은 값이다.

```
== 로짓, softmax, temperature
  [OK  ] exp(2.0)                         원고 7.39         계산 7.389056
  [OK  ] softmax 1위 확률                 원고 0.66         계산 0.659001
  [OK  ] T=0.5의 1위 확률                 원고 0.86         계산 0.86018
```

전부 통과하면 종료 코드가 0이고, 하나라도 어긋나면 1이다. 값을 바꿔 가며 결과가 어떻게 달라지는지 보는 것이 이 코드의 사용법이다. 예를 들어 `softmax`에 넘기는 temperature를 바꾸거나, 표본 크기를 줄여 t-검정의 판정이 뒤집히는 지점을 찾아볼 수 있다.

## Commons Math로 다루는 방법

- **자동 미분**: `DerivativeStructure`는 4장의 autograd와 같은 자동 미분이라 도함수를 오차 없이 정확히 계산한다.
- **가설 검정과 k-평균**: `TTest`, `OneWayAnova`, `KMeansPlusPlusClusterer`를 그대로 쓴다.
- **주성분 분석**: Commons Math에는 없어서 공분산 행렬의 고유분해로 직접 계산했다.
- **엔트로피와 KL 발산**: Commons Math에는 없어서 정의 그대로 구현했다.

없는 기능을 직접 구현한 부분이 오히려 7장의 정의를 코드로 확인하는 자리가 된다.

## 7장의 주장 가운데 검산으로 확인되는 것들

단순히 숫자를 맞춰 보는 데 그치지 않고, 7장이 문장으로 서술한 관계를 코드로 확인한다.

- 상관계수는 평균을 뺀 두 배열의 코사인 유사도와 정확히 같은 값이다.
- 체인 룰 예제 y = (2x + 1)²의 x = 1에서의 기울기는 자동 미분으로 정확히 12가 나오고, 학습률 0.1과 1.1의 경사 하강법 경로는 7장에 적힌 수열 그대로다.
- 교차 엔트로피(P, Q) = 엔트로피(P) + KL(P || Q)라는 관계식이 성립한다.
- 시그모이드는 선택지가 두 개일 때의 softmax와 같은 함수이고, 로그 손실은 교차 엔트로피와 같은 식이다.
- 평균 0, 분산 1인 d차원 벡터 두 개의 내적은 분산이 d가 되고, `sqrt(d)`로 나누면 차원과 무관하게 분산이 1 근처로 유지된다.
- 정규 분포의 분위수에 격자를 두면 균등 격자보다 양자화 오차가 작다(8장 NF4의 근거). 분포를 모른 채 데이터에서 격자를 찾는 1차원 k-평균도 같은 결과에 도달한다.
- 정확도 80%, 문항 100개면 표준 오차가 4%p이므로 79%와 83%의 차이는 유의하지 않다. 같은 데이터라도 대응표본 t-검정은 그 개선을 잡아낸다.
- 쌍별로 t-검정을 반복하면 위양성률이 유의수준 5%를 훌쩍 넘는다(다중 비교 문제).

## 준비물

- Java 21 이상. Gradle wrapper가 있으므로 Gradle을 따로 설치하지 않아도 된다.
