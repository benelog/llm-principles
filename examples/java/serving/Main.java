// KV 캐시와 프리필·디코드가 만드는 추론 비용 구조를 측정하는 예제.
// 학습 없이 무작위 가중치를 쓴다. 계산량과 메모리 크기는 가중치 값과
// 무관하므로 시간과 용량을 재는 데는 지장이 없다.

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class Main {
    // 시간을 잴 수 있도록 microGPT보다 키운 장난감 트랜스포머 크기.
    static final int vocabSize = 256; // 어휘 크기
    static final int nEmbd = 64;      // 임베딩 차원
    static final int nHead = 8;       // 어텐션 헤드 수
    static final int headDim = nEmbd / nHead;
    static final int nLayer = 4;      // 트랜스포머 블록 수
    static final int maxCtx = 512;    // 최대 문맥 길이

    // Layer는 트랜스포머 블록 하나의 가중치 묶음이다.
    static class Layer {
        double[][] wq, wk, wv, wo; // 어텐션 Q, K, V, 출력 투영
        double[][] wfc, wproj;     // MLP 확장과 축소
    }

    // Model은 임베딩, 블록들, 최종 로짓 투영으로 이루어진다.
    static class Model {
        final double[][] wte;  // 토큰 임베딩 (vocabSize x nEmbd)
        final double[][] wpe;  // 위치 임베딩 (maxCtx x nEmbd)
        final List<Layer> layers = new ArrayList<>();
        final double[][] wout; // 최종 로짓 투영 (vocabSize x nEmbd)

        Model(Random rng) {
            wte = newMatrix(vocabSize, nEmbd, rng);
            wpe = newMatrix(maxCtx, nEmbd, rng);
            wout = newMatrix(vocabSize, nEmbd, rng);
            for (int l = 0; l < nLayer; l++) {
                Layer layer = new Layer();
                layer.wq = newMatrix(nEmbd, nEmbd, rng);
                layer.wk = newMatrix(nEmbd, nEmbd, rng);
                layer.wv = newMatrix(nEmbd, nEmbd, rng);
                layer.wo = newMatrix(nEmbd, nEmbd, rng);
                layer.wfc = newMatrix(4 * nEmbd, nEmbd, rng);
                layer.wproj = newMatrix(nEmbd, 4 * nEmbd, rng);
                layers.add(layer);
            }
        }

        // 토큰 임베딩과 위치 임베딩을 더한 입력 벡터다.
        double[] embed(int tokenId, int pos) {
            return add(wte[tokenId], wpe[pos]);
        }

        // tag::kvcache[]
        // KvCache는 처리를 마친 위치들의 키와 밸류를 층별로 보관한다.
        static class KvCache {
            final List<List<double[]>> keys = new ArrayList<>();   // [층][위치][차원]
            final List<List<double[]>> values = new ArrayList<>();

            KvCache() {
                for (int l = 0; l < nLayer; l++) {
                    keys.add(new ArrayList<>());
                    values.add(new ArrayList<>());
                }
            }
        }

        KvCache newCache() {
            return new KvCache();
        }

        // 토큰 하나를 처리해 캐시에 K, V를 쌓고 마지막 은닉 벡터를 돌려준다.
        // 2부 microGPT의 forward와 같은 구조이며, 학습이 필요 없으므로
        // 자동 미분 없이 double로만 계산한다.
        double[] decodeStep(int tokenId, int pos, KvCache cache) {
            double[] x = embed(tokenId, pos);
            for (int l = 0; l < layers.size(); l++) {
                Layer layer = layers.get(l);
                // 어텐션: 현재 토큰의 K, V는 캐시에 추가하고, Q는 캐시 전체를 참조한다
                double[] xn = rmsnorm(x);
                double[] q = matvec(layer.wq, xn);
                cache.keys.get(l).add(matvec(layer.wk, xn));
                cache.values.get(l).add(matvec(layer.wv, xn));
                x = add(x, matvec(layer.wo, attend(q, cache.keys.get(l), cache.values.get(l))));
                // MLP
                xn = rmsnorm(x);
                x = add(x, matvec(layer.wproj, tanhVec(matvec(layer.wfc, xn))));
            }
            return rmsnorm(x);
        }
        // end::kvcache[]

        // tag::prefill[]
        // 프롬프트의 모든 위치를 층 단위로 한꺼번에 처리한다.
        // 프롬프트 토큰은 전부 미리 알고 있으므로 위치별 계산이 서로 독립이고,
        // 병렬로 실행할 수 있다. 디코드는 직전 토큰이 나와야 다음 토큰을 시작할
        // 수 있으므로 이런 병렬화가 불가능하다.
        PrefillResult prefill(int[] tokens) {
            int n = tokens.length;
            KvCache cache = newCache();
            double[][] xs = new double[n][];
            parallelFor(n, i -> xs[i] = embed(tokens[i], i));
            for (int l = 0; l < layers.size(); l++) {
                Layer layer = layers.get(l);
                double[][] qs = new double[n][];
                double[][] ks = new double[n][];
                double[][] vs = new double[n][];
                // 1) 모든 위치의 Q, K, V를 병렬로 계산해 캐시를 채운다
                parallelFor(n, i -> {
                    double[] xn = rmsnorm(xs[i]);
                    qs[i] = matvec(layer.wq, xn);
                    ks[i] = matvec(layer.wk, xn);
                    vs[i] = matvec(layer.wv, xn);
                });
                List<double[]> keys = new ArrayList<>(Arrays.asList(ks));
                List<double[]> values = new ArrayList<>(Arrays.asList(vs));
                cache.keys.set(l, keys);
                cache.values.set(l, values);
                // 2) 어텐션과 MLP도 병렬. 위치 i는 0..i의 키·밸류만 참조한다(causal)
                parallelFor(n, i -> {
                    double[] attn = attend(qs[i], keys.subList(0, i + 1), values.subList(0, i + 1));
                    double[] x = add(xs[i], matvec(layer.wo, attn));
                    double[] xn = rmsnorm(x);
                    xs[i] = add(x, matvec(layer.wproj, tanhVec(matvec(layer.wfc, xn))));
                });
            }
            return new PrefillResult(cache, rmsnorm(xs[n - 1]));
        }
        // end::prefill[]

        // 로짓이 가장 큰 토큰을 고른다(greedy 디코딩). 캐시 유무를 비교할 때
        // 두 경로가 같은 토큰 열을 내는지 확인하기 위해 무작위성을 뺐다.
        int nextToken(double[] h) {
            double[] logits = matvec(wout, h);
            int best = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            return best;
        }

        // tag::nocache[]
        // 캐시 없이 토큰을 생성한다. 새 토큰을 하나 만들 때마다
        // 빈 캐시에서 출발해 시퀀스 전체를 처음부터 다시 계산한다.
        int[] generateNoCache(int[] prompt, int numTokens) {
            int[] seq = Arrays.copyOf(prompt, prompt.length + numTokens);
            for (int len = prompt.length; len < seq.length; len++) {
                KvCache cache = newCache();
                double[] h = null;
                for (int pos = 0; pos < len; pos++) { // 앞 토큰 전체를 다시 계산한다
                    h = decodeStep(seq[pos], pos, cache);
                }
                seq[len] = nextToken(h);
            }
            return Arrays.copyOfRange(seq, prompt.length, seq.length);
        }

        // 프리필이 채운 캐시를 이어받아, 새 토큰마다 decodeStep 한 번만
        // 실행한다. 앞 토큰들의 K, V는 캐시에서 그대로 재사용된다.
        int[] generate(KvCache cache, double[] h, int startPos, int numTokens) {
            int[] out = new int[numTokens];
            for (int i = 0; i < numTokens; i++) {
                out[i] = nextToken(h);
                h = decodeStep(out[i], startPos + i, cache);
            }
            return out;
        }
        // end::nocache[]
    }

    // 프리필의 결과. 채워진 캐시와 마지막 위치의 은닉 벡터다.
    record PrefillResult(Model.KvCache cache, double[] hidden) {}

    static double[][] newMatrix(int rows, int cols, Random rng) {
        double[][] w = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                w[i][j] = rng.nextGaussian() * 0.02;
            }
        }
        return w;
    }

    // 행렬 w와 벡터 x의 곱이다.
    static double[] matvec(double[][] w, double[] x) {
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            double sum = 0;
            for (int j = 0; j < x.length; j++) {
                sum += w[i][j] * x[j];
            }
            out[i] = sum;
        }
        return out;
    }

    // 벡터를 제곱 평균의 제곱근으로 나눠 크기를 일정하게 유지한다.
    static double[] rmsnorm(double[] x) {
        double ss = 0;
        for (double v : x) {
            ss += v * v;
        }
        double scale = 1 / Math.sqrt(ss / x.length + 1e-5);
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * scale;
        }
        return out;
    }

    static double[] add(double[] a, double[] b) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    static double[] tanhVec(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.tanh(x[i]);
        }
        return out;
    }

    // 질의 벡터 하나가 캐시된 키·밸류 전체를 참조하는 멀티헤드 어텐션이다.
    static double[] attend(double[] q, List<double[]> keys, List<double[]> values) {
        double[] out = new double[nEmbd];
        double invSqrt = 1 / Math.sqrt(headDim);
        for (int h = 0; h < nHead; h++) {
            int lo = h * headDim;
            // 질의와 각 키의 내적으로 점수를 매긴다
            double[] scores = new double[keys.size()];
            double maxScore = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < keys.size(); t++) {
                double[] k = keys.get(t);
                double dot = 0;
                for (int d = 0; d < headDim; d++) {
                    dot += q[lo + d] * k[lo + d];
                }
                scores[t] = dot * invSqrt;
                if (scores[t] > maxScore) {
                    maxScore = scores[t];
                }
            }
            // softmax 가중치로 밸류들을 섞는다
            double sum = 0;
            for (int t = 0; t < scores.length; t++) {
                scores[t] = Math.exp(scores[t] - maxScore);
                sum += scores[t];
            }
            for (int t = 0; t < values.size(); t++) {
                double[] v = values.get(t);
                double w = scores[t] / sum;
                for (int d = 0; d < headDim; d++) {
                    out[lo + d] += w * v[lo + d];
                }
            }
        }
        return out;
    }

    // f(0)부터 f(n-1)까지를 CPU 코어들에 나눠 실행한다.
    static void parallelFor(int n, IntConsumer f) {
        IntStream.range(0, n).parallel().forEach(f);
    }

    // tag::memory[]
    // 컨텍스트를 가득 채웠을 때 KV 캐시가 차지하는 크기다.
    // K와 V 두 벌 × 층 수 × KV 헤드 수 × 헤드 차원 × 컨텍스트 길이 × 원소 크기.
    static long kvCacheBytes(int layers, int kvHeads, int headDim, int ctxLen, int bytesPerElem) {
        return 2L * layers * kvHeads * headDim * ctxLen * bytesPerElem;
    }
    // end::memory[]

    static String formatBytes(long b) {
        if (b >= 1L << 30) {
            return String.format("%.1f GiB", b / (double) (1L << 30));
        } else if (b >= 1L << 20) {
            return String.format("%.1f MiB", b / (double) (1L << 20));
        }
        return String.format("%.1f KiB", b / (double) (1L << 10));
    }

    // 나노초 단위 시간을 읽기 좋은 단위로 바꾼다.
    static String formatDuration(long nanos) {
        if (nanos >= 1_000_000_000L) {
            return String.format("%.4fs", nanos / 1e9);
        } else if (nanos >= 1_000_000L) {
            return String.format("%.2fms", nanos / 1e6);
        }
        return String.format("%dµs", Math.round(nanos / 1e3));
    }

    public static void main(String[] args) {
        Random rng = new Random(42);
        Model model = new Model(rng);

        final int promptLen = 256;
        final int genLen = 64;
        int[] prompt = new int[promptLen];
        for (int i = 0; i < promptLen; i++) {
            prompt[i] = rng.nextInt(vocabSize);
        }

        System.out.printf("모델: %d층, 임베딩 %d, 헤드 %d개 | 프롬프트 %d토큰, 생성 %d토큰, CPU 코어 %d개%n%n",
                nLayer, nEmbd, nHead, promptLen, genLen, Runtime.getRuntime().availableProcessors());

        // 워밍업: JVM은 처음 실행하는 코드를 인터프리터로 돌리다가 자주 쓰이는
        // 메서드를 JIT 컴파일한다. 먼저 재는 쪽이 느리게 나오는 편향을 없애기
        // 위해 측정 전에 같은 경로를 한 번 지나가 둔다.
        int[] warmup = Arrays.copyOf(prompt, 64);
        for (int round = 0; round < 20; round++) {
            Model.KvCache c = model.newCache();
            for (int pos = 0; pos < warmup.length; pos++) {
                model.decodeStep(warmup[pos], pos, c);
            }
            model.prefill(warmup);
        }

        // 1) 프리필: 순차 처리와 병렬 처리
        long t0 = System.nanoTime();
        Model.KvCache seqCache = model.newCache();
        double[] hSeq = null;
        for (int pos = 0; pos < promptLen; pos++) {
            hSeq = model.decodeStep(prompt[pos], pos, seqCache);
        }
        long seqPrefill = System.nanoTime() - t0;

        t0 = System.nanoTime();
        PrefillResult result = model.prefill(prompt);
        long parPrefill = System.nanoTime() - t0;
        Model.KvCache cache = result.cache();
        double[] h = result.hidden();

        System.out.println("== 프리필: 프롬프트 처리 ==");
        System.out.printf("한 토큰씩 순차 처리:  %8s%n", formatDuration(seqPrefill));
        System.out.printf("위치들을 병렬 처리:   %8s (%.1f배)%n", formatDuration(parPrefill),
                (double) seqPrefill / parPrefill);
        System.out.printf("두 방식의 마지막 은닉 벡터 일치: %b%n%n", Arrays.equals(hSeq, h));

        // 2) 디코드: 캐시 사용과 캐시 없음
        t0 = System.nanoTime();
        int[] withCache = model.generate(cache, h, promptLen, genLen);
        long cachedTime = System.nanoTime() - t0;

        t0 = System.nanoTime();
        int[] noCache = model.generateNoCache(prompt, genLen);
        long noCacheTime = System.nanoTime() - t0;

        System.out.println("== 디코드: 토큰 생성 ==");
        System.out.printf("KV 캐시 사용:  %8s (토큰당 %s)%n", formatDuration(cachedTime),
                formatDuration(cachedTime / genLen));
        System.out.printf("캐시 없음:     %8s (토큰당 %s, %.0f배)%n", formatDuration(noCacheTime),
                formatDuration(noCacheTime / genLen), (double) noCacheTime / cachedTime);
        System.out.printf("두 방식의 생성 결과 일치: %b%n%n", Arrays.equals(withCache, noCache));

        // 3) KV 캐시 메모리 크기
        System.out.println("== KV 캐시 크기 ==");
        System.out.printf("이 예제 모델 (%d층, 헤드 %d개, 컨텍스트 %d, double): %s%n",
                nLayer, nHead, maxCtx, formatBytes(kvCacheBytes(nLayer, nHead, headDim, maxCtx, 8)));
        // Llama 3 8B: 32층, KV 헤드 8개(GQA), 헤드 차원 128, FP16(2바이트)
        System.out.println("Llama 3 8B (32층, KV 헤드 8개, 헤드 차원 128, FP16):");
        for (int ctx : new int[] {8192, 131072}) {
            System.out.printf("  컨텍스트 %7d 토큰: %s%n", ctx, formatBytes(kvCacheBytes(32, 8, 128, ctx, 2)));
        }
        System.out.printf("GQA 없이 KV 헤드 32개라면, 컨텍스트 %d 토큰: %s%n",
                131072, formatBytes(kvCacheBytes(32, 32, 128, 131072, 2)));
    }
}
