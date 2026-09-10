import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// GPT는 토큰 임베딩, 위치 임베딩, 트랜스포머 블록, 최종 lm_head로 이루어진다.
public class Gpt {
    // 모델 크기 설정. 파라미터 수천 개 수준의 아주 작은 GPT다.
    static final int nEmbd = 16;              // 임베딩 차원
    static final int nHead = 2;               // 어텐션 헤드 수
    static final int headDim = nEmbd / nHead; // 헤드 하나가 담당하는 차원
    static final int nLayer = 1;              // 트랜스포머 블록 수
    static final int blockSize = 16;          // 최대 문맥 길이

    // Block은 트랜스포머 블록 하나의 가중치 묶음이다.
    static class Block {
        Value[][] attnQ, attnK, attnV; // 어텐션 Q, K, V 투영 (nEmbd x nEmbd)
        Value[][] attnProj;            // 어텐션 출력 투영 (nEmbd x nEmbd)
        Value[][] mlpFc;               // MLP 확장 (4*nEmbd x nEmbd)
        Value[][] mlpProj;             // MLP 축소 (nEmbd x 4*nEmbd)
    }

    // KvCache는 이미 처리한 위치들의 키와 값을 층별로 보관한다.
    // 학습과 샘플링 모두 이 캐시에 한 위치씩 쌓아 가며 진행한다.
    static class KvCache {
        final List<List<Value[]>> keys = new ArrayList<>();   // [층][위치][차원]
        final List<List<Value[]>> values = new ArrayList<>();

        KvCache() {
            for (int l = 0; l < nLayer; l++) {
                keys.add(new ArrayList<>());
                values.add(new ArrayList<>());
            }
        }
    }

    final int vocabSize;
    final Value[][] wte;    // 토큰 임베딩 (vocabSize x nEmbd)
    final Value[][] wpe;    // 위치 임베딩 (blockSize x nEmbd)
    final List<Block> blocks = new ArrayList<>();
    final Value[][] lmHead; // 최종 로짓 투영 (vocabSize x nEmbd)
    final List<Value> params = new ArrayList<>(); // 옵티마이저가 갱신할 전체 파라미터 목록

    // 작은 GPT를 만들고 가중치를 정규분포(표준편차 0.02)로 초기화한다.
    public Gpt(int vocabSize, Random rng) {
        this.vocabSize = vocabSize;
        wte = newMatrix(vocabSize, nEmbd, rng);
        wpe = newMatrix(blockSize, nEmbd, rng);
        for (int l = 0; l < nLayer; l++) {
            Block blk = new Block();
            blk.attnQ = newMatrix(nEmbd, nEmbd, rng);
            blk.attnK = newMatrix(nEmbd, nEmbd, rng);
            blk.attnV = newMatrix(nEmbd, nEmbd, rng);
            blk.attnProj = newMatrix(nEmbd, nEmbd, rng);
            blk.mlpFc = newMatrix(4 * nEmbd, nEmbd, rng);
            blk.mlpProj = newMatrix(nEmbd, 4 * nEmbd, rng);
            blocks.add(blk);
        }
        lmHead = newMatrix(vocabSize, nEmbd, rng);
    }

    private Value[][] newMatrix(int rows, int cols, Random rng) {
        Value[][] w = new Value[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                w[i][j] = new Value(rng.nextGaussian() * 0.02);
                params.add(w[i][j]);
            }
        }
        return w;
    }

    // 시퀀스 하나를 처리할 빈 캐시를 만든다.
    public KvCache newCache() {
        return new KvCache();
    }

    // 행렬 w와 벡터 x의 곱을 계산한다.
    static Value[] matvec(Value[][] w, Value[] x) {
        Value[] out = new Value[w.length];
        for (int i = 0; i < w.length; i++) {
            Value sum = w[i][0].mul(x[0]);
            for (int j = 1; j < x.length; j++) {
                sum = sum.add(w[i][j].mul(x[j]));
            }
            out[i] = sum;
        }
        return out;
    }

    // 벡터를 제곱 평균의 제곱근으로 나눠 크기를 일정하게 유지한다.
    static Value[] rmsnorm(Value[] x) {
        Value ss = x[0].mul(x[0]);
        for (int i = 1; i < x.length; i++) {
            ss = ss.add(x[i].mul(x[i]));
        }
        // 작은 값을 더해 0으로 나누는 상황을 막는다
        Value scale = ss.mul(new Value(1.0 / x.length)).add(new Value(1e-5)).pow(-0.5);
        Value[] out = new Value[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].mul(scale);
        }
        return out;
    }

    // 로짓을 확률 분포로 바꾼다.
    static Value[] softmax(Value[] logits) {
        // 수치 안정성을 위해 최댓값을 빼고 지수를 취한다
        double maxVal = logits[0].data;
        for (Value l : logits) {
            if (l.data > maxVal) {
                maxVal = l.data;
            }
        }
        Value shift = new Value(maxVal);
        Value[] exps = new Value[logits.length];
        Value sum = logits[0].sub(shift).exp();
        exps[0] = sum;
        for (int i = 1; i < logits.length; i++) {
            exps[i] = logits[i].sub(shift).exp();
            sum = sum.add(exps[i]);
        }
        Value inv = sum.pow(-1);
        Value[] out = new Value[logits.length];
        for (int i = 0; i < exps.length; i++) {
            out[i] = exps[i].mul(inv);
        }
        return out;
    }

    // tag::forward[]
    // 토큰 하나를 받아 다음 토큰의 로짓을 계산한다.
    // 이전 위치의 K, V는 캐시에 쌓여 있으므로 현재 토큰은
    // 과거 위치만 참조하게 되고, causal 마스크가 따로 필요 없다.
    public Value[] forward(int tokenId, int posId, KvCache cache) {
        // 토큰 임베딩과 위치 임베딩을 더해 입력 벡터를 만든다
        Value[] x = new Value[nEmbd];
        for (int i = 0; i < nEmbd; i++) {
            x[i] = wte[tokenId][i].add(wpe[posId][i]);
        }

        for (int l = 0; l < blocks.size(); l++) {
            Block blk = blocks.get(l);
            // tag::attention[]
            // 1) 멀티헤드 어텐션: 현재 토큰이 이전 토큰들의 정보를 모은다
            Value[] xn = rmsnorm(x);
            Value[] q = matvec(blk.attnQ, xn);
            Value[] k = matvec(blk.attnK, xn);
            Value[] v = matvec(blk.attnV, xn);
            List<Value[]> keys = cache.keys.get(l);
            List<Value[]> values = cache.values.get(l);
            keys.add(k);
            values.add(v);

            Value[] attnOut = new Value[nEmbd];
            Value invSqrt = new Value(1 / Math.sqrt(headDim));
            for (int h = 0; h < nHead; h++) {
                int lo = h * headDim;
                // 현재 질의(q)와 캐시된 키들의 내적을 headDim 제곱근으로 나눈다
                Value[] scores = new Value[keys.size()];
                for (int t = 0; t < keys.size(); t++) {
                    Value[] kt = keys.get(t);
                    Value dot = q[lo].mul(kt[lo]);
                    for (int d = 1; d < headDim; d++) {
                        dot = dot.add(q[lo + d].mul(kt[lo + d]));
                    }
                    scores[t] = dot.mul(invSqrt);
                }
                // 소프트맥스 가중치로 캐시된 값(v) 벡터들을 섞는다
                Value[] weights = softmax(scores);
                for (int d = 0; d < headDim; d++) {
                    Value sum = weights[0].mul(values.get(0)[lo + d]);
                    for (int t = 1; t < weights.length; t++) {
                        sum = sum.add(weights[t].mul(values.get(t)[lo + d]));
                    }
                    attnOut[lo + d] = sum;
                }
            }
            // 출력 투영을 거쳐 잔차 연결로 원래 벡터에 더한다
            Value[] proj = matvec(blk.attnProj, attnOut);
            for (int i = 0; i < x.length; i++) {
                x[i] = x[i].add(proj[i]);
            }
            // end::attention[]

            // 2) MLP 블록: 토큰별 특징을 4배 넓힌 뒤 비선형 변환하고 되돌린다
            xn = rmsnorm(x);
            Value[] hidden = matvec(blk.mlpFc, xn);
            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = hidden[i].tanh();
            }
            proj = matvec(blk.mlpProj, hidden);
            for (int i = 0; i < x.length; i++) {
                x[i] = x[i].add(proj[i]);
            }
        }

        // 최종 정규화 후 어휘 크기의 로짓으로 투영한다
        return matvec(lmHead, rmsnorm(x));
    }
    // end::forward[]
}
