import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws IOException {
        // 재현 가능하도록 난수 시드를 고정한다
        Random rng = new Random(42);

        Path path = Path.of("names.txt");
        if (!Files.exists(path)) {
            System.err.println("names.txt를 읽을 수 없습니다: " + path.toAbsolutePath());
            System.exit(1);
        }
        List<String> docs = Files.readAllLines(path).stream()
                .map(String::strip).filter(s -> !s.isEmpty()).toList();

        Tokenizer tok = new Tokenizer(docs);
        Gpt model = new Gpt(tok.vocabSize(), rng);
        System.out.printf("데이터 %d건, 어휘 크기 %d, 파라미터 %d개%n",
                docs.size(), tok.vocabSize(), model.params.size());

        // tag::training[]
        final int numSteps = 1500;   // 학습 스텝 수
        final double baseLr = 0.01;  // 기본 학습률
        final double beta1 = 0.9;    // Adam 1차 모멘트 감쇠율
        final double beta2 = 0.95;   // Adam 2차 모멘트 감쇠율
        final double epsAdam = 1e-8;
        List<Value> params = model.params;
        double[] mBuf = new double[params.size()]; // Adam 1차 모멘트(기울기 이동 평균)
        double[] vBuf = new double[params.size()]; // Adam 2차 모멘트(기울기 제곱 이동 평균)

        double lossSum = 0;
        int lossCount = 0;
        for (int step = 1; step <= numSteps; step++) {
            // 이름 하나를 골라 시퀀스를 만든다. 입력과 목표는 한 칸씩 어긋난다.
            // 예: seq = [BOS, e, m, m, a, BOS] 이면
            //     입력 BOS -> 목표 e, 입력 e -> 목표 m, ... 입력 a -> 목표 BOS
            String doc = docs.get(rng.nextInt(docs.size()));
            int[] ids = tok.encode(doc);
            int[] seq = new int[ids.length + 2];
            seq[0] = tok.bos;
            System.arraycopy(ids, 0, seq, 1, ids.length);
            seq[seq.length - 1] = tok.bos;
            int n = Math.min(seq.length - 1, Gpt.blockSize);

            // 순전파: 위치마다 다음 토큰의 확률을 구하고 교차 엔트로피를 쌓는다
            Gpt.KvCache cache = model.newCache();
            Value loss = new Value(0);
            for (int pos = 0; pos < n; pos++) {
                Value[] logits = model.forward(seq[pos], pos, cache);
                Value[] probs = Gpt.softmax(logits);
                // 정답 토큰 확률의 로그에 음수를 취한 값이 손실이다
                loss = loss.add(probs[seq[pos + 1]].log().neg());
            }
            loss = loss.mul(new Value(1.0 / n));

            // 역전파: 기울기를 0으로 되돌린 뒤 손실에서 출발해 전파한다
            for (Value p : params) {
                p.grad = 0;
            }
            loss.backward();

            // Adam 갱신. 학습률은 마지막 스텝에서 0이 되도록 선형 감소시킨다.
            double lr = baseLr * (1 - (double) (step - 1) / numSteps);
            for (int i = 0; i < params.size(); i++) {
                Value p = params.get(i);
                mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * p.grad;
                vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * p.grad * p.grad;
                double mHat = mBuf[i] / (1 - Math.pow(beta1, step));
                double vHat = vBuf[i] / (1 - Math.pow(beta2, step));
                p.data -= lr * mHat / (Math.sqrt(vHat) + epsAdam);
            }

            lossSum += loss.data;
            lossCount++;
            if (step == 1 || step % 100 == 0) {
                System.out.printf("스텝 %4d | 평균 손실 %.4f%n", step, lossSum / lossCount);
                lossSum = 0;
                lossCount = 0;
            }
        }
        // end::training[]

        // tag::sampling[]
        // 샘플링: BOS 토큰으로 시작해 다음 토큰을 확률에 따라 뽑는다.
        // temperature가 낮을수록 확률이 높은 토큰에 집중한다.
        final double temperature = 0.8;
        System.out.println("\n생성된 이름 20개:");
        for (int i = 0; i < 20; i++) {
            Gpt.KvCache cache = model.newCache();
            int tokenId = tok.bos;
            List<Integer> out = new ArrayList<>();
            for (int pos = 0; pos < Gpt.blockSize; pos++) {
                Value[] logits = model.forward(tokenId, pos, cache);
                double[] probs = softmaxData(logits, temperature);
                int next = sampleFrom(probs, rng);
                if (next == tok.bos) {
                    break; // BOS가 나오면 이름이 끝난 것으로 본다
                }
                out.add(next);
                tokenId = next;
            }
            System.out.println(tok.decode(out));
        }
        // end::sampling[]
    }

    // 역전파 그래프 없이 로짓 값만으로 확률을 계산한다.
    static double[] softmaxData(Value[] logits, double temperature) {
        double maxVal = logits[0].data;
        for (Value l : logits) {
            if (l.data > maxVal) {
                maxVal = l.data;
            }
        }
        double[] probs = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp((logits[i].data - maxVal) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    // 누적 확률을 따라 토큰 하나를 뽑는다.
    static int sampleFrom(double[] probs, Random rng) {
        double r = rng.nextDouble();
        double acc = 0;
        for (int i = 0; i < probs.length; i++) {
            acc += probs[i];
            if (r < acc) {
                return i;
            }
        }
        return probs.length - 1;
    }
}
