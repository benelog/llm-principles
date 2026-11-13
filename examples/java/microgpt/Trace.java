import java.util.ArrayList;
import java.util.List;

// Trace는 forward 한 번의 중간값을 기록해 사람이 읽을 수 있게 출력한다.
// 학습과 생성의 계산에는 관여하지 않는다. 임베딩에서 손실까지 값이 실제로
// 어떻게 흘러가는지 숫자로 확인하는 데만 쓴다.
public class Trace {
    final double[][] scores = new double[Gpt.nHead][];  // [헤드][위치] 어텐션 점수
    final double[][] weights = new double[Gpt.nHead][]; // [헤드][위치] softmax 가중치
    final double[][] attnOut = new double[Gpt.nHead][]; // [헤드][차원] 밸류의 가중합
    Gpt.KvCache cache; // 입력을 처리하며 쌓인 키와 밸류
    Value[] logits;    // 마지막 위치의 forward가 내놓은 로짓

    // Gpt.forward가 헤드 하나의 어텐션을 계산한 직후에 부른다.
    // forward를 부를 때마다 덮어쓰므로 마지막 위치의 값만 남는다.
    void record(int head, Value[] s, Value[] w, Value[] out) {
        scores[head] = new double[s.length];
        weights[head] = new double[w.length];
        for (int t = 0; t < s.length; t++) {
            scores[head][t] = s[t].data;
            weights[head][t] = w[t].data;
        }
        attnOut[head] = new double[Gpt.headDim];
        for (int d = 0; d < Gpt.headDim; d++) {
            attnOut[head][d] = out[head * Gpt.headDim + d].data;
        }
    }

    // tag::trace[]
    // 토큰 열을 위치 0부터 차례로 넣고, 마지막 토큰을 넣은 forward의 중간값을 남긴다.
    static Trace capture(Gpt model, int[] input) {
        Trace trace = new Trace();
        trace.cache = model.newCache();
        model.trace = trace; // forward가 어텐션 점수와 가중치를 기록하게 한다
        for (int pos = 0; pos < input.length; pos++) {
            trace.logits = model.forward(input[pos], pos, trace.cache);
        }
        model.trace = null;
        return trace;
    }
    // end::trace[]

    // BOS 뒤에 prefix의 글자들을 넣고, 그다음 글자를 예측하는 forward를 추적해 출력한다.
    // answer는 손실을 계산할 정답 글자다.
    static void run(Gpt model, Tokenizer tok, String prefix, char answer) {
        int[] ids = tok.encode(prefix);
        int[] input = new int[ids.length + 1];
        input[0] = tok.bos;
        System.arraycopy(ids, 0, input, 1, ids.length);
        int target = tok.encode(String.valueOf(answer))[0];

        capture(model, input).print(model, tok, input, target);
    }

    void print(Gpt model, Tokenizer tok, int[] input, int target) {
        int pos = input.length - 1;
        int tokenId = input[pos];
        List<String> names = new ArrayList<>();
        for (int id : input) {
            names.add(label(tok, id));
        }
        System.out.printf("%n== forward 한 번 추적: 입력 %s, 다음 토큰 예측 ==%n", names);

        // 1) 임베딩: 표에서 행 하나씩을 꺼내 성분끼리 더한다
        Value[] tokEmb = model.wte[tokenId];
        Value[] posEmb = model.wpe[pos];
        double[] x = new double[Gpt.nEmbd];
        for (int i = 0; i < Gpt.nEmbd; i++) {
            x[i] = tokEmb[i].add(posEmb[i]).data;
        }
        System.out.printf("현재 토큰 %s (ID %d), 위치 %d%n", label(tok, tokenId), tokenId, pos);
        System.out.printf("  wte[%d] = %s  토큰 임베딩 %d개 중 앞 4개%n", tokenId, head4(data(tokEmb)), Gpt.nEmbd);
        System.out.printf("  wpe[%d]  = %s  위치 임베딩%n", pos, head4(data(posEmb)));
        System.out.printf("  x       = %s  두 벡터를 성분끼리 더한 값%n", head4(x));

        // 2) 어텐션: 점수, 가중치, 밸류의 가중합
        List<Value[]> values = cache.values.get(Gpt.nLayer - 1);
        for (int h = 0; h < Gpt.nHead; h++) {
            System.out.printf("어텐션 헤드 %d (위치 0~%d의 키와 밸류를 참조)%n", h, pos);
            System.out.printf("  scores  = %s%n", row(scores[h]));
            System.out.printf("  weights = %s  합 %.3f%n", row(weights[h]), sum(weights[h]));
            if (h == 0) {
                double[] v0 = new double[values.size()];
                StringBuilder terms = new StringBuilder();
                for (int t = 0; t < v0.length; t++) {
                    v0[t] = values.get(t)[0].data;
                    terms.append(t == 0 ? "" : " + ").append(String.format("%.3f×(%.3f)", weights[h][t], v0[t]));
                }
                System.out.printf("  v[t][0] = %s  각 위치 밸류의 0번 성분%n", row(v0));
                System.out.printf("  attnOut[0] = %s = %.3f%n", terms, attnOut[h][0]);
            }
        }

        // 3) 출력: 로짓, 확률, 정답 토큰의 손실. 학습 루프와 같은 식을 쓴다.
        Value[] probs = Gpt.softmax(logits);
        Value loss = probs[target].log().neg();
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < logits.length; i++) {
            order.add(i);
        }
        order.sort((a, b) -> Double.compare(logits[b].data, logits[a].data));
        System.out.println("로짓 상위 5개와 softmax 확률");
        for (int i = 0; i < 5; i++) {
            int id = order.get(i);
            System.out.printf("  %-3s 로짓 %6.3f  확률 %.3f%n", label(tok, id), logits[id].data, probs[id].data);
        }
        System.out.printf("정답 %s: 확률 %.3f, 손실 -ln(%.3f) = %.2f%n",
                label(tok, target), probs[target].data, probs[target].data, loss.data);
    }

    private static String label(Tokenizer tok, int id) {
        return id == tok.bos ? "BOS" : tok.decode(List.of(id));
    }

    private static double[] data(Value[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i].data;
        }
        return out;
    }

    private static double sum(double[] values) {
        double total = 0;
        for (double v : values) {
            total += v;
        }
        return total;
    }

    // 벡터의 앞 4개 성분만 보여 준다.
    private static String head4(double[] values) {
        return row(new double[] {values[0], values[1], values[2], values[3]}).replace("]", ", ...]");
    }

    private static String row(double[] values) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            sb.append(i == 0 ? "" : ", ").append(String.format("%6.3f", values[i]));
        }
        return sb.append("]").toString();
    }
}
