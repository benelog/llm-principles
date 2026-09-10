// safetensors 포맷을 직접 만들어 보고 다시 읽어 보는 예제.
// 외부 라이브러리 없이 JDK 표준 라이브러리만 사용한다.

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class Main {

    // tag::header-type[]
    // TensorInfo는 safetensors 헤더에 기록되는 텐서 하나의 메타데이터다.
    // 실제 숫자값은 헤더 뒤의 바이너리 영역에 있고,
    // 헤더는 그 값을 어떻게 해석할지(자료형, 모양, 위치)만 알려 준다.
    record TensorInfo(
            String dtype,      // 예: "F32"
            int[] shape,       // 예: [4, 3]
            long[] dataOffsets // 데이터 영역 안에서의 [시작, 끝] 바이트 위치
    ) {
        // JSON 헤더에서 이 텐서에 해당하는 항목 하나를 해석한다.
        // {"dtype":"F32","shape":[4,3],"data_offsets":[0,48]} 형태다.
        static TensorInfo fromJson(Map<String, Object> entry) {
            List<?> shape = (List<?>) entry.get("shape");
            List<?> offsets = (List<?>) entry.get("data_offsets");
            return new TensorInfo(
                    (String) entry.get("dtype"),
                    shape.stream().mapToInt(v -> ((Number) v).intValue()).toArray(),
                    offsets.stream().mapToLong(v -> ((Number) v).longValue()).toArray());
        }

        // 헤더에 쓸 JSON 항목 하나를 만든다.
        String toJson() {
            return "{\"dtype\":\"" + dtype + "\""
                    + ",\"shape\":" + Arrays.toString(shape).replace(" ", "")
                    + ",\"data_offsets\":" + Arrays.toString(dataOffsets).replace(" ", "")
                    + "}";
        }
    }
    // end::header-type[]

    // tag::write[]
    // writeSafetensors는 텐서들을 safetensors 파일로 저장한다.
    // 파일 구조: [8바이트 헤더 길이(리틀 엔디언)] [JSON 헤더] [텐서 데이터]
    static void writeSafetensors(Path path, Map<String, float[]> tensors, Map<String, int[]> shapes)
            throws IOException {
        // 텐서 이름을 정렬해서 데이터 배치 순서를 고정한다.
        List<String> names = new ArrayList<>(tensors.keySet());
        Collections.sort(names);

        // 헤더(JSON)를 만들면서 각 텐서의 바이트 위치를 계산한다.
        StringBuilder header = new StringBuilder("{\"__metadata__\":{\"format\":\"java-example\"}");
        long offset = 0;
        ByteArrayOutputStream data = new ByteArrayOutputStream();
        for (String name : names) {
            float[] values = tensors.get(name);
            int byteLen = values.length * 4; // F32는 4바이트
            TensorInfo info = new TensorInfo("F32", shapes.get(name), new long[] {offset, offset + byteLen});
            header.append(",\"").append(name).append("\":").append(info.toJson());
            // float를 리틀 엔디언 바이트로 이어 붙인다.
            ByteBuffer buf = ByteBuffer.allocate(byteLen).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : values) {
                buf.putFloat(v);
            }
            data.write(buf.array());
            offset += byteLen;
        }
        header.append('}');
        byte[] headerJson = header.toString().getBytes(StandardCharsets.UTF_8);

        // 처음 8바이트: 헤더 JSON의 길이
        ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        lenBuf.putLong(headerJson.length);

        ByteArrayOutputStream file = new ByteArrayOutputStream();
        file.write(lenBuf.array());
        file.write(headerJson);
        file.write(data.toByteArray());
        Files.write(path, file.toByteArray());
    }
    // end::write[]

    // tag::read[]
    // readSafetensors는 safetensors 파일의 헤더를 해석하고
    // 각 텐서의 이름, 자료형, 모양, 값을 읽어 온다.
    static void readSafetensors(Path path) throws IOException {
        byte[] raw = Files.readAllBytes(path);

        // 1) 처음 8바이트에서 헤더 길이를 읽는다.
        long headerLen = ByteBuffer.wrap(raw, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();

        // 2) 헤더는 순수한 JSON이라 어떤 언어에서도 해석할 수 있다.
        //    pickle 기반 포맷과 달리 코드 실행 없이 읽기만 하므로 안전하다.
        String headerJson = new String(raw, 8, (int) headerLen, StandardCharsets.UTF_8);
        @SuppressWarnings("unchecked")
        Map<String, Object> header = (Map<String, Object>) Json.parse(headerJson);

        int dataStart = 8 + (int) headerLen;
        System.out.printf("헤더 길이: %d바이트, 데이터 영역: %d바이트%n%n", headerLen, raw.length - dataStart);

        List<String> names = new ArrayList<>(header.keySet());
        names.remove("__metadata__");
        Collections.sort(names);

        // 3) 각 텐서를 헤더의 오프셋 정보로 찾아 float로 복원한다.
        for (String name : names) {
            @SuppressWarnings("unchecked")
            TensorInfo info = TensorInfo.fromJson((Map<String, Object>) header.get(name));
            int begin = (int) info.dataOffsets()[0];
            int end = (int) info.dataOffsets()[1];
            ByteBuffer bytes = ByteBuffer.wrap(raw, dataStart + begin, end - begin).order(ByteOrder.LITTLE_ENDIAN);
            float[] values = new float[(end - begin) / 4];
            for (int i = 0; i < values.length; i++) {
                values[i] = bytes.getFloat();
            }
            System.out.printf("텐서 \"%s\"  dtype=%s shape=%s%n  값: %s%n",
                    name, info.dtype(), Arrays.toString(info.shape()), Arrays.toString(values));
        }
    }
    // end::read[]

    public static void main(String[] args) throws IOException {
        Path path = Path.of("model.safetensors");

        // 아주 작은 "모델"을 저장한다. 실제 LLM이라면 이 텐서가 수백 개,
        // 원소 수가 수십억 개일 뿐 파일 구조는 같다.
        Map<String, float[]> tensors = Map.of(
                "embedding.weight", new float[] {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f, 1.0f, -1.1f, 1.2f},
                "lm_head.bias", new float[] {0.01f, 0.02f, 0.03f});
        Map<String, int[]> shapes = Map.of(
                "embedding.weight", new int[] {4, 3}, // 어휘 4개, 임베딩 차원 3
                "lm_head.bias", new int[] {3});

        writeSafetensors(path, tensors, shapes);
        System.out.printf("%s 저장 완료%n%n", path);

        readSafetensors(path);
    }
}
