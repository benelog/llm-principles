import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * safetensors 파일은 [8바이트 헤더 길이][JSON 헤더][텐서 데이터] 세 부분이다.
 * writeSafetensors가 만든 파일을 바이트 단위로 뜯어 보면서 이 구조를 확인하고,
 * readSafetensors가 같은 파일에서 텐서를 복원하는지 본다.
 */
class SafetensorsTest {

    @TempDir
    Path dir;

    private static final Map<String, float[]> TENSORS = Map.of(
            "embedding.weight", new float[] {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f},
            "lm_head.bias", new float[] {0.01f, 0.02f, 0.03f});
    private static final Map<String, int[]> SHAPES = Map.of(
            "embedding.weight", new int[] {2, 3},
            "lm_head.bias", new int[] {3});

    @Test
    @DisplayName("처음 8바이트는 리틀 엔디언으로 적은 JSON 헤더의 길이다")
    void firstEightBytesHoldHeaderLength() throws Exception {
        Path file = dir.resolve("model.safetensors");
        Main.writeSafetensors(file, TENSORS, SHAPES);
        byte[] raw = Files.readAllBytes(file);

        long headerLen = ByteBuffer.wrap(raw, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        String header = new String(raw, 8, (int) headerLen, StandardCharsets.UTF_8);

        assertThat(header).startsWith("{").endsWith("}");
        assertThat(raw.length).as("헤더 뒤에 F32 9개 = 36바이트가 이어진다").isEqualTo(8 + headerLen + 9 * 4);
    }

    @Test
    @DisplayName("헤더는 순수한 JSON이라 예제의 파서로 텐서 이름, 자료형, 모양, 오프셋을 읽을 수 있다")
    void headerDescribesEveryTensor() throws Exception {
        Path file = dir.resolve("model.safetensors");
        Main.writeSafetensors(file, TENSORS, SHAPES);

        Map<String, Object> header = readHeader(file);

        assertThat(header).containsKeys("__metadata__", "embedding.weight", "lm_head.bias");
        @SuppressWarnings("unchecked")
        Main.TensorInfo info = Main.TensorInfo.fromJson((Map<String, Object>) header.get("embedding.weight"));
        assertThat(info.dtype()).isEqualTo("F32");
        assertThat(info.shape()).containsExactly(2, 3);
        // 이름 정렬 순서로 배치하므로 embedding.weight가 데이터 영역의 맨 앞(0부터 24바이트)에 온다.
        assertThat(info.dataOffsets()).containsExactly(0L, 24L);
    }

    @Test
    @DisplayName("헤더의 오프셋으로 찾아간 바이트를 float로 읽으면 원래 값이 나온다")
    void tensorBytesRoundTrip() throws Exception {
        Path file = dir.resolve("model.safetensors");
        Main.writeSafetensors(file, TENSORS, SHAPES);
        byte[] raw = Files.readAllBytes(file);
        long headerLen = ByteBuffer.wrap(raw, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        int dataStart = 8 + (int) headerLen;
        @SuppressWarnings("unchecked")
        Main.TensorInfo info = Main.TensorInfo.fromJson((Map<String, Object>) readHeader(file).get("lm_head.bias"));

        ByteBuffer bytes = ByteBuffer.wrap(raw, dataStart + (int) info.dataOffsets()[0],
                (int) (info.dataOffsets()[1] - info.dataOffsets()[0])).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[3];
        for (int i = 0; i < values.length; i++) {
            values[i] = bytes.getFloat();
        }

        assertThat(values).containsExactly(0.01f, 0.02f, 0.03f);
    }

    @Test
    @DisplayName("TensorInfo의 toJson과 fromJson은 서로 역연산이다")
    void tensorInfoJsonRoundTrip() {
        Main.TensorInfo info = new Main.TensorInfo("F32", new int[] {4, 3}, new long[] {0, 48});

        String json = info.toJson();
        @SuppressWarnings("unchecked")
        Main.TensorInfo restored = Main.TensorInfo.fromJson((Map<String, Object>) Json.parse(json));

        assertThat(json).isEqualTo("{\"dtype\":\"F32\",\"shape\":[4,3],\"data_offsets\":[0,48]}");
        // 레코드의 equals는 배열을 참조로 비교하므로 내용 비교를 쓴다.
        assertThat(restored).usingRecursiveComparison()
                .isEqualTo(new Main.TensorInfo("F32", new int[] {4, 3}, new long[] {0, 48}));
        assertThat(restored.shape()).containsExactly(4, 3);
        assertThat(restored.dataOffsets()).containsExactly(0L, 48L);
    }

    @Test
    @DisplayName("readSafetensors는 파일을 읽어 텐서 이름과 값을 출력한다")
    void readSafetensorsPrintsTensors() throws Exception {
        Path file = dir.resolve("model.safetensors");
        Main.writeSafetensors(file, TENSORS, SHAPES);
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.out;
        System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8));
        try {
            Main.readSafetensors(file);
        } finally {
            System.setOut(original);
        }

        String output = captured.toString(StandardCharsets.UTF_8);
        assertThat(output).contains("\"embedding.weight\"", "dtype=F32", "shape=[2, 3]", "0.1, -0.2, 0.3");
        assertThat(output).contains("\"lm_head.bias\"", "0.01, 0.02, 0.03");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> readHeader(Path file) throws Exception {
        byte[] raw = Files.readAllBytes(file);
        long headerLen = ByteBuffer.wrap(raw, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        String header = new String(raw, 8, (int) headerLen, StandardCharsets.UTF_8);
        Object parsed = Json.parse(header);
        assertThat(parsed).as("헤더의 최상위는 객체").isInstanceOf(Map.class);
        return (Map<String, Object>) parsed;
    }
}
