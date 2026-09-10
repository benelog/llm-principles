import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Json은 safetensors 헤더를 읽기 위한 최소 파서다. 객체는 Map, 배열은 List,
 * 숫자는 정수면 Long, 소수면 Double로 돌려준다.
 */
class JsonTest {

    @Test
    @DisplayName("safetensors 헤더 형태의 객체를 Map으로 읽는다")
    void parsesSafetensorsHeader() {
        String header = "{\"__metadata__\":{\"format\":\"pt\"},"
                + "\"embedding.weight\":{\"dtype\":\"F32\",\"shape\":[4,3],\"data_offsets\":[0,48]}}";

        @SuppressWarnings("unchecked")
        Map<String, Object> parsed = (Map<String, Object>) Json.parse(header);

        assertThat(parsed).containsOnlyKeys("__metadata__", "embedding.weight");
        Map<?, ?> tensor = (Map<?, ?>) parsed.get("embedding.weight");
        assertThat(tensor.get("dtype")).isEqualTo("F32");
        assertThat(tensor.get("shape")).isEqualTo(List.of(4L, 3L));
        assertThat(tensor.get("data_offsets")).isEqualTo(List.of(0L, 48L));
    }

    @Test
    @DisplayName("정수는 Long, 소수와 지수 표기는 Double로 읽는다")
    void numbersBecomeLongOrDouble() {
        Object numbers = Json.parse("[1, -2, 3.5, 1e3]");

        assertThat(numbers).isEqualTo(List.of(1L, -2L, 3.5, 1000.0));
    }

    @Test
    @DisplayName("문자열 이스케이프와 유니코드를 해석한다")
    void decodesEscapes() {
        Object value = Json.parse("\"a\\\"b\\\\c\\n\\ud55c\"");

        assertThat(value).isEqualTo("a\"b\\c\n한");
    }

    @Test
    @DisplayName("true, false, null과 빈 객체·배열, 공백을 처리한다")
    void handlesLiteralsAndEmptyContainers() {
        Map<?, ?> parsed = (Map<?, ?>) Json.parse(" { \"t\" : true , \"f\":false, \"n\":null, \"o\":{}, \"a\":[ ] } ");

        assertThat(parsed.get("t")).isEqualTo(Boolean.TRUE);
        assertThat(parsed.get("f")).isEqualTo(Boolean.FALSE);
        assertThat(parsed.get("n")).isNull();
        assertThat((Map<?, ?>) parsed.get("o")).isEmpty();
        assertThat((List<?>) parsed.get("a")).isEmpty();
    }

    @Test
    @DisplayName("문서 끝 뒤에 문자가 남거나 구조가 깨지면 위치를 담은 예외를 던진다")
    void rejectsMalformedInput() {
        assertThatThrownBy(() -> Json.parse("{} x"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("문서 끝");
        assertThatThrownBy(() -> Json.parse("{\"a\" 1}"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("위치");
        assertThatThrownBy(() -> Json.parse("[1, 2"))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
