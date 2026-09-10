# safetensors 파서

safetensors 파일을 JDK 표준 라이브러리만으로 직접 만들고 다시 읽어 보는 예제다. 파일 구조는 세 부분이다. 처음 8바이트에 헤더 길이(리틀 엔디언), 이어서 JSON 헤더, 그 뒤에 텐서 데이터가 이어진다. 어휘 4개, 임베딩 차원 3인 장난감 모델을 저장하고 복원한다.

## 실행 방법

```bash
java Main.java
```

JDK 22 이상이면 컴파일 없이 바로 실행된다. 실행하면 현재 디렉터리에 `model.safetensors` 파일이 생긴다.

## 실행 결과

```text
model.safetensors 저장 완료

헤더 길이: 179바이트, 데이터 영역: 60바이트

텐서 "embedding.weight"  dtype=F32 shape=[4, 3]
  값: [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2]
텐서 "lm_head.bias"  dtype=F32 shape=[3]
  값: [0.01, 0.02, 0.03]
```

## 파일 구성

| 파일 | 내용 |
|------|------|
| `Main.java` | 텐서 메타데이터 `TensorInfo`, 파일을 쓰는 `writeSafetensors`, 읽는 `readSafetensors` |
| `Json.java` | 헤더를 해석하는 최소 JSON 파서. JDK에는 JSON 파서가 없어서 예제에 필요한 만큼만 직접 만들었다. |

파서 어디에도 코드 실행이 없다. JSON과 바이트 오프셋만 다루면 어떤 언어로도 읽고 쓸 수 있다는 것이 이 포맷이 표준이 된 이유다.
