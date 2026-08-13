// safetensors 포맷을 직접 만들어 보고 다시 읽어 보는 예제.
// 외부 라이브러리 없이 표준 라이브러리만 사용한다.
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
)

// tag::header-type[]
// TensorInfo는 safetensors 헤더에 기록되는 텐서 하나의 메타데이터다.
// 실제 숫자값은 헤더 뒤의 바이너리 영역에 있고,
// 헤더는 그 값을 어떻게 해석할지(자료형, 모양, 위치)만 알려 준다.
type TensorInfo struct {
	DType       string  `json:"dtype"`        // 예: "F32"
	Shape       []int   `json:"shape"`        // 예: [4, 3]
	DataOffsets [2]int  `json:"data_offsets"` // 데이터 영역 안에서의 [시작, 끝] 바이트 위치
}
// end::header-type[]

// tag::write[]
// writeSafetensors는 텐서들을 safetensors 파일로 저장한다.
// 파일 구조: [8바이트 헤더 길이(리틀 엔디언)] [JSON 헤더] [텐서 데이터]
func writeSafetensors(path string, tensors map[string][]float32, shapes map[string][]int) error {
	// 텐서 이름을 정렬해서 데이터 배치 순서를 고정한다.
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	// 헤더(JSON)를 만들면서 각 텐서의 바이트 위치를 계산한다.
	header := map[string]any{
		"__metadata__": map[string]string{"format": "go-example"},
	}
	offset := 0
	var data []byte
	for _, name := range names {
		values := tensors[name]
		byteLen := len(values) * 4 // F32는 4바이트
		header[name] = TensorInfo{
			DType:       "F32",
			Shape:       shapes[name],
			DataOffsets: [2]int{offset, offset + byteLen},
		}
		// float32를 리틀 엔디언 바이트로 이어 붙인다.
		buf := make([]byte, byteLen)
		for i, v := range values {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		data = append(data, buf...)
		offset += byteLen
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// 처음 8바이트: 헤더 JSON의 길이
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(headerJSON); err != nil {
		return err
	}
	_, err = f.Write(data)
	return err
}
// end::write[]

// tag::read[]
// readSafetensors는 safetensors 파일의 헤더를 해석하고
// 각 텐서의 이름, 자료형, 모양, 값을 읽어 온다.
func readSafetensors(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	// 1) 처음 8바이트에서 헤더 길이를 읽는다.
	headerLen := binary.LittleEndian.Uint64(raw[:8])

	// 2) 헤더는 순수한 JSON이라 어떤 언어에서도 해석할 수 있다.
	//    pickle 기반 포맷과 달리 코드 실행 없이 읽기만 하므로 안전하다.
	headerJSON := raw[8 : 8+headerLen]
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return err
	}

	dataStart := 8 + int(headerLen)
	fmt.Printf("헤더 길이: %d바이트, 데이터 영역: %d바이트\n\n", headerLen, len(raw)-dataStart)

	names := make([]string, 0, len(header))
	for name := range header {
		if name != "__metadata__" {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	// 3) 각 텐서를 헤더의 오프셋 정보로 찾아 float32로 복원한다.
	for _, name := range names {
		var info TensorInfo
		if err := json.Unmarshal(header[name], &info); err != nil {
			return err
		}
		begin, end := info.DataOffsets[0], info.DataOffsets[1]
		bytes := raw[dataStart+begin : dataStart+end]
		values := make([]float32, len(bytes)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
		}
		fmt.Printf("텐서 %q  dtype=%s shape=%v\n  값: %v\n", name, info.DType, info.Shape, values)
	}
	return nil
}
// end::read[]

func main() {
	path := "model.safetensors"

	// 아주 작은 "모델"을 저장한다. 실제 LLM이라면 이 텐서가 수백 개,
	// 원소 수가 수십억 개일 뿐 파일 구조는 같다.
	tensors := map[string][]float32{
		"embedding.weight": {0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2},
		"lm_head.bias":     {0.01, 0.02, 0.03},
	}
	shapes := map[string][]int{
		"embedding.weight": {4, 3}, // 어휘 4개, 임베딩 차원 3
		"lm_head.bias":     {3},
	}

	if err := writeSafetensors(path, tensors, shapes); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s 저장 완료\n\n", path)

	if err := readSafetensors(path); err != nil {
		log.Fatal(err)
	}
}
