package main

import (
	"fmt"
	"math"
	"strings"
)

// checker는 부록이 본문에 적어 둔 값과 라이브러리가 계산한 값을 비교하고
// 그 결과를 한 줄씩 출력한다. 마지막에 통과와 실패 개수를 집계한다.
type checker struct {
	pass, fail int
}

func (c *checker) section(title string) {
	fmt.Printf("\n== %s\n", title)
}

// near는 두 값이 허용 오차 안에서 같은지 확인한다.
// want에는 부록 본문에 적힌 값을, got에는 라이브러리 계산값을 넣는다.
func (c *checker) near(label string, want, got, tol float64) {
	c.mark(math.Abs(want-got) <= tol, label,
		fmt.Sprintf("부록 %-12s 계산 %.6g", fmtNum(want), got))
}

// ok는 수치 비교가 아니라 참·거짓으로 판정하는 주장을 확인한다.
func (c *checker) ok(label string, cond bool, detail string) {
	c.mark(cond, label, detail)
}

// note는 판정 없이 계산 결과만 보여 준다.
func (c *checker) note(format string, a ...any) {
	fmt.Printf("         %s\n", fmt.Sprintf(format, a...))
}

func (c *checker) mark(ok bool, label, detail string) {
	tag := "OK  "
	if ok {
		c.pass++
	} else {
		c.fail++
		tag = "FAIL"
	}
	fmt.Printf("  [%s] %s %s\n", tag, pad(label, 32), detail)
}

func (c *checker) summary() bool {
	fmt.Printf("\n검산 %d건 중 통과 %d건, 실패 %d건\n", c.pass+c.fail, c.pass, c.fail)
	return c.fail == 0
}

// fmtNum은 아주 작거나 아주 큰 값만 지수 표기로 바꾸고 나머지는 소수로 보여 준다.
func fmtNum(v float64) string {
	if a := math.Abs(v); v != 0 && (a < 1e-4 || a >= 1e9) {
		return fmt.Sprintf("%.3e", v)
	}
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.6f", v), "0"), ".")
}

// pad는 한글을 두 칸으로 세어 라벨 폭을 맞춘다.
func pad(s string, width int) string {
	w := 0
	for _, r := range s {
		if isWide(r) {
			w += 2
		} else {
			w++
		}
	}
	if w >= width {
		return s
	}
	return s + strings.Repeat(" ", width-w)
}

func isWide(r rune) bool {
	switch {
	case r >= 0x1100 && r <= 0x115F, // 한글 자모
		r >= 0x2E80 && r <= 0x303E, // 한중일 부수, 구두점
		r >= 0x3041 && r <= 0x33FF, // 가나, 한글 호환 자모, 기호
		r >= 0x3400 && r <= 0x4DBF, // 한자 확장 A
		r >= 0x4E00 && r <= 0x9FFF, // 한자
		r >= 0xAC00 && r <= 0xD7A3, // 한글 음절
		r >= 0xF900 && r <= 0xFAFF, // 한자 호환
		r >= 0xFF00 && r <= 0xFF60: // 전각 영숫자
		return true
	}
	return false
}
