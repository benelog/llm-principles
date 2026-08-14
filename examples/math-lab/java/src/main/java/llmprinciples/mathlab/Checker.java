package llmprinciples.mathlab;

/**
 * 부록이 본문에 적어 둔 값과 라이브러리가 계산한 값을 비교하고 결과를 한 줄씩 출력한다.
 * 마지막에 통과와 실패 개수를 집계한다.
 */
public class Checker {

    private int pass;
    private int fail;

    public void section(String title) {
        System.out.printf("%n== %s%n", title);
    }

    /**
     * 두 값이 허용 오차 안에서 같은지 확인한다.
     * expected에는 부록 본문에 적힌 값을, actual에는 라이브러리 계산값을 넣는다.
     */
    public void near(String label, double expected, double actual, double tolerance) {
        mark(Math.abs(expected - actual) <= tolerance, label,
                String.format("부록 %-12s 계산 %s", num(expected), num(actual)));
    }

    /** 수치 비교가 아니라 참·거짓으로 판정하는 주장을 확인한다. */
    public void ok(String label, boolean condition, String detail) {
        mark(condition, label, detail);
    }

    /** 판정 없이 계산 결과만 보여 준다. */
    public void note(String format, Object... args) {
        System.out.printf("         %s%n", String.format(format, args));
    }

    private void mark(boolean ok, String label, String detail) {
        String tag;
        if (ok) {
            pass++;
            tag = "OK  ";
        } else {
            fail++;
            tag = "FAIL";
        }
        System.out.printf("  [%s] %s %s%n", tag, pad(label, 32), detail);
    }

    public boolean summary() {
        System.out.printf("%n검산 %d건 중 통과 %d건, 실패 %d건%n", pass + fail, pass, fail);
        return fail == 0;
    }

    /** 아주 작거나 아주 큰 값만 지수 표기로 바꾸고 나머지는 소수로 보여 준다. */
    public static String num(double v) {
        double a = Math.abs(v);
        if (v != 0 && (a < 1e-4 || a >= 1e9)) {
            return String.format("%.3e", v);
        }
        String s = String.format("%.6f", v);
        s = s.replaceAll("0+$", "");
        return s.endsWith(".") ? s.substring(0, s.length() - 1) : s;
    }

    /** 한글을 두 칸으로 세어 라벨 폭을 맞춘다. */
    private static String pad(String s, int width) {
        int w = 0;
        for (int i = 0; i < s.length(); i++) {
            w += isWide(s.charAt(i)) ? 2 : 1;
        }
        return w >= width ? s : s + " ".repeat(width - w);
    }

    private static boolean isWide(char ch) {
        return (ch >= 0x1100 && ch <= 0x115F)   // 한글 자모
                || (ch >= 0x2E80 && ch <= 0x303E) // 한중일 부수, 구두점
                || (ch >= 0x3041 && ch <= 0x33FF) // 가나, 한글 호환 자모, 기호
                || (ch >= 0x3400 && ch <= 0x4DBF) // 한자 확장 A
                || (ch >= 0x4E00 && ch <= 0x9FFF) // 한자
                || (ch >= 0xAC00 && ch <= 0xD7A3) // 한글 음절
                || (ch >= 0xF900 && ch <= 0xFAFF) // 한자 호환
                || (ch >= 0xFF00 && ch <= 0xFF60); // 전각 영숫자
    }
}
