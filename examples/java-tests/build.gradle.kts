// examples/java/ 아래 순수 자바 예제들의 JUnit 테스트 프로젝트.
// 예제 소스는 그대로 두고(빌드 파일 없이 java Main.java로 실행하는 원칙을 유지한다),
// 하위 프로젝트마다 소스 디렉터리를 ../java/<이름>으로 지정해 컴파일한다.
// 예제 클래스들이 package 선언 없이 기본 패키지에 있으므로 테스트도 기본 패키지에 둔다.

subprojects {
    apply(plugin = "java")

    repositories {
        mavenCentral()
    }

    configure<JavaPluginExtension> {
        sourceSets["main"].java.setSrcDirs(listOf(rootDir.resolve("../java/${project.name}")))
    }

    dependencies {
        "testImplementation"(platform("org.junit:junit-bom:5.14.4"))
        "testImplementation"("org.junit.jupiter:junit-jupiter")
        "testRuntimeOnly"("org.junit.platform:junit-platform-launcher")
        "testImplementation"("org.assertj:assertj-core:3.27.6")
    }

    tasks.withType<JavaCompile>().configureEach {
        options.encoding = "UTF-8"
        options.release = 21
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
        testLogging {
            events("passed", "failed", "skipped")
        }
    }
}
