plugins {
    application
}

repositories {
    mavenCentral()
}

dependencies {
    // Apache Commons Math 하나로 부록의 선형대수, 분포, 회귀, 검정을 모두 다룬다.
    implementation("org.apache.commons:commons-math3:3.6.1")
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
}

application {
    mainClass = "llmprinciples.mathlab.Main"
}
