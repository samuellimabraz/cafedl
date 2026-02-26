# Multi-stage build for CafeDL core library (headless, no JavaFX GUI)

FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /app

COPY pom.xml .
RUN mvn dependency:go-offline -B || true

# Copy source and build (skip tests during image build)
COPY src ./src
COPY .mvn ./.mvn
RUN mvn package -DskipTests -B

FROM eclipse-temurin:21-jre
WORKDIR /app

# Copy built artifact
COPY --from=build /app/target/*.jar ./app.jar
COPY --from=build /app/target/dependency/ ./lib/ 2>/dev/null || true

# JVM options for ND4J compatibility
ENV JAVA_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED \
               --add-opens java.base/java.lang.invoke=ALL-UNNAMED \
               --add-opens java.base/java.io=ALL-UNNAMED \
               --add-opens java.base/java.nio=ALL-UNNAMED \
               --add-opens java.base/java.util=ALL-UNNAMED"

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
