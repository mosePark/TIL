library(earth)
library(ggplot2)

# 데이터 생성 설정
set.seed(123)
n <- 100 # 샘플 크기

# 독립변수 생성
x1 <- runif(n, -2, 2)
x2 <- runif(n, -2, 2)
x3 <- runif(n, -2, 2)
x4 <- runif(n, -2, 2)
x5 <- runif(n, -2, 2)
x6 <- runif(n, -2, 2)

# 비선형 관계 및 잡음 추가
y <- 2 * x1^2 + 3 * log(abs(x2) + 1) + x3 * x4 - cos(x5) + sqrt(abs(x6)) + rnorm(n)

data <- data.frame(x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6, y = y)


# 비선형 구조
ggplot(data, aes(x = x1, y = y)) +
  geom_point(color = 'black') +
  ggtitle("x1 와 y의 비선형 구조") +
  xlab("x1") +
  ylab("y")


# train, test 분할
train_idx <- sample(1:n, size = 0.7 * n)
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

mars_model <- earth(y ~ ., data = train_data)

# 모델 요약
print(summary(mars_model))

# 테스트 데이터셋을 사용한 예측
predictions <- predict(mars_model, newdata = test_data)

# RMSE 계산
rmse <- sqrt(mean((predictions - test_data$y)^2))
print(paste("RMSE:", rmse))
