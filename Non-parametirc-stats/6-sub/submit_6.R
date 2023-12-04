'''lars 패키지와 biglasso 패키의 lasso 회귀 결과와 시간 및 메모리 사용량을 비교하시오.'''

set.seed(123)

library(lars)
library(biglasso)
library(profmem)

# 데이터
n <- 1000L
p <- 5000L
X <- matrix(rnorm(n * p), ncol = p)
beta <- c(seq(1, 0.1, length.out=(10L)), rep(0, p - 10L))
y <- X %*% beta + rnorm(n = n, sd = 0.15)



# 시간 비교
lars_time <- system.time(
  
  lars(X, y , type = "lasso")
  
)

X <- as.big.matrix(X)

biglasso_time <- system.time(
  
  biglasso::biglasso(X, y, penalty = "lasso")
  
)


# 메모리 비교
lars_mem <- profmem(
  
  lars(X, y , type = "lasso")
  
)

big_mem <- profmem(
  
  biglasso::biglasso(X, y, penalty = "lasso")
  
)


total(lars_mem)
total(big_mem)
