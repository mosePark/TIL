# 5장 3번
'''
casl_glm_nr 함수가 아닌 casl_glm_nr_logistic 함수에 대하여 할 것(GLM이 아닌 로지스틱의 경우만 고려)
'''
## glm_logistic
casl_glm_nr_logistic <-
  function(X, y, maxit=25L, tol=1e-10)
  {
    beta <- rep(0,ncol(X)) # beta 초기값
    for(j in seq(1L, maxit)) # 실험
    {
      b_old <- beta
      p <- 1 / (1 + exp(- X %*% beta))
      W <- as.numeric(p * (1 - p)) # digonal matrix, 식 5.20
      XtX <- crossprod(X, diag(W) %*% X) # 식 5.19
      score <- t(X) %*% (y - p) # 식 5.14
      delta <- solve(XtX, score)
      beta <- beta + delta
      if(sqrt(crossprod(beta - b_old)) < tol) break
    }
    beta
  }

casl_glm_gd_logistic <- 
  function(X, y, maxit=25L)
  {
    beta <- rep(0, ncol(X))
    for(j in seq(1L, maxit))
    {
      b_old <- beta
      b_new <- b_old - lam * #(gradient of log likelihood b_old)
      
      
      
      
    }
    beta
  }


## 데이터
n <- 1000; p <- 3
beta <- c(0.2, 2, 1)
X <- cbind(1, matrix(rnorm(n * (p- 1)), ncol = p - 1))
mu <- 1 / (1 + exp(-X %*% beta))
y <- as.numeric(runif(n) > mu)

## 비교


# 5장 5번
'''
casl_glm_irwls 함수가 아닌 casl_glm_nr_logistic 함수에 대하여 할 것(GLM이 아닌 로지스틱의 경우만 고려)
'''
