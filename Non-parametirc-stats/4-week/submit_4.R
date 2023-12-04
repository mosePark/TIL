# 5장 3번
'''
casl_glm_nr 함수가 아닌 casl_glm_nr_logistic 함수에 대하여 할 것(GLM이 아닌 로지스틱의 경우만 고려)
'''
## glm_logistic
casl_glm_nr_logistic <-function(X, y, maxit=25L, tol=1e-10)
{
  beta <- rep(0,ncol(X)) # beta 초기값
  for(j in seq(1L, maxit)) # 실험
  {
    b_old <- beta
    p <- 1 / (1 + exp(- X %*% beta))
    W <- as.numeric(p * (1 - p)) # digonal matrix, 식 5.20
    XtX <- crossprod(X, diag(W) %*% X) # 식 5.19, inverse of H
    score <- t(X) %*% (y - p) # 식 5.14
    delta <- solve(XtX, score)
    beta <- beta + delta
    if(sqrt(crossprod(beta - b_old)) < tol) break
  }
  beta
}
#################################################
casl_glm_gd_logistic <- function(X, y, maxit=100, tol = 1e-10){
  beta <- rep(0, ncol(X))
  
  for(j in 1:maxit){
    lam = 0.0001
    p <- 1 / (1 + exp(- X %*% beta))
    b_old <- beta
    score <- -t(X) %*% (y - p)
    beta <- b_old - lam * score #(gradient of log likelihood)
    
    if (sqrt(sum(beta - b_old)^2) < tol) break
  }
  return(beta)
}

n <- 1000; p <- 3
beta <- c(0.2, 2, 1)
X <- cbind(1, matrix(rnorm(n * (p- 1)), ncol = p - 1))
mu <- 1 / (1 + exp(-X %*% beta))
y <- matrix(as.numeric(runif(n) > mu))

cbind(casl_glm_gd_logistic(X, y), casl_glm_nr_logistic(X, y))



# 5장 5번
'''
casl_glm_irwls 함수가 아닌 casl_glm_nr_logistic 함수에 대하여 할 것(GLM이 아닌 로지스틱의 경우만 고려)
'''
casl_glm_nr_logistic_L2 <-function(X, y, maxit=25L, tol=1e-10, lambda){
  beta <- rep(0,ncol(X))
  for(j in seq(1L, maxit))
  {
    b_old <- beta
    p <- 1 / (1 + exp(- X %*% beta))
    W <- as.numeric(p * (1 - p))
    XtX <- crossprod(X, diag(W) %*% X)
    score <- t(X) %*% (y - p) - 2 * lambda * beta
    delta <- solve(XtX, score)
    beta <- beta + delta
    if(sqrt(crossprod(beta - b_old)) < tol) break
  }
  beta
}

cbind(casl_glm_nr_logistic_L2(X, y, lambda = 0.1), casl_glm_nr_logistic(X, y))
