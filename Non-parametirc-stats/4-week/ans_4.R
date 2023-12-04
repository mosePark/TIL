# 5.3 newton rhapson vs GD
casl_glm_gd_logistic <-
function(X, y, maxit=2000L, tol=1e-5)
{
  beta <- rep(0,ncol(X))
  for(j in seq(1L, maxit))
  {
    b_old <- beta
    p = 1 / ( 1 + exp(-X %*% beta))
    score <- t(X) %*% (y - p)
    lambda = 0.0001
    beta <- beta + lambda * score      
    if(sqrt(crossprod(beta - b_old)) < tol) break
  }
  print(j)
  beta
}


set.seed(1)
n = 1000; p = 3
beta = c(0.2, 2, 1)
X = cbind(1, matrix(rnorm(n*(p-1)), ncol=p-1))
mu = 1 / (1+ exp(-X %*% beta))
y = as.numeric(runif(n) > mu)

beta = casl_glm_nr_logistic(X, y)
beta_gd = casl_glm_gd_logistic(X, y)

cbind(beta, as.numeric(beta_gd))

# 5.5 L2 norm setting, newton-rhapson method
casl_glm_pen_logistic <-
function(X, y, lambda = 1, maxit=100L, tol=1e-10)
{
  beta <- rep(0,ncol(X))
  for(j in seq(1L, maxit))
  {
    b_old <- beta
    p <- 1 / (1 + exp(- X %*% beta))
    W <- as.numeric(p * (1 - p))
    XtX <- crossprod(X, diag(W) %*% X) + 2 * lambda * diag(length(beta))
    score <- -t(X) %*% (y - p) + 2 * lambda * beta
    delta <- solve(XtX, score)
    beta <- beta - delta
    if(sqrt(crossprod(beta - b_old)) < tol) break
  }
  beta
}


set.seed(1)
n = 1000; p = 3
beta = c(0.2, 2, 1)
X = cbind(1, matrix(rnorm(n*(p-1)), ncol=p-1))
mu = 1 / (1+ exp(-X %*% beta))
y = as.numeric(runif(n) > mu)

beta = casl_glm_nr_logistic(X, y)
beta_pen1 = casl_glm_pen_logistic(X, y, lambda=1)
beta_pen1e4 = casl_glm_pen_logistic(X, y, lambda=1e4)
cbind(beta, as.numeric(beta_pen1), as.numeric(beta_pen1e4))
