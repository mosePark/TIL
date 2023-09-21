library(MASS)
library(glmnet)
library(caret)

###############################################################################
''' 스케일, 절편 및 패널티 처리에 주의하여,
함수 casl_lm_ridge가 특정 λ 값에 대해
MASS::lm.ridge 및 glmnet::glmnet과 유사한 결과를 생성하는지 확인하십시오.
'''
###############################################################################
# def of casl_lm_ridge
casl_lm_ridge <-function(X, y, lambda_vals)
  {
    svd_obj <- svd(X)
    U <- svd_obj$u
    V <- svd_obj$v
    svals <- svd_obj$d
    k <- length(lambda_vals)
    ridge_beta <- matrix(NA_real_, nrow = k, ncol = ncol(X))
    for (j in seq_len(k))
    {
      D <- diag(svals / (svals^2 + lambda_vals[j]))
      ridge_beta[j,] <- V %*% D %*% t(U) %*% y
    }
    ridge_beta
  }

# dataset
n <- 200; p <- 4; N <- 500; M <- 20
beta <- c(1, -1, 0.5, 0)
mu <- rep(0, p)
Sigma <- matrix(0.9, nrow = p, ncol = p)
diag(Sigma) <- 1
X <- MASS::mvrnorm(n, mu, Sigma)
y <- X %*% beta + rnorm(n, sd = 5)

# lambda tuning
lam = 0.0001

# Modeling
MASS_ridge <- MASS::lm.ridge(y ~ X, lambda = lam)
glmnet_ridge <- glmnet::glmnet(X, y, lambda = lam)

#  beta hat of casl, MASS and glmnet
casl_lm_ridge(X, y, lambda_vals = lam)

MASS_ridge$coef
glmnet_ridge$beta
