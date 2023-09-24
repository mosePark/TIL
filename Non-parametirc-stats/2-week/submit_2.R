library(MASS)
library(glmnet)
library(caret)
library(ggplot2)

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
glmnet_ridge <- glmnet::glmnet(X, y, lambda = lam, alpha = 0)

#  beta hat of casl, MASS and glmnet
casl_lm_ridge(X, y, lambda_vals = lam)

MASS_ridge$coef
glmnet_ridge$beta



###############################################################################
'''릿지 회귀에서 최적의 λ 값을 선택하기 위해
10-fold 교차 검증을 수행하는 cv.ridge_reg 함수를 만드세요.
'''
###############################################################################
cv.ridge_reg <- function(X, y, lambda_vals, k)
{
  results <- list()  # 결과를 저장할 리스트 초기화
  
  for (i in 1:k)
  {
    folds <- caret::createFolds(1:nrow(X), k = k)
    
    X_train <- X[-folds[[i]], ]
    X_test <- X[folds[[i]], ]
    y_train <- y[-folds[[i]], ]
    y_test <- y[folds[[i]], ]
    
    # Training
    svd_obj <- svd(X_train)
    U <- svd_obj$u
    V <- svd_obj$v
    svals <- svd_obj$d
    l <- length(lambda_vals)
    ridge_beta <- matrix(NA_real_, nrow = l, ncol = ncol(X))
    for (j in seq_len(l))
    {
      D <- diag(svals / (svals^2 + lambda_vals[j]))
      ridge_beta[j,] <- V %*% D %*% t(U) %*% y_train
    }
    
    # Test
    y_hat <- tcrossprod(X_test, ridge_beta)
    mse <- apply((y_hat - y_test)^2, 2, mean)
    
    # 결과를 리스트에 추가
    results[[i]] <- list(beta = ridge_beta, lam = lambda_vals[which.min(mse)], k = i)
  }
  
  return(results)  # 모든 폴드에 대한 결과를 반환
}

cv.ridge_reg(X, y, lambda_vals = c(0.1, 0.01, 0.001, 0.0001), k = 10)
