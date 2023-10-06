'''
세 함수에 대하여 비교하기 위해 X와 y 모두 표준화를 먼저 적용함. MASS의 lm.ridge는 모형식에서 -1로 상수항을 제거하면 casl_lm_ridge와 비슷한 결과를 얻음.
glmnet의 경우 alpha=0으로 놓으면 목적함수가 RSS2n+λ2∥β∥2 이 되고 결국 RSS+nλ∥β∥2
에 비례하며 intercept=FALSE로 놓고 s=lambda /n으로 설정하면 앞의 두 함수와 대략 비슷한 계수추정치를 얻음
'''

#############################################################
## 3.1 ######################################################
#############################################################
set.seed(20230101)
n = 200; p = 4
beta = c(1, -1, 0.5, 0)
mu = rep(0, p)
Sigma = matrix(0.9, p, p)
diag(Sigma) = 1
X = MASS::mvrnorm(n, mu, Sigma)
y = X %*% beta + rnorm(n, sd=5)
X.s = scale(X)
y.s = scale(y)

lam = 150

library(MASS)
library(glmnet)

# No scaling
casl_lm_ridge(X, y, lam)

fit = lm.ridge(y~X-1, lambda=lam)
fit

out=glmnet(X, y, alpha=0, lambda=lam/n, intercept=FALSE)
coef(out)

# X: scaled
casl_lm_ridge(X.s, y, lam)

# X, Y: scaled
casl_lm_ridge(X.s, y.s, lam)

fit = lm.ridge(y.s~X.s-1, lambda=lam)
fit

out=glmnet(X.s, y.s, alpha=0, lambda=lam/n, intercept=FALSE)
coef(out)

#############################################################
## 3.4 ######################################################
#############################################################

cv.ridge_reg = function(X, y, nfolds=10, N=100)
{
  N = 100 # num of grid points
  n = length(y)
  
  fold = sample(1:nfolds, n, replace=T)
  
  # grid for lambda
  beta.ols = coef(lm(y~X-1))
  l2.ols = sum(beta.ols^2)
  print(l2.ols)
  lambda_vals = log(seq(exp(0), exp(1), length.out=N)) * l2.ols
  cv.mse = rep(0, ncol = length(lambda_vals))
  for (i in 1:nfolds) 
  {
    X.tr = X[fold != i, ]
    y.tr = y[fold != i]
    beta_mat = casl_lm_ridge(X.tr, y.tr, lambda_vals)
    
    X.te = X[fold == i, ]
    y.te = y[fold == i]
    y_hat = tcrossprod(X.te, beta_mat)
    cv.mse = cv.mse + apply((y_hat - y.te)^2, 2, mean) / nfolds
  } 
  return(lambda_vals[which.min(cv.mse)])
}



opt.lambda = cv.ridge_reg(X, y)


opt.lambda


casl_lm_ridge(X, y, opt.lambda)
