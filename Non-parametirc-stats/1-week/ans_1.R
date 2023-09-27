###################################################################################
def ordinary least squre gradient descent 짜기
###################################################################################

# Sum of Squre Error

lm_sse = function(X, y, beta)
{
  sum((y - as.vector(X %*% beta))^2)
}

# ordinary least squre gradient descent

ols_grad_descent = function(X, y)
{
  n = nrow(X); p = ncol(X)

  maxiter = 100
  beta.old = rep(0, p)
  for (i in 1:maxiter) {
    lambda = 1
    for (k in 1:20) {
      beta.new = beta.old + lambda * 2 * t(X) %*% (y - as.vector(X %*% beta.old))
      if (lm_sse(X, y, beta.new) < lm_sse(X, y, beta.old)) break
      lambda = lambda / 2
    }
    beta.old = beta.new
  }
  beta.old
}


set.seed(1)
n=1e4; p=4
X = matrix(rnorm(n*p), ncol=p)
beta = c(1,2,3,4)
epsilon = rnorm(n)
y = X %*% beta + epsilon

# OLS based on QR decomposition from packages "casl"
beta_h_qr = casl_ols_orth_proj(X, y)
beta_h_qr

# OLS based on GD
beta_gd = ols_grad_descent(X, y)
beta_gd

###################################################################################
인자 eps, max_iter 추가
###################################################################################

ols_grad_descent_refined = function(X, y, eps = 1e-4, max_iter = 100)
{
  n = nrow(X); p = ncol(X)

  beta.old = rep(0, p)
  for (i in 1:max_iter) {
    lambda = 1
    for (k in 1:20) {
      beta.new = beta.old + lambda * 2 * t(X) %*% (y - as.vector(X %*% beta.old))
      if (lm_sse(X, y, beta.new) < lm_sse(X, y, beta.old)) break
      lambda = lambda / 2
    }
    if (max(abs(beta.old - beta.new)) < eps) 
    {
      print(i)
      break
    }
    beta.old = beta.new
  }
  beta.old
}

# OLS based on the refined version of GD
beta_gd_refined = ols_grad_descent_refined(X, y, 1e-7)

beta_gd_refined # beta_hat 출력
