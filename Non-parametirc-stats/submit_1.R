## Gradient Descent
# 연습문제 2.2

install.packages("casl")
library(casl)

set.seed(1)
n=1e4; p=4
X = matrix(rnorm(n*p), ncol=p)
beta = c(1,2,3,4)
epsilon = rnorm(n)
y = X %*% beta + epsilon

# beta_h_qr = casl_ols_orth_proj(X, y) # QR분해기반 OLS, 비교 대상
# beta_h_qr

b_old = c(0, 0, 0, 0) # init value
lam = 0.0001 # learing rate

ols_grad_descent = function(X, y) {
  
  grad = lam * t(X) %*% (y - (X %*% b_old))
  b_new = b_old + grad
  
  return(b_new)
}

iter = 100
for (i in 0:iter){
  
  ols_grad_descent(X, y)
  
  }
}

print(ols_grad_descent(X, y))


# 연습문제 2.3
eps = 10^-16
max_iter = 1000
idx = 0
lam = 0.0001

ols_grad_descent = function(X, y, eps, max_iter) {
  b_old = rep(0, ncol(X))
  ans = 0
  iterations = 0  # 반복 횟수를 추적하기 위한 변수 추가
  
  for (i in 1:max_iter) {
    iterations = i  # 반복 횟수 증가
    grad = lam * t(X) %*% (y - (X %*% b_old))
    b_new = b_old + grad
    
    max_change = max(abs(b_new - b_old))  # 최대 변경량 계산
    
    if (max_change < eps) {
      break  # `∞-norm` 변화가 `eps`보다 작으면 반복 종료
    }
    
    b_old = b_new
  }
  
  # 반복 횟수와 결과를 출력
  cat("Number of iter:", iterations, "\n")
  cat("Final coef:", b_new, "\n")
}

# 예제 데이터 생성 (X, y를 생성해야 함)
# X와 y를 설정해야 ols_grad_descent 함수 호출 가능

# 예제 데이터로 함수 호출
ols_grad_descent(X, y, eps, max_iter)
