# 2
'''
데이터 갯수가 1000개이고 변수가 500개인 모의실험에 대하여 lars와 biglasso의 해의 경로는 대략 비슷하게 나오며 biglasso가 실행속도와 메모리 측면에서 더 효율적임
'''

set.seed(3)
n = 1000
p = 500
X = matrix(rnorm(n*p), ncol=p)
beta = c(3, 2, 1, rep(0, p-3))
y = X %*% beta + rnorm(n=n, sd=0.1)


library(profmem)

tval = rep(0, 2)

library(lars)

p1 = profmem({
  tval[1] = system.time(fit1 <- lars(X, y))[[3]]
  summary(fit1)
})
plot(fit1)

library(biglasso)

X.bm <- as.big.matrix(X)
p2 = profmem({
  tval[2] = system.time(fit2 <- biglasso(X.bm, y, family="gaussian"))[[3]]
  summary(fit2)
})
plot(fit2, log.l = TRUE, main = 'lasso')

tval

total(p1) ; total(p2)
