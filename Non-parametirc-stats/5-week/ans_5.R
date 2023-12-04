# 1번
## 가법모형 
n = 10000; p = 4
X = matrix(runif(n*p, min=-2, max=2), ncol=p)
f1 = cos(X[,1] * 4) + sin(X[,1] * 10) + X[,1]^2
f2 = -1.5 * X[,2]^2 + (X[,2] > 1) * (X[,2]^3 - 1)
f3 = 0
f4 = sign(X[,4]) * 1.5

f1 = f1 - mean(f1); f2 = f2 - mean(f2)
f3 = f3 - mean(f3); f4 = f4 - mean(f4)

y = 10 + f1 + f2 + f3 + f4 + rnorm(n, sd=1.2)
dat = data.frame(X, y)


tval = rep(0, 3)
tval[1] = system.time(models <- casl_am_backfit(X, y))[[3]]
pred = casl_am_predict(models, X)


library(gam) # smoothing splines

'''
교재의 함수와 gam 패키지의 함수는 경우 평활스플라인에 기반하고 mgcv의 함수는 회귀스플라인에 기반하며 세 가지 모두 결과는 다르게 나옴. 교재와 mgcv의 함수의 실행 속도는 비슷하며 gam함수가 가장 빠름
'''

tval[2] = system.time(fit1 <- gam(y ~ s(X1) + s(X2) + s(X3) + s(X4), data=dat))[[3]]
summary(fit1)

library(mgcv) # regression splines

tval[3] = system.time(fit2 <- gam(y ~ s(X1) + s(X2) + s(X3) + s(X4), data=dat))[[3]]
summary(fit2)

x = seq(-2, 2, length=100)

par(mfrow=c(3, 4))
plot(models[[1]], type="l")
plot(models[[2]], type="l")
plot(models[[3]], type="l", ylim = c(-1, 1))
plot(models[[4]], type="l")

plot(fit1, se=T)
plot(fit2, se=T)

par(mfrow=c(1, 1))

tval

# 6.5
'''
bam에서 discrete=TRUE 옵션을 이용하였고 추정 결과는 gam과 유사하게 나오며 속도는 더 빠르며 전체 사용메모리가 훨씬 적음
'''

n = 150000; p = 15
X = matrix(runif(n*p, min=-2, max=2), ncol=p)
f1 = cos(X[,1] * 4) + sin(X[,1] * 10) + X[,1]^2
f2 = -1.5 * X[,2]^2 + (X[,2] > 1) * (X[,2]^3 - 1)
f3 = 0
f4 = sign(X[,4]) * 1.5

f1 = f1 - mean(f1); f2 = f2 - mean(f2)
f3 = f3 - mean(f3); f4 = f4 - mean(f4)

y = 10 + f1 + f2 + f3 + f4 + rnorm(n, sd=1.2)
dat = data.frame(X, y)

library(profmem)
options(profmem.threshold = 20000)

tval = rep(0, 2)

library(mgcv) 
p1 = profmem({
  tval[1] = system.time(fit1 <- gam(y ~ s(X1) + s(X2) + s(X3) + s(X4), data=dat))[[3]]
  summary(fit1)
})

p2 = profmem({
  tval[2] = system.time(fit2 <- bam(y ~ s(X1) + s(X2) + s(X3) + s(X4), data=dat, discrete=TRUE))[[3]]
  summary(fit2)
})

tval

total(p1) ; total(p2)
