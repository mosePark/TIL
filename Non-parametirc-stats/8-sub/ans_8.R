# 1.
linearKernel <- function(x,y){
  crossprod(x, y)
}

krr <- function(method, dr, trainX, trainY, testX, lambda){
  N <- nrow(trainX)
  M <- nrow(testX)
  Km <- matrix(0,N,N)
  km <- matrix(0,N,M)
  # method = 0, means use linear kernel
  if (method == 0) {
    for (i in 1:N) {
      for (j in 1:N) {
        Km[i,j] <- linearKernel(trainX[i,],trainX[j,])
      }
      for (k in 1:M) {
        km[i,k] <- linearKernel(trainX[i,],testX[k,])
      }
    }
  }
  # method = 1, means use polynomial kernel
  if (method == 1) {
    for (i in 1:N) {
      for (j in 1:N) {
        Km[i,j] <- polynomialKernel(trainX[i,],trainX[j,],dr)
      }
      for (k in 1:M) {
        km[i,k] <- polynomialKernel(trainX[i,],testX[k,],dr)
      }
    }
  }
  # method = 2, means use radial kernel
  if (method == 2) {
    for (i in 1:N) {
      for (j in 1:N) {
        Km[i,j] <- radialKernel(trainX[i,],trainX[j,],dr)
      }
      for (k in 1:M) {
        km[i,k] <- radialKernel(trainX[i,],testX[k,],dr)
      }
    }
  }
  
  predY <- c()
  for (i in 1:M) {
    predY[i] <- trainY%*%solve(lambda*diag(N)+Km)%*%km[,i]
  }
  
  return(predY)
}

set.seed(20230101)
n = 200; p = 4
beta = c(1, -1, 0.5, 0)
mu = rep(0, p)
Sigma = matrix(0.9, p, p)
diag(Sigma) = 1
X = MASS::mvrnorm(n, mu, Sigma)
y = X %*% beta + rnorm(n, sd=5)

method = 0
d = 1
lam = 1

library(MASS)


fit1 = lm.ridge(y~X-1, lambda=lam)
pred1 = X %*% as.matrix(coef(fit1)) 
t(pred1)[1:5]

mean((pred1-y)^2)

pred2 = krr(0, d, X, t(y), X, lambda=lam)
pred2[1:5]

mean((pred2-y)^2)

'''
선형커널에 대하여 krr 함수의 예측 결과와 MASS::lm.ridge의 예측 결과가 거의 일치함
'''

# 2. 연습문제 9.7
set.seed(1)
n = 200

theta = seq(0, 4*pi, length.out=n)
r = seq(1, 3, length.out=n)
Q = cbind(cos(theta)*r, sin(theta)*r)
X = Q + matrix(runif(n*2, min=-0.15, max=0.15), ncol=2)

S1 = casl_util_knn_sim(X, k=8L)
Z1 = casl_spectral_clust(S1, k=2L)

S2 = casl_util_knn_sim(X, k=4L)
Z2 = casl_spectral_clust(S2, k=2L)

S3 = casl_util_knn_sim(X, k=3L)
Z3 = casl_spectral_clust(S3, k=2L)

par(mfrow=c(1,4))
plot(X, main="samples")
plot(Z1, main="2 SC: 8 neighbors", xlab="comp1", ylab="comp2")
plot(Z2, main="2 SC: 4 neighbors", xlab="comp1", ylab="comp2")
plot(Z3, main="2 SC: 3 neighbors", xlab="comp1", ylab="comp2")
par(mfrow=c(1,1))
'''
스펙트럴 군집 예제에서 군집의 갯수를 8개에서 4개로 줄였을 때 앞의 2 스펙트럴 차원에서 차이가 나타남. 3으로 더 줄이면 명확한 차이가 나타남(근방의 갯수가 작아지면 주변의 데이터들이 서로 영향을 덜 미치게 됨)
'''

# 3. kernlab::kpca와 casl_kernel_pca의 비교

library(profmem)
library(kernlab)

set.seed(1)
n = 2000

theta = seq(0, 4*pi, length.out=n)
r = seq(1, 3, length.out=n)
Q = cbind(cos(theta)*r, sin(theta)*r)
X = Q + matrix(runif(n*2, min=-0.15, max=0.15), ncol=2)

tval = rep(0, 2)

p1 = profmem({
  tval[1] = system.time(pca_radial <- casl_kernel_pca(X, k=2L, gamma=1, 
                                                      kfun=casl_util_kernel_radial))[[3]]
  })

p2 = profmem({
  tval[2] = system.time(kpc <- kernlab::kpca(X, kernel="rbfdot", 
                                    kpar=list(sigma=1), features=2))[[3]]
  })

par(mfrow=c(1,2))
plot(pca_radial, main="radial kernel", xlab="pca_radial1", ylab="pca_radial2")
plot(rotated(kpc), main="radial kernel(kernlab)", xlab="pca_radial1", ylab="pca_radial2")

'''
속도는 kpca가 빠르고 메모리는 더 많이 사용함
'''

par(mfrow=c(1,1))

tval

total(p1) ; total(p2) # 메모리 결과 비교

# 4. 10.3

library(Matrix)

for (s in c(1, 3, 6, 9)) {
  set.seed(1)
  Xd = matrix(0, ncol=100, nrow=4000)
  Xd[sample(seq_along(Xd), 10000/s)] = 1
  Xs = Matrix(Xd, sparse=TRUE)
  qrd = base::qr(Xd)
  qrs = Matrix::qr(Xs)
  
  cat(s, ":", format(object.size(Xs), units = "auto"),
      format(object.size(qrs), units = "auto"), "\n")
}

'''
성긴 행렬 Xs의 크기는 성긴정도에 거의 비례하여 크기가 감소하는 반면 qrs(Xs에 대한 QR 분해)는 성긴정도에 따라 크기가 감소하지만 정비례하는 것 같지는 않음
'''
