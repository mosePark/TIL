rm(list = ls())

library(MASS)
library(kernlab)
library(irlba)
library(profmem)
library(Matrix)

################################################################################
# 1. 선형커널 함수를 작성하고 커널능형회귀의 예제에 대하여 적용한 후 3장의 MASS::lm.rdige 함수의 결과와 비교하시오. 
################################################################################
# 데이터
x <- matrix(rnorm(200 * 10), 200, 10)
y <- x[, 1] + x[, 2] ^ 2 + x[, 3] * x[, 4] + rnorm(200)

xnew <- matrix(rnorm(1000 * 10), 1000, 10)
ytrue <- xnew[, 1] + xnew[, 2] ^ 2 + xnew[, 3] * xnew[, 4]

# 사용자함수
polynomialKernel <- function(x,y,d){
  kxy <- (1+sum(x*y))^d
  kxx <- (1+sum(x*x))^d
  kyy <- (1+sum(y*y))^d
  if (kxx==0 | kyy==0) {
    res <- 0
  }
  else{
    res <- kxy/(sqrt(kxx)*sqrt(kyy))
  }
  #res <- (1+sum(x*y))^d+1
  return(res+1)
}

radialKernel <- function(x,y,r){
  kxy <- exp(-r*(sum(x-y))^2)
  kxx <- exp(-r*(sum(x-x))^2)
  kyy <- exp(-r*(sum(y-y))^2)
  if (kxx==0 | kyy==0) {
    res <- 0
  }
  else{
    res <- kxy/(sqrt(kxx)*sqrt(kyy))
  }
  #res <- exp(-r*(sum(x-y))^2)+1
  return(res+1)
}

# casl_kernel_ridge_reg
krr <- function(method, dr, trainX, trainY, testX, lambda){
  N <- nrow(trainX)
  M <- nrow(testX)
  Km <- matrix(0,N,N)
  km <- matrix(0,N,M)
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

lam = 0.001
# 사용자함수
pred_casl_kr <- krr(method=1, dr=1, trainX=x, trainY=y, testX=xnew, lam)

# MASS ridge
mass_ridge <- MASS::lm.ridge(y ~ x, lambda = lam)
pred_mass_ridge <- xnew %*% mass_ridge$coef

mean((pred_casl_kr - ytrue)^2) ; mean((pred_mass_ridge - ytrue)^2) # MSE

# head(pred_casl_kr)
# head(pred_mass_ridge)
'''MSE가 각각 3.38, 4.39 정도 나온다. 책에서 제공하는 사용자함수가 좀 더 우수하다. '''

################################################################################
# 2. 9장 연습문제 7번(근방의 갯수를 3으로 할 것)
################################################################################
casl_util_knn_sim <- function(X, k=10L)
{
  d <- as.matrix(stats::dist(X, upper = TRUE))^2
  N <- apply(d, 1L, function(v) v <= sort(v)[k + 1L])
  S <- (N + t(N) >= 2)
  diag(S) <- 0
  S
}

casl_spectral_clust <- function(S, k=1)
{
  D <- diag(apply(S, 1, sum))
  L <- D - S
  e <- eigen(L)
  Z <- e$vector[,rev(which(e$value > 1e-8))[seq_len(k)]]
  Z
}

# 데이터
set.seed(1)
n <- 200
theta <- seq(0, 4 * pi, length.out = n)
r <- seq(1, 3, length.out = n)
Q <- cbind(cos(theta) * r, sin(theta) * r)
X <- Q + matrix(runif(n * 2, min = -0.15, max = 0.15), ncol = 2)


# 모델링
S = casl_util_knn_sim(X, k=3) # 근방의 개수는 3개
Z = casl_spectral_clust(S, k=2L)

plot(X, main="X plot")
plot(Z, main="2 - spectral components", xlab="PC 1", ylab="PC 2")




################################################################################
# 3. kernlab의 kpca 함수와 casl_kernel_pca 함수의 결과와 속도/메모리사용량 등 비교하시오.
################################################################################

# 사용자함수
casl_util_kernel_norm <-function(M)
{
  n <- ncol(M)
  ones <- matrix(1 / n, ncol = n, nrow = n)
  Ms <- M - 2 * ones %*% M + ones %*% M %*% ones
  Ms
}


casl_kernel_pca <-function(X, k=2L, kfun=casl_util_kernel_poly, ...)
{
  M <- casl_util_kernel_norm(kfun(X, ...))
  e <- irlba::partial_eigen(M, n = k)
  pc <- e$vectors %*% diag(sqrt(e$values))
  pc[]
}

casl_util_kernel_poly <- function(X, d=2L, c=0)
{
  cross <- tcrossprod(X, X)
  M <- (cross + c)^d
  M
}

casl_util_kernel_radial <- function(X, gamma=1)
{
  d <- as.matrix(dist(X))^2
  M <- exp(-1 * gamma * d)
}

# 데이터
set.seed(1)
n <- 200
theta <- seq(0, 4 * pi, length.out = n)
r <- seq(1, 3, length.out = n)
Q <- cbind(cos(theta) * r, sin(theta) * r)
X <- Q + matrix(runif(n * 2, min = -0.15, max = 0.15), ncol = 2)

# 모델링
# RBF 커널을 사용한 PCA
pca_radial <- casl_kernel_pca(X, k=2L, gamma=25,
                              kfun=casl_util_kernel_radial)
# 다항 커널을 사용한 PCA
pca_poly <- casl_kernel_pca(X, k=2L, c=1,
                            kfun=casl_util_kernel_poly)


# kernlab 패키지지
# RBF 커널을 사용한 PCA
pca_radial_kernlab <- kernlab::kpca(X, kernel = "rbfdot", kpar = list(sigma = 1))
# 다항 커널을 사용한 PCA
pca_poly_kernlab <- kernlab::kpca(X, kernel = "polydot", kpar = list(degree = 2, scale = 1, offset = 1))



# 시간비교
time_casl_radial <- system.time(pca_radial <- casl_kernel_pca(X, k=2L, gamma=1,kfun=casl_util_kernel_radial))

time_radial <- system.time(pca_radial_kernlab <- kernlab::kpca(X, kernel = "rbfdot", kpar = list(sigma = 1)))

time_casl_poly <- system.time(pca_poly <- casl_kernel_pca(X, k=2L, c=1,
                                                          kfun=casl_util_kernel_poly))

time_poly <- system.time(pca_poly_kernlab <- kernlab::kpca(X, kernel = "polydot", kpar = list(degree = 2, scale = 1, offset = 1)))

time_casl_radial; time_radial

time_casl_poly ; time_poly
# 메모리비교

mem_casl_radial <- profmem::profmem(pca_radial <- casl_kernel_pca(X, k=2L, gamma=1,kfun=casl_util_kernel_radial))
mem_radiall <- profmem::profmem(pca_radial_kernlab <- kernlab::kpca(X, kernel = "rbfdot", kpar = list(sigma = 1)))
mem_casl_poly <- profmem::profmem(pca_poly <- casl_kernel_pca(X, k=2L, c=1, kfun=casl_util_kernel_poly))
mem_poly <- profmem::profmem(pca_poly_kernlab <- kernlab::kpca(X, kernel = "polydot", kpar = list(degree = 2, scale = 1, offset = 1)))

profmem::total(mem_casl_radial); profmem::total(mem_radiall)
profmem::total(mem_casl_poly); profmem::total(mem_poly)

'''상대적으로 casl이 패키지 profmem에 제공된 kpca 비해 조금더 느리면서 메모리는 적다.'''
################################################################################
# 4. 10장 연습문제 3
################################################################################

# sparse matrix
Xd <- matrix(0, ncol=100, nrow=4000)
Xd[sample(seq_along(Xd), 10000)] <- 1
Xs <- Matrix(Xd, sparse=TRUE)
qrd <- base::qr(Xd) # QR분해
qrs <- Matrix::qr(Xs) # QR분해2

# 행렬 10배 spare하게
Xd_10 <- matrix(0, ncol=100, nrow=4000*10)
Xd_10[sample(seq_along(Xd_10), 10000)] <- 1
Xs_10 <- Matrix(Xd_10, sparse=TRUE)
qrd_10 <- base::qr(Xd_10)
qrs_10 <- Matrix::qr(Xs_10)


10000 / (ncol(Xd) * nrow(Xd)) ; 10000 / (ncol(Xd_10) * nrow(Xd_10)) # 희소성 10배

# 비교
# 객체 크기 계산 및 출력
print_size_comparison <- function(object1, object2, name1, name2) {
  size1 <- format(object.size(object1), units = "auto")
  size2 <- format(object.size(object2), units = "auto")
  cat(paste(name1, "size:", size1, "\n"))
  cat(paste(name2, "size:", size2, "\n"))
  cat(paste("Difference ratio:", as.numeric(object.size(object2)) / as.numeric(object.size(object1)), "bytes ratio\n\n"))
}

# 비교 실행
print_size_comparison(Xd, Xd_10, "Xd", "Xd_10")
print_size_comparison(Xs, Xs_10, "Xs", "Xs_10")
print_size_comparison(qrd, qrd_10, "qrd", "qrd_10")
print_size_comparison(qrs, qrs_10, "qrs", "qrs_10")

'''sparsity 하게 10배로 실행했을때 바이트 사이즈는 10배 규모정도 나온다.
신기한점은 sparse하게 행렬을 처리(0 그냥 없는값으로)해준것은 메모리차이가 없다.

'''

# 3배씩 늘려보기
size_Xs <- rep(NA, 5)
size_qrs <- rep(NA, 5)

for (i in 1:5) {
  Xd <- matrix(0, ncol=1000, nrow=40000)
  Xd[sample(seq_along(Xd), 100000 / (i * 3))] <- 1
  Xs <- Matrix(Xd, sparse=TRUE)
  qrs <- Matrix::qr(Xs)
  
  size_Xs[i] <- format(object.size(Xs), units = "kB", standard = 'SI')
  size_qrs[i] <- format(object.size(qrs), units = "kB", standard = 'SI')
}

size_Xs ; size_qrs
'''사이즈가 줄어들수록 메모리 크기도 줄어드는 것을 볼 수 있다.'''

################################################################################
