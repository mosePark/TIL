library(rbenchmark)
library(mgcv)
library(gam)

# ? mgcv::gam()
# ? gam::gam()

casl_am_backfit <-function(X, y, maxit=10L)
  {
    p <- ncol(X)
    id <- seq_len(nrow(X))
    alpha <- mean(y)
    f <- matrix(0, ncol = p, nrow = nrow(X))
    models <- vector("list", p + 1L)
    for (i in seq_len(maxit))
    {
      for (j in seq_len(p))
      {
        p_resid <- y - alpha - apply(f[, -j], 1L, sum)
        id <- order(X[,j])
        models[[j]] <- smooth.spline(X[id,j], p_resid[id])
        f[,j] <- predict(models[[j]], X[,j])$y
      }
      alpha <- mean(y - apply(f, 1L, sum))
    }
    models[[p + 1L]] <- alpha
    return(models)
}

## 데이터
set.seed(123)

n <- 500; p <- 4
X <- matrix(runif(n * p, min = -2, max = 2), ncol = p)
f1 <- cos(X[,1] * 4) + sin(X[,1] * 10) + X[,1]^(2) 
f2 <- -1.5 * X[,2]^2 + (X[,2] > 1) * (X[,2]^3 - 1)
f3 <- 0
f4 <- sign(X[,4]) * 1.5
f1 <- f1 - mean(f1); f2 <- f2 - mean(f2)
f3 <- f3 - mean(f3); f4 <- f4 - mean(f4)
y <- 10 + f1 + f2 + f3 + f4 + rnorm(n, sd = 1.2)

colnames(X) <- c("X1", "X2", "X3", "X4")
df <- data.frame(y, X)

rbenchmark::benchmark(
  "casl" = {
    casl_am_backfit(X, y, maxit = 1)
  },
  
  "mgcv" = {
    mgcv::gam(y~s(X1) + s(X2) + s(X3) + s(X4), data = df)
  },
  
  "gam" = {
    gam::gam(y~s(X1, 4) + s(X2, 4) + s(X3, 4) + s(X4, 4), data = df)
  },
  replications = 1000,
  columns = c("test", "replications", "elapsed", "relative", "user.self", "sys.self",
              "user.child", "sys.child")
)
'''gam, mgcv, casl 순으로 빠르다.'''


################################################################################
# bam vs gam

## 데이터, 15 col, 150,000 row
n <- 150000 ; p <- 15
X <- matrix(runif(n * p, min = -2, max = 2), ncol = p)

f1 <- cos(X[,1] * 4) + sin(X[,1] * 10) + X[,1]^(2)
f2 <- -1.5 * X[,2]^2 + (X[,2] > 1) * (X[,2]^3 - 1)
f3 <- X[,3]^3 + sin(X[,3] * 5)
f4 <- log(1 + abs(X[,4]))
f5 <- exp(X[,5])
f6 <- sqrt(abs(X[,6]))
f7 <- X[,7]^2.5
f8 <- sin(X[,8] * 3) * X[,8]^2
f9 <- log(abs(X[,9]) + 1)
f10 <- X[,10]^4
f11 <- exp(-X[,11])
f12 <- X[,12]^(1/3)
f13 <- cos(X[,13]) * X[,13]^2
f14 <- sin(X[,14]) / (1 + X[,14]^2)
f15 <- X[,15] / (1 + abs(X[,15]))

# Centering
f1 <- f1 - mean(f1) ; f2 <- f2 - mean(f2) ; f3 <- f3 - mean(f3)
f4 <- f4 - mean(f4) ; f5 <- f5 - mean(f5) ; f6 <- f6 - mean(f6)
f7 <- f7 - mean(f7) ; f8 <- f8 - mean(f8) ; f9 <- f9 - mean(f9)
f10 <- f10 - mean(f10) ; f11 <- f11 - mean(f11) ; f12 <- f12 - mean(f12)
f13 <- f13 - mean(f13) ; f14 <- f14 - mean(f14) ; f15 <- f15 - mean(f15)

# Response Variable
y <- 10 + f1 + f2 + f3 + f4 + f5 
        + f6 + f7 + f8 + f9 + f10
        + f11 + f12 + f13 + f14 + f15 + rnorm(n, sd = 1.2)

df2 <- data.frame(y, X)

bam_time <- system.time(
mgcv::bam(y~s(X[,1]) + s(X[,2]) + s(X[,3]) + s(X[,4]) + s(X[,5])
          + s(X[,6]) + s(X[,7]) + s(X[,8]) + s(X[,9]) + s(X[,10])
          + s(X[,11]) + s(X[,12]) + s(X[,13]) + s(X[,14]) + s(X[,15]), data = df2)
)


gam_time <- system.time(
mgcv::gam(y~s(X[,1]) + s(X[,2]) + s(X[,3]) + s(X[,4]) + s(X[,5])
          + s(X[,6]) + s(X[,7]) + s(X[,8]) + s(X[,9]) + s(X[,10])
          + s(X[,11]) + s(X[,12]) + s(X[,13]) + s(X[,14]) + s(X[,15]), data = df2)
)
