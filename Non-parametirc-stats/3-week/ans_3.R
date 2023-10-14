## 패키지 로드

library(devtools)
library(regbook)
library(MASS)
library(dplyr)
library(tidyr)
library(scatterplot3d)

set.seed(123)

## 4-2
```{r}
n = 100
x = seq(0, 1, length.out=n)
y_bar = sin(x*3*pi) + cos(x*5*pi) + x^2
y = y_bar + rnorm(n, sd=0.25)
x_test = sort(runif(n))
y_bar = sin(x_test*3*pi) + cos(x_test*5*pi) + x_test^2 
y_test = y_bar + rnorm(n, sd=0.25)
```

```{r}
casl_nlm1d_poly <- function(x, y, n=1L)
  {
    Z <- cbind(1/length(x), stats::poly(x, n=n))
    beta_hat <- crossprod(Z, y)
    beta_hat
  }

casl_nlm1d_poly_predict <- function(beta, x, x_new)
  {
    pobj <- stats::poly(x, n=(length(beta) - 1L))
    Z_new <- cbind(1.0, stats::predict(pobj, x_new))
    y_hat <- Z_new %*% beta
    y_hat
  }
```

### casl 다항회귀
```{r}
casl_poly <- casl_nlm1d_poly(x, y, 3)
casl_poly
```
### 기본함수 다항회귀
```{r}
lm_poly <- lm(y ~ poly(x, degree = 3))
lm_poly$coefficients
```


### 비교
```{r}
casl_predict <- casl_nlm1d_poly_predict(casl_poly, x, x_test)
predict(lm_poly, newdata = data.frame(x = x_test))
```

```{r}
casl_nlm1d_poly_predict(casl_poly, x, x_test)
```
아주 비슷한 값의 결과를 얻을 수 있다.



## 4-10
```{r}
casl_kernel <- function(x, h=1)
  {
    x <- x/h
    r <- as.numeric(abs(x) <= 1)
    val <- (0.75) * ( 1 - x^2 ) * r
    val
  }



ker_2d <- function(x, y, x_new, h=1)
  {
    sapply(data.frame(t(x_new)), function(v)
    {
      w <- casl_kernel((x - v)^2, h=h)
      yhat <- sum(w * y) / sum(w)
      yhat
    })
  }
```

### 세팅
```{r}
n <- 1000
x_1 = runif(n, min = 0, max = 10)
x_2 = runif(n, min = 0, max = 1)

x <- data.frame(x_1, x_2)

y_bar = cos(x_1) + (x_2)^2
y = y_bar + rnorm(n, sd=0.1)
```

### 예측값
```{r}
x_1_new <- seq(0, 10, length = 20)
x_2_new <- seq(0, 1, length = 20)

expand_grid <- expand.grid(x_1_new, x_2_new)
x_new <- data.frame(expand_grid)

y_new_hat <- matrix(ker_2d(x, y, x_new, h = 0.5),
                    nrow = length(x_1_new),
                    ncol = length(x_2_new), byrow = T
                    )

y_new_hat
```

### 시각화
```{r}
scatterplot3d::scatterplot3d(x = x_1, y = x_2, z = y)
```
