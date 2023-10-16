# 4.2
set.seed(7)

n = 100
x = seq(0, 1, length.out=n)
y_bar = sin(x*3*pi) + cos(x*5*pi) + x^2
y = y_bar + rnorm(n, sd=0.25)
x_test = sort(runif(n))
y_bar = sin(x_test*3*pi) + cos(x_test*5*pi) + x_test^2 
y_test = y_bar + rnorm(n, sd=0.25)

beta = casl_nlm1d_poly(x=x, y=y, n=2L) 
print(beta)

y_test_hat = casl_nlm1d_poly_predict(beta, x, x_test)
head(y_test_hat)

fit = lm(y~poly(x, n=2))
coef(fit)

y_test_hat2 = casl_nlm1d_poly_predict(coef(fit), x, x_test)
head(y_test_hat2

# 4.10
kernel_reg_2d =
function(x, y, x_new, h=0.5)
{
  n = nrow(x); m = nrow(x_new)
  yhat = rep(NA, m)
  for (i in 1:m) {
    mat = x - rep(1,n) %o% as.numeric(x_new[i,]) 
    w <- casl_util_kernel_epan(sqrt(rowSums(mat^2)), h=h)
    yhat[i] <- sum(w * y) / sum(w)
  }
  return(yhat)
}

n = 100
x1 = sort(runif(n, 0, 2*pi))
x2 = sort(runif(n))
x = cbind(x1, x2)
y_bar = cos(x1) + x2^2
y = y_bar + rnorm(n, sd=0.25)
dat = data.frame(x, y)

x1_new = seq(0, 2*pi, length.out=n)
x2_new = seq(0, 1, length.out=n)
x_new = expand.grid(x1_new, x2_new)
y_new = kernel_reg_2d(x, y, x_new, h=1)
z = matrix(y_new, n, n)

library(plotly)

## 시각화
fig <- plot_ly(x = x1_new, y = x2_new, z = z) %>% add_surface() 
fig  
