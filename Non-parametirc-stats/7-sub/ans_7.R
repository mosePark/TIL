# 1
'''
캐나다 날씨 데이터에서 밴쿠버의 경우 K=13이 선택된 반면 퀘벡 지역에 대하여 CV를 이용하여 푸리에 기저 갯수를 선택한 결과 K=5가 선택되었음
'''

library(fda)

data(CanadianWeather)
attach(CanadianWeather)

y.precip=dailyAv[,,2]
l = which(place=="Quebec") 
t.day = 1:365  
y=y.precip[,l]

m=length(y)

K.vec = 2*c(1:8)+1; 

fbasis=create.fourier.basis(rangeval = c(1, 365), nbasis=max(K.vec), period=365)
bvals = eval.basis(t.day,fbasis)

CVfit = matrix(0,  nrow=m, ncol=length(K.vec))
for(j in 1:m){
  Y.star = y[-j]
  # fit using Fourier basis and K basis functions
  index=0
  for (K in K.vec){
    index=index+1
    Xbasis=bvals[, 1:K];
    Xbasis.j =  Xbasis[-j, ]; 
    lm.fit = lm(Y.star~0+Xbasis.j); Xbasis.coeff = lm.fit$coefficients
    y.fit = Xbasis%*%Xbasis.coeff
    CVfit[j,index] = (y[j] - y.fit[j])^2
  }
}

CV_L2 = apply(CVfit, 2, sum)
plot(K.vec, CV_L2, type="n",
     xlab="Number of basis functions", ylab="Total cross-validation error")
points(K.vec, CV_L2, type='b', col="royalblue2", lwd=2)
title(paste0("K = ", K.vec[which(CV_L2==min(CV_L2))], " with the smallest CV score!"))

# 2
'''
CD4 데이터에 대하여 fPCA에서 데이터의 변동성이 99%이상 설명되도록 주성분들을 갯수를 결정하시오. 이 때 선택된 고유함수들을 시각화하고 환자별 CD4 추정 곡선(몇 개만)을 시각화 하시오.
'''

## CD4(sparse)데이터에 대한 fPCA
# CD4 데이터(sparse)
library(refund)
data(cd4)
n <- nrow(cd4)
month <- as.numeric(colnames(cd4)) # months -18 and 42 since seroconversion
m <- ncol(cd4)

# sparse 데이터에 대한 FPCA 함수를 이용하여 분석
library(refund)

fpca.res <- fpca.sc(cd4, argvals = month, pve = 0.99, var = TRUE)
m <- length(month)

fpca.res$npc


efns <- fpca.res$efunctions*sqrt(m)
evals <- fpca.res$evalues/m

matplot(month, efns, type='l', lwd=2, lty=1, 
        ylab="", xlab="month", main="eigenfunctions")
legend("bottomright", lwd=2, lty=1, col=1:fpca.res$npc, 
       legend = paste0("fPC", 1:fpca.res$npc))

# 추정된 고유함수와 스코어를 이용한 환자별 추정 곡선
Yhat <- t(matrix(rep(fpca.res$mu, n), length(month))) + 
  fpca.res$scores %*% t(fpca.res$efunctions)


set.seed(12345)
n.crv <- 2
sel.crv <- sample(1:n, size = n.crv, replace = FALSE)

matplot(month, t(cd4), type='n', 
        main="CD4 cell counts", ylab="number of cd4 cells", 
        xlab="months since seroconversion" )
for(irow in 1:n){
  temp <- na.omit(data.frame(x = month, y = cd4[irow,]))
  lines(temp$x, temp$y, col="light grey")
}
for(i in 1:n.crv){
  irow <- sel.crv[i]
  temp <- na.omit(data.frame(x = month, y = cd4[irow,]))
  points(temp$x, temp$y, col=rainbow(n)[sel.crv[i]], 
         pch = 16, cex=1)
  lines(temp$x, temp$y, col=rainbow(n)[sel.crv[i]], lwd=2)
  lines(month, Yhat[sel.crv[i],], 
        col=rainbow(n)[sel.crv[i]], lwd=2)
}
