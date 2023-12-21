library(openxlsx)
library(caret)
library(earth)

CCPP <- read.xlsx('Folds5x2_pp.xlsx')
colnames(CCPP)
Fold <- createFolds(CCPP$PE, k=5, list=TRUE, returnTrain = T)



## 단변량 ##

list_nprune <- 3:20
CV_MSE <- matrix(NA, nrow = 5, ncol = length(list_nprune))

for (k in 1:5) {
  
  I.tr <- unlist(Fold[k])
  
  CCPP.tr <- CCPP[I.tr, ]
  CCPP.te <- CCPP[-I.tr, ]
  
  idx <- 0
  for (nprune in list_nprune){
    idx <- idx + 1
    MARS.model <- earth(PE ~ AT, data = CCPP.tr, pmethod = 'forward', degree = 1, nprune = nprune, thresh = 0)
    
    test_pred <- predict(MARS.model, newdata = CCPP.te)
    test_MSE <- mean((CCPP.te$PE - test_pred)^2)
    CV_MSE[k, idx] <- test_MSE
    
  }
}


## CV MSE ##
plot(list_nprune, colMeans(CV_MSE), type = 'b', xlab = '# of Basis', ylab = 'MSE')

optimal_nprune <- list_nprune[which.min(colMeans(CV_MSE))]
optimal_nprune


MARS.model <- earth(PE ~ AT, data = CCPP, pmethod = 'forward', degree = 1, nprune = optimal_nprune, thresh = 0)
linear.model <- lm(PE ~ AT, data = CCPP)


line_AT <- seq(0, 37, by = 0.01)

line_MARS_pred <- predict(MARS.model, newdata = data.frame(AT = line_AT))
line_linear_pred <- predict(linear.model, newdata = data.frame(AT = line_AT))

SP <- smooth.spline(CCPP$AT, CCPP$PE)


p <- ggplot() + geom_point(data = CCPP, mapping = aes(x = AT, y = PE), size = 0.5) + 
  geom_line(mapping = aes(x = line_AT, y = line_MARS_pred, col = 'MARS'), linewidth = 1.5) + 
  geom_line(mapping = aes(x = line_AT, y = line_linear_pred, col = 'Linear'), linewidth = 1.5) +
  geom_line(mapping = aes(x = line_AT, y = predict(SP, x = line_AT)$y, col = 'smooth.spline'), linewidth = 1.5) +
  scale_color_manual(values = c("red", "cyan", 'orange'), name = "Lines") # 범주 설정
p




## 다변량 ##

sample.idx <- sample(9568, 1000, replace = F)
CCPP.sample <- CCPP[sample.idx, ]


# 가법모형

list_nprune <- 3:20
CV_MSE <- matrix(NA, nrow = 5, ncol = length(list_nprune))

for (k in 1:5) {
  
  I.tr <- unlist(Fold[k])
  
  CCPP.tr <- CCPP[I.tr, ]
  CCPP.te <- CCPP[-I.tr, ]
  
  idx <- 0
  for (nprune in list_nprune){
    idx <- idx + 1
    
    
    MARS.model <- earth(PE ~ ., data = CCPP.tr, pmethod = 'forward', degree = 1, nprune = nprune, thresh = 0)
    
    test_pred <- predict(MARS.model, newdata = CCPP.te)
    test_MSE <- mean((CCPP.te$PE - test_pred)^2)
    CV_MSE[k, idx] <- test_MSE
    
  }
}


plot(list_nprune, colMeans(CV_MSE), type = 'b', xlab = '# of Basis', ylab = 'MSE')

optimal_nprune <- list_nprune[which.min(colMeans(CV_MSE))]
optimal_nprune


MARS.model <- earth(PE ~ ., data = CCPP, pmethod = 'forward', degree = 1, nprune = optimal_nprune, thresh = 0)
MARS.pred <- predict(MARS.model, newdata = CCPP.sample)

linear.model <- lm(PE ~ ., data = CCPP)
linear.pred <- predict(linear.model, newdata = CCPP.sample)


library(gridExtra)

p_AT <- ggplot() + geom_point(data = CCPP, mapping = aes(x = AT, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = AT, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = AT, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정

p_V <- ggplot() + geom_point(data = CCPP, mapping = aes(x = V, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = V, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = V, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정

p_AP <- ggplot() + geom_point(data = CCPP, mapping = aes(x = AP, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = AP, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = AP, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정

p_RH <- ggplot() + geom_point(data = CCPP, mapping = aes(x = RH, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = RH, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = RH, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정


grid.arrange(p_AT, p_V, p_AP, p_RH, nrow=2, ncol=2)


p <- ggplot() + geom_point(mapping = aes(x = CCPP.sample[, 'PE'], y = MARS.pred, col = 'MARS')) +
  geom_point(mapping = aes(x = CCPP.sample[, 'PE'], y = linear.pred, col = 'linear')) +
  geom_abline(intercept = 0, slope = 1, linewidth = 2, col = 'black') +
  scale_color_manual(values = c("red", "cyan"))
p




# 2차 상호작용모형 -> 거의 차이 없음

# 가법모형

list_nprune <- 5:25
CV_MSE <- matrix(NA, nrow = 5, ncol = length(list_nprune))

for (k in 1:5) {
  
  I.tr <- unlist(Fold[k])
  
  CCPP.tr <- CCPP[I.tr, ]
  CCPP.te <- CCPP[-I.tr, ]
  
  idx <- 0
  for (nprune in list_nprune){
    idx <- idx + 1
    
    
    MARS.model <- earth(PE ~ ., data = CCPP.tr, pmethod = 'forward', degree = 2, nprune = nprune, thresh = 0)
    
    test_pred <- predict(MARS.model, newdata = CCPP.te)
    test_MSE <- mean((CCPP.te$PE - test_pred)^2)
    CV_MSE[k, idx] <- test_MSE
    
  }
}


plot(list_nprune, colMeans(CV_MSE), type = 'b', xlab = '# of Basis', ylab = 'MSE')

optimal_nprune <- list_nprune[which.min(colMeans(CV_MSE))]
optimal_nprune


MARS.model <- earth(PE ~ ., data = CCPP, pmethod = 'forward', degree = 2, nprune = optimal_nprune, thresh = 0)
MARS.pred <- predict(MARS.model, newdata = CCPP.sample)

linear.model <- lm(PE ~ ., data = CCPP)
linear.pred <- predict(linear.model, newdata = CCPP.sample)


library(gridExtra)

p_AT <- ggplot() + geom_point(data = CCPP, mapping = aes(x = AT, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = AT, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = AT, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정

p_V <- ggplot() + geom_point(data = CCPP, mapping = aes(x = V, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = V, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = V, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정

p_AP <- ggplot() + geom_point(data = CCPP, mapping = aes(x = AP, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = AP, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = AP, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정

p_RH <- ggplot() + geom_point(data = CCPP, mapping = aes(x = RH, y = PE), size = 0.2) + 
  geom_point(data = CCPP.sample, mapping = aes(x = RH, y = MARS.pred, col = 'MARS'), size = 2) +
  geom_point(data = CCPP.sample, mapping = aes(x = RH, y = linear.pred, col = 'linear'), size = 2) +
  scale_color_manual(values = c("red", "cyan")) # 범주 설정


grid.arrange(p_AT, p_V, p_AP, p_RH, nrow=2, ncol=2)


p <- ggplot() + geom_point(mapping = aes(x = CCPP.sample[, 'PE'], y = MARS.pred, col = 'MARS')) +
  geom_point(mapping = aes(x = CCPP.sample[, 'PE'], y = linear.pred, col = 'linear')) +
  geom_abline(intercept = 0, slope = 1, linewidth = 2, col = 'black') +
  scale_color_manual(values = c("red", "cyan"))
p
