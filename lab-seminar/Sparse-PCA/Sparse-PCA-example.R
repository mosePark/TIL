library(elasticnet)
'''
spca(x, K, para, type=c("predictor","Gram"),
     sparse=c("penalty","varnum"), use.corr=FALSE, lambda=1e-6,
     max.iter=200, trace=FALSE, eps.conv=1e-3)
'''

## example

# NOT RUN {
data(pitprops) # 목재 데이터
out1<-spca(pitprops,K=6,type="Gram",sparse="penalty",trace=TRUE,para=c(0.06,0.16,0.1,0.5,0.5,0.5))
## print the object out1
out1
out2<-spca(pitprops,K=6,type="Gram",sparse="varnum",trace=TRUE,para=c(7,4,4,1,1,1))
out2
## to see the contents of out2
names(out2) 
## to get the loadings
out2$loadings
# }

### from documentaiton of elasticnet package example code in R.
