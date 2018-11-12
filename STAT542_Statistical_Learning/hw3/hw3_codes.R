### Problem 1
library(quadprog)
library(e1071)

## a)
set.seed(1)
n <- 40
p <- 2
xpos <- matrix(rnorm(n*p,mean=0,sd=1),n,p)
xneg <- matrix(rnorm(n*p,mean=4,sd=1),n,p)
x <- rbind(xpos,xneg)
y <- matrix(c(rep(1,n),rep(-1,n)))

# Constuct parameters in solve.QR
Dmat <- diag(p+1)
Dmat[1,1] <- 0
ridge <- 10^(-5)*diag(p+1)
Dmat <- Dmat+ridge
dvec <- rep(0,p+1)
X <- cbind(rep(1,(2*n)),x)
Amat <- diag(c(y)) %*% X
bvec <- rep(1,2*n)

sol <- solve.QP(Dmat,dvec,t(Amat),bvec)
beta = sol$solution

# Obtain support vectors
which((y*(X %*% beta)-1) < 1e-10)
sv = x[which((y*(X %*% beta)-1) < 1e-10),]

# Plot the results produced by quadprog package
plot(x,col=ifelse(y>0,"red", "blue"), pch = 19, cex = 1.2, lwd = 2, xlab = "X1", ylab = "X2", cex.lab = 1.5)
legend("bottomleft", c("Positive","Negative"),col=c("red", "blue"),pch=c(19, 19),text.col=c("red", "blue"), cex = 1.5)
w_quad <- t(as.matrix(sol$solution[-1]))
b_quad <- sol$solution[1]
abline(a= -b_quad/w_quad[1,2], b=-w_quad[1,1]/w_quad[1,2], col="black", lty=1, lwd = 2)
abline(a= (-b_quad-1)/w_quad[1,2], b=-w_quad[1,1]/w_quad[1,2], col="black", lty=3, lwd = 2)
abline(a= (-b_quad+1)/w_quad[1,2], b=-w_quad[1,1]/w_quad[1,2], col="black", lty=3, lwd = 2)
points(sv, col="black", cex=3)

# Plot the results produced by e1071 package
plot(x,col=ifelse(y>0,"red", "blue"), pch = 19, cex = 1.2, lwd = 2, xlab = "X1", ylab = "X2", cex.lab = 1.5)
legend("bottomleft", c("Positive","Negative"),col=c("red", "blue"),pch=c(19, 19),text.col=c("red", "blue"), cex = 1.5)
svm.fit <- svm(y ~ ., data = data.frame(x, y), type='C-classification', kernel='linear',scale=FALSE, cost = 10000)
w <- t(svm.fit$coefs) %*% svm.fit$SV
b <- -svm.fit$rho
abline(a= -b/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=1, lwd = 2)
abline(a= (-b-1)/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=3, lwd = 2)
abline(a= (-b+1)/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=3, lwd = 2)
points(x[svm.fit$index, ], col="black", cex=3)

## b)
n <- 80
Dmat <- matrix(0,n,n)
for (i in 1:n){
  for (j in 1:n){
    Dmat[i,j] = y[i]*y[j]*(t(x[i,]) %*% x[j,])
  }
}
ridge <- 10^(-5)*diag(n)
Dmat <- Dmat+ridge
dvec <- rep(1,n)
A.Equality <- y
Amat <- cbind(A.Equality,diag(n))
bvec <- c(0,rep(0,n))

sol <- solve.QP(Dmat,dvec,Amat,bvec,meq = 1)
alpha = sol$solution

beta_hat <- rep(0,p)
for (i in 1:n){
  beta_hat = beta_hat+alpha[i]*y[i]*x[i,]
}
beta0_hat = -(max(x[41:80,] %*% beta_hat)+min(x[1:40,] %*% beta_hat))/2

sv3 = x[which(abs(alpha) > 1e-2),]

# Plot the results produced by quadprog package
plot(x,col=ifelse(y>0,"red", "blue"), pch = 19, cex = 1.2, lwd = 2, xlab = "X1", ylab = "X2", cex.lab = 1.5)
legend("bottomleft", c("Positive","Negative"),col=c("red", "blue"),pch=c(19, 19),text.col=c("red", "blue"), cex = 1.5)
w_dual <- t(as.matrix(beta_hat))
b_dual <- beta0_hat
abline(a= -b_dual/w_dual[1,2], b=-w_dual[1,1]/w_dual[1,2], col="black", lty=1, lwd = 2)
abline(a= (-b_dual-1)/w_dual[1,2], b=-w_dual[1,1]/w_dual[1,2], col="black", lty=3, lwd = 2)
abline(a= (-b_dual+1)/w_dual[1,2], b=-w_dual[1,1]/w_dual[1,2], col="black", lty=3, lwd = 2)
points(sv3, col="black", cex=3)

# Plot the results produced by e1071 package
svm.fit <- svm(y ~ ., data = data.frame(x, y), type='C-classification', kernel='linear',scale=FALSE, cost = 10000)
w <- t(svm.fit$coefs) %*% svm.fit$SV
b <- -svm.fit$rho
sv4 = x[svm.fit$index, ]
plot(x,col=ifelse(y>0,"red", "blue"), pch = 19, cex = 1.2, lwd = 2, xlab = "X1", ylab = "X2", cex.lab = 1.5, main = "e1071")
legend("topleft", c("Positive","Negative"),col=c("red", "blue"),pch=c(19, 19),text.col=c("red", "blue"), cex = 0.9)
abline(a= -b/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=1, lwd = 2)
abline(a= (-b-1)/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=3, lwd = 2)
abline(a= (-b+1)/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=3, lwd = 2)
points(sv4, col="black", cex=3)

## c)
library(quadprog)
library(e1071)
set.seed(70)
n <- 10 # number of data points for each class
p <- 2 # dimension

# Generate the positive and negative examples
xpos <- matrix(rnorm(n*p,mean=0,sd=1),n,p)
xneg <- matrix(rnorm(n*p,mean=1.5,sd=1),n,p)
x <- rbind(xpos,xneg)
y <- matrix(c(rep(1,n),rep(-1,n)))

# Visualize the data
plot(x,col=ifelse(y>0,"red", "blue"), pch = 19, cex = 1.2, lwd = 2, xlab = "X1", ylab = "X2", cex.lab = 1.5)
legend("topright", c("Positive","Negative"),col=c("red", "blue"),pch=c(19, 19),text.col=c("red", "blue"), cex = 1.5)

# Use solve.QP
C <- 0.1
n <- 20
Dmat <- matrix(0,n,n)
for (i in 1:n){
  for (j in 1:n){
    Dmat[i,j] = y[i]*y[j]*(t(x[i,]) %*% x[j,])
  }
}
ridge <- 10^(-5)*diag(n)
Dmat <- Dmat+ridge
dvec <- rep(1,n)
A.Equality <- y
Amat <- cbind(A.Equality,diag(n),-diag(n))
bvec <- c(0,rep(0,n),rep(-C,n))

sol <- solve.QP(Dmat,dvec,Amat,bvec,meq = 1)
alpha = sol$solution
index <- which((alpha >= 10^(-4))&(alpha <= 0.11))

beta_hat <- rep(0,p)
for (i in index){
  beta_hat = beta_hat+alpha[i]*y[i]*x[i,]
}

x_new = x %*% beta_hat
index_new <- which((alpha >= 10^(-4))&(alpha <= 0.1-10^(-4)))
beta0_hat <- -mean(x_new[index_new])

b_new <- beta0_hat
w_new <- t(as.matrix(beta_hat))
plot(x,col=ifelse(y>0,"red", "blue"), pch = 19, cex = 1.2, lwd = 2, xlab = "X1", ylab = "X2", cex.lab = 1.5)
legend("bottomleft", c("Positive","Negative"),col=c("red", "blue"),pch=c(19, 19),text.col=c("red", "blue"), cex = 1.5)
abline(a= -b_new/w_new[1,2], b=-w_new[1,1]/w_new[1,2], col="blue", lty=1, lwd = 2)
abline(a= (-b_new-1)/w_new[1,2], b=-w_new[1,1]/w_new[1,2], col="blue", lty=3, lwd = 2)
abline(a= (-b_new+1)/w_new[1,2], b=-w_new[1,1]/w_new[1,2], col="blue", lty=3, lwd = 2)
points(x[index, ], col="black", cex=3)

svm.fit <- svm(y ~ ., data = data.frame(x, y), type='C-classification', kernel='linear',scale=FALSE, cost = 0.1)
w <- t(svm.fit$coefs) %*% svm.fit$SV
b <- -svm.fit$rho
abline(a= -b/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=1, lwd = 2)
abline(a= (-b-1)/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=3, lwd = 2)
abline(a= (-b+1)/w[1,2], b=-w[1,1]/w[1,2], col="black", lty=3, lwd = 2)
points(x[svm.fit$index, ], col="black", cex=3)


## d)
library(quadprog)
library(e1071)
library(ElemStatLearn)
data(SAheart)

SAheart$chd[SAheart$chd == 0] <- -1
SAheart$famhist <- model.matrix( ~ famhist - 1, data = SAheart) [,2]

set.seed(1)
n = dim(SAheart)[1]
sample <- sample(sequence(n))
SAheart_new <- SAheart[sample,]
rownames(SAheart_new) <- sequence(n)
SAheart_new <- as.matrix(SAheart_new)

mysvm <- function(x,y,C){
  n = dim(x)[1]
  p = dim(x)[2]
  Dmat <- matrix(0,n,n)
  for (i in 1:n){
    for (j in 1:n){
      Dmat[i,j] = y[i]*y[j]*(t(x[i,]) %*% x[j,])
    }
  }
  ridge <- 10^(-5)*diag(n)
  Dmat <- Dmat+ridge
  dvec <- rep(1,n)
  A.Equality <- y
  Amat <- cbind(A.Equality,diag(n),-diag(n))
  bvec <- c(0,rep(0,n),rep(-C,n))
  
  sol <- solve.QP(Dmat,dvec,Amat,bvec,meq = 1)
  alpha = sol$solution
  index <- which((alpha >= 10^(-4))&(alpha <= C+10^(-5)))
  
  beta_hat <- rep(0,p)
  for (i in index){
    beta_hat = beta_hat+alpha[i]*y[i]*x[i,]
  }
  
  x_new = x %*% beta_hat
  beta0_hat <- -mean(x_new[index])
  return(c(beta0_hat,beta_hat))
}

nrFolds <- 5
# generate array containing fold-number for each sample (row)
folds <- rep_len(1:nrFolds, nrow(SAheart_new))

mycv <- function(C){
  myerror <- c()
  # actual cross validation
  for (k in 1:nrFolds) {
    fold <- which(folds == k)
    data.train <- SAheart_new[-fold,]
    data.test <- SAheart_new[fold,]
    
    beta <- mysvm(data.train[,-10],data.train[,10],C)
    
    n_test <- dim(data.test)[1]
    X <- cbind(rep(1,n_test),data.test[,-10])
    
    mypre <- sign(X %*% beta)
    
    myerror[k] <- mean(mypre == data.test[,10])
  }
  return(mean(myerror))
}

cv <- function(C){
  error <- c()
  for (k in 1:nrFolds) {
    fold <- which(folds == k)
    data.train <- SAheart_new[-fold,]
    data.test <- SAheart_new[fold,]
    
    svm.fit <- svm(chd ~ ., data = data.train, type='C-classification', kernel='linear',scale=FALSE, cost=C)
    pre <- predict(svm.fit,data.test[,-10])
    error[k] <- mean(pre == data.test[,10])
  }
  return(mean(error))
}

C_seq <- c(0.05,0.1,0.5,1,5)
error_mycv <- c()
error_cv <- c()
for (i in 1:length(C_seq)){
  C <- C_seq[i]
  error_mycv[i] <- mycv(C)
  error_cv[i] <- cv(C)
}


## Problem3
shooting <- read.csv("C:/Users/guoshuhui/Desktop/542/homework/hw3/Mass Shootings Dataset Ver 2.csv")
victims <- shooting$Total.victims
date <- as.Date(shooting$Date, "%m/%d/%Y")

library(ggplot2)
library(reshape2)
target <- data.frame(date,victims)
qplot(date,victims,data = target,geom = "line",main = "Total Victims")
ggplot(target,aes(date,victims))+ geom_point()+labs(title="Total Victims")

library(lubridate)
year = year(date)
year_sum <- tapply(victims, year, FUN=sum)
year_row <- as.numeric(rownames(year_sum))
year_sum <- data.frame(year_row,year_sum)
colnames(year_sum) <- c("year","victims")
year_miss <- c(1967:1970,1973,1975,1977,1978,1980,1981)
victims_miss <- rep(0,length(year_miss))
year_miss <- data.frame(year_miss,victims_miss)
colnames(year_miss) <- c("year","victims")
year_sum <- rbind(year_sum,year_miss)
rownames(year_sum)[43:52] <- c(1967:1970,1973,1975,1977,1978,1980,1981)
ggplot(year_sum,aes(year_sum$year,year_sum$victims))+ geom_point()+labs(x="year",y="victims",title="Total Victims Per Year")

library(splines)
fit = smooth.spline(year_sum$year, year_sum$victims)
fitted = predict(fit,year_sum$year)
fitted = as.data.frame(fitted)
gplot = ggplot(year_sum,aes(year_sum$year,year_sum$victims))+ geom_point()+labs(x="year",y="victims",title="Total Victims Per Year")
gplot = gplot + geom_line(data=fitted,aes(x=x,y=y),color = "red")
gplot
