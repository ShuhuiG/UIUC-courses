## Question 1
# a)

set.seed(1)
# generate some data
x <- runif(1000, 0, 2*pi)
y <- 2*sin(x) + rnorm(length(x))

# generate testing data points
test.x = runif(1000, 0, 2*pi)
test.y = 2*sin(test.x) + rnorm(length(test.x))
test.y = test.y[order(test.x)]
test.x = test.x[order(test.x)]

# construct my functions
# Nadaraya-Watson kernel regression estimator
my_est <- function(X, x, y, lambda){
  u <- abs(X-x)/lambda
  K <- exp(-u^2/2)
  my_test.pred = sum(K*y)/sum(K)
  return(my_test.pred)
}

my_test.pred = sapply(test.x, function(X) my_est(X, x, y, 0.5))

# ksmooth
test.pred = ksmooth(x, y, kernel = "normal", bandwidth = 0.5/0.3706506, x.points = test.x)

# ggplot
library(ggplot2)
test = data.frame(cbind(test.x, test.y))
test.fit = data.frame(cbind(test.pred$x, test.pred$y))
my_test.fit = data.frame(cbind(test.x, my_test.pred))
# scatter plot
gplot = ggplot(test, aes(test$test.x, test$test.y))+ geom_point()+labs(x="test.x",y="test.y",title="toy data")+theme(plot.title = element_text(hjust = 0.5))
# ksmooth fit
gplot = gplot + geom_line(data=test.fit,aes(x=test.fit$X1,y=test.fit$X2,color = "ksmooth fit"), lwd = 1.5)
gplot
# my fit
gplot = gplot + geom_line(data=my_test.fit,aes(x=my_test.fit$test.x,y=my_test.fit$my_test.pred,color = "my fit"), lwd = 1.5)
gplot = gplot + scale_colour_manual(values = c("deepskyblue", "darkorange"))
gplot


# b)
library(readr)
video <- read_csv("C:/Users/guoshuhui/Desktop/542/homework/hw4/Video_Games_Sales_as_at_22_Dec_2016.csv")
video <- na.omit(video)
y <- log(1+video$Global_Sales)
cs <- video$Critic_Score
cc <- video$Critic_Count
us <- as.numeric(video$User_Score)

set.seed(1)
n = dim(video)[1]
sample <- sample(sequence(n))
video_new <- video[sample,]

nrFolds <- 5
# generate array containing fold-number for each sample (row)
folds <- rep_len(1:nrFolds, nrow(video_new))

# do cross validation
# output the MSE
my_cv <- function(x, C){
  my_error <- c()
  for (k in 1:nrFolds){
    fold <- which(folds == k)
    train_x <- x[-fold]
    test_x <- x[fold]
    train_y <- y[-fold]
    test_y <- y[fold]
    test.pred = sapply(test_x, function(X) my_est(X, train_x, train_y, C))
    my_error = c(my_error, mean((test_y-test.pred)^2))
  }
  return(mean(my_error))
}

# bandwidth
# large interval
C <- seq(0.5, 20.5, 1)

error_cs <- c()
for (i in C){
  error_cs <- c(error_cs, my_cv(cs, i))
}

error_cc <- c()
for (i in C){
  error_cc <- c(error_cc, my_cv(cc, i))
}

error_us <- c()
for (i in C){
  error_us <- c(error_us, my_cv(us, i))
}

save(C, error_cs, error_cc, error_us, file="hw4_error.RData")

load("hw4_error.RData")

# plot
library(ggplot2)
large_cs = data.frame(cbind(C, error_cs))
gplot = ggplot(large_cs, aes(large_cs$C, large_cs$error_cs))+labs(x="bandwidth",y="error",title="Critic Score")+theme(plot.title = element_text(hjust = 0.5))
gplot = gplot + geom_line(data=large_cs,aes(x=large_cs$C,y=large_cs$error_cs),color = "deepskyblue", lwd = 1.5)
gplot

large_cc = data.frame(cbind(C, error_cc))
gplot = ggplot(large_cc, aes(large_cc$C, large_cc$error_cc))+labs(x="bandwidth",y="error",title="Critic Count")+theme(plot.title = element_text(hjust = 0.5))
gplot = gplot + geom_line(data=large_cc,aes(x=large_cc$C,y=large_cc$error_cc),color = "deepskyblue", lwd = 1.5)
gplot

large_us = data.frame(cbind(C, error_us))
gplot = ggplot(large_us, aes(large_us$C, large_us$error_us))+labs(x="bandwidth",y="error",title="User Score")+theme(plot.title = element_text(hjust = 0.5))
gplot = gplot + geom_line(data=large_us,aes(x=large_us$C,y=large_us$error_us),color = "deepskyblue", lwd = 1.5)
gplot

# try smaller internal
C[which.min(error_cs)]
C_cs <- seq(0.5, 2.5, 0.1)
error2_cs <- c()
for (i in C_cs){
  error2_cs <- c(error2_cs, my_cv(cs, i))
}

C[which.min(error_cc)]
C_cc <- seq(3.5, 5.5, 0.1)
error2_cc <- c()
for (i in C_cc){
  error2_cc <- c(error2_cc, my_cv(cc, i))
}

C[which.min(error_us)]
C_us <- seq(0.1, 2.0, 0.1)
error2_us <- c()
for (i in C_us){
  error2_us <- c(error2_us, my_cv(us, i))
}

C_cs[which.min(error2_cs)]
C_cc[which.min(error2_cc)]
C_us[which.min(error2_us)]

# plot
small_cs = data.frame(cbind(C_cs, error2_cs))
gplot = ggplot(small_cs, aes(small_cs$C_cs, small_cs$error2_cs))+labs(x="bandwidth",y="error",title="Critic Score")+theme(plot.title = element_text(hjust = 0.5))
gplot = gplot + geom_line(data=small_cs,aes(x=small_cs$C_cs,y=small_cs$error2_cs),color = "deepskyblue", lwd = 1.5)
gplot

small_cc = data.frame(cbind(C_cc, error2_cc))
gplot = ggplot(small_cc, aes(small_cc$C_cc, small_cc$error2_cc))+labs(x="bandwidth",y="error",title="Critic Count")+theme(plot.title = element_text(hjust = 0.5))
gplot = gplot + geom_line(data=small_cc,aes(x=small_cc$C_cc,y=small_cc$error2_cc),color = "deepskyblue", lwd = 1.5)
gplot

small_us = data.frame(cbind(C_us, error2_us))
gplot = ggplot(small_us, aes(small_us$C_us, small_us$error2_us))+labs(x="bandwidth",y="error",title="User Score")+theme(plot.title = element_text(hjust = 0.5))
gplot = gplot + geom_line(data=small_us,aes(x=small_us$C_us,y=small_us$error2_us),color = "deepskyblue", lwd = 1.5)
gplot


## Question 2
# a)
rm(list=ls())

library(MASS)
library(randomForest)
set.seed(1)
P = 20
N = 200
V <- diag(P)
X = as.matrix(mvrnorm(N, mu=rep(0,P), Sigma=V))

f = 1+0.5*(X[,1]+X[,2]+X[,3]+X[,4])

# fix ntree, tune mtry and nodesize
myfun <- function(times, mtry, nodesize){
  y <- matrix(0, times, N)
  yhat <- matrix(0, times, N)  #matrix of the estimations
  for(i in 1:times){
    Y = f+rnorm(N)
    y[i,] = Y
    
    rf.fit = randomForest(X, Y, mtry = mtry, nodesize = nodesize)
    yhat[i,] = predict(rf.fit, X)
  }
  
  #Calculate the covariance
  Ybar = f
  cov <- c()
  for (i in 1:N){
    cov[i] = sum((yhat[,i]-mean(yhat[,i]))%*%(y[,i]-Ybar[i]))/(times-1)
  }
  covariance = sum(cov)
  return(covariance)
}

set.seed(1)
mtry <- c(1, 5, 15)
nodesize <- c(5, 15, 25)

df <- matrix(0, length(mtry), length(nodesize))
for(i in 1:length(mtry)){
  for(j in 1:length(nodesize)){
    df[i,j] <- myfun(20, mtry[i], nodesize[j])
  }
}

rownames(df) <- c("mtry=1", "mtry=5", "mtry=15")
colnames(df) <- c("nodesize=5", "nodesize=15", "nodesize=25")


# b)
library(MASS)
library(randomForest)
set.seed(1)
P = 20
N = 200
V <- diag(P)
X = as.matrix(mvrnorm(N, mu=rep(0,P), Sigma=V))

f = 1+0.5*(X[,1]+X[,2]+X[,3]+X[,4])
Y = f+rnorm(N)

# fix mtry and nodesize, tune ntree
myfun2 <- function(times, ntree){
  yhat <- matrix(0, times, N)  #matrix of the estimations
  for(i in 1:times){
    rf.fit = randomForest(X, Y, ntree = ntree)
    yhat[i,] = predict(rf.fit, X)
  }
  E_yhat = apply(yhat, 2, mean)
  dif = t(yhat)-E_yhat
  E = apply(dif^2, 1, mean)
  varr = mean(E)
  return(varr)
}

set.seed(1)
ntree <- seq(20, 200, 20)
variance = sapply(ntree, function(ntree) myfun2(20, ntree))

library(ggplot2)
data <- data.frame(ntree, variance)
gplot = ggplot(data, aes(data$ntree, data$variance))+ geom_point()+labs(x="ntree",y="variance",title="variances of the estimator using each value of ntree")+theme(plot.title = element_text(hjust = 0.5))
gplot


## Question 3

# a)
rm(list=ls())

# fit the stump model with instruction in assignment
stump <- function(x, y, w){
  n <- length(x)
  xnew <- sort(x)
  ynew <- y[order(x)]
  wnew <- w[order(x)]
  
  score <- c()
  fleft <- c()
  fright <- c()
  for (i in 1:n){
    # left
    xleft <- xnew[1:i]
    yleft <- ynew[1:i]
    wleft <- wnew[1:i]
    pleft = sum(wleft[yleft == 1])/sum(wleft)
    Gleft = pleft*(1-pleft)
    
    # right
    xright <- xnew[(i+1):n]
    yright <- ynew[(i+1):n]
    wright <- wnew[(i+1):n]
    pright = sum(wright[yright ==1])/sum(wright)
    Gright = pright*(1-pright)
    
    # score
    score[i] = -(sum(wleft)/sum(w))*Gleft-(sum(wright)/sum(w))*Gright
    
    # predictions
    fleft[i] <- (pleft >= 0.5)-(pleft < 0.5)
    fright[i] <- (pright >= 0.5)-(pright < 0.5)
  }
  # find the output
  c <- xnew[which.max(score)]
  fl <- fleft[which.max(score)]
  fr <- fright[which.max(score)]
  
  return(c(c, fl, fr))  
}


# b)

# training model
# output the required parameter in model
adaboost1 <- function(x.train, y.train, step){
  n <- length(x.train)
  w <- rep(1, n)/n
  
  c <- c()
  fl <- c()
  fr <- c()
  alpha <- c()
  for (t in 1:step){
    output <- stump(x.train, y.train, w)
    c[t] <- output[1]
    fl[t] <- output[2]
    fr[t] <- output[3]
    f <- (x.train <= c[t])*fl[t]+(x.train > c[t])*fr[t]
    epsilon = sum(w[y.train != f])
    alpha[t] = log((1-epsilon)/epsilon)/2
    z = sum(w*exp(-alpha[t]*y.train*f))
    w <- w*exp(-alpha[t]*y.train*f)/z
  }
  
  return(list(alpha = alpha, left = fl, right = fr, split = c))
}

# output the final model
adaboost2 <- function(x, ada_model, step){
  alpha = ada_model$alpha
  fl = ada_model$left
  fr = ada_model$right
  c = ada_model$split
  
  # final model
  fit <- rep(0,length(x))
  for (t in 1:step){
    temp = alpha[t]*((x <= c[t])*fl[t]+(x > c[t])*fr[t])
    fit = fit+temp
  }
  
  return(fit)
}

# generate data
set.seed(1)
n = 300
x = runif(n)
y = (rbinom(n, 1, (sin(4*pi*x)+1)/2)-0.5)*2

step1 = 100
ada_model <- adaboost1(x, y, step1)

fit1 = adaboost2(x, ada_model, 1)
fit2 = adaboost2(x, ada_model, 10)
fit3 = adaboost2(x, ada_model, 30)
fit4 = adaboost2(x, ada_model, 100)

p <- (sin(4*pi*x)+1)/2

# plot
library(ggplot2)
data1 <- data.frame(x = x, my_p = 1/(1+exp(-2*fit1)), p = p)
gplot1 <- ggplot(data1)+geom_line(aes(x = sort(x), y = p[order(x)]), size = 2, color = "grey")
gplot1 <- gplot1+geom_line(aes(x = sort(x), y = my_p[order(x)]), size = 1, color = "red")
gplot1 <- gplot1+labs(title = "iteration=1", y = "")+theme(plot.title = element_text(hjust = 0.5))

data2 <- data.frame(x = x, my_p = 1/(1+exp(-2*fit2)), p = p)
gplot2 <- ggplot(data2)+geom_line(aes(x = sort(x), y = p[order(x)]), size = 2, color = "grey")
gplot2 <- gplot2+geom_line(aes(x = sort(x), y = my_p[order(x)]), size = 1, color = "red")
gplot2 <- gplot2+labs(title = "iteration=10", y = "")+theme(plot.title = element_text(hjust = 0.5))

data3 <- data.frame(x = x, my_p = 1/(1+exp(-2*fit3)), p = p)
gplot3 <- ggplot(data3)+geom_line(aes(x = sort(x), y = p[order(x)]), size = 2, color = "grey")
gplot3 <- gplot3+geom_line(aes(x = sort(x), y = my_p[order(x)]), size = 1, color = "red")
gplot3 <- gplot3+labs(title = "iteration=30", y = "")+theme(plot.title = element_text(hjust = 0.5))

data4 <- data.frame(x = x, my_p = 1/(1+exp(-2*fit4)), p = p)
gplot4 <- ggplot(data4)+geom_line(aes(x = sort(x), y = p[order(x)]), size = 2, color = "grey")
gplot4 <- gplot4+geom_line(aes(x = sort(x), y = my_p[order(x)]), size = 1, color = "red")
gplot4 <- gplot4+labs(title = "iteration=100", y = "")+theme(plot.title = element_text(hjust = 0.5))

cowplot::plot_grid(gplot1, gplot2, gplot3, gplot4, align = "v")


# generate training data
set.seed(1)
n = 300
x.train = runif(n)
y.train = (rbinom(n, 1, (sin(4*pi*x.train)+1)/2)-0.5)*2

# generate testing data
x.test = runif(n)
y.test = (rbinom(n, 1, (sin(4*pi*x.test)+1)/2)-0.5)*2

# calculate fitting results of training and testing data
step_new <- 2000
ada.model_new = adaboost1(x.train, y.train, step_new)

fit.train <- matrix(0, step_new, n)
fit.test <- matrix(0, step_new, n)
for (i in 1:step_new){
  fit.train[i,] = adaboost2(x.train, ada.model_new, i)
  fit.test[i,] = adaboost2(x.test, ada.model_new, i)
}

fit.train = sign(fit.train)
fit.test = sign(fit.test)

# calculate the misclassification error
error.train <- c()
error.test <- c()
for (i in 1:step_new){
  error.train[i] = sum(fit.train[i,] != y.train)/300
  error.test[i] = sum(fit.test[i,] != y.test)/300
}

library(ggplot2)
error <- data.frame(step = seq(1, step_new, 1), error.train = error.train, error.test = error.test)
gplot <- ggplot(error)+geom_line(aes(x = step, y = error.train, color = "training"), size = 1)
gplot <- gplot+geom_line(aes(x = step, y = error.test, color = "testing"), size = 1)
gplot <- gplot+labs(x = "iteration",y = "misclassification error")
gplot <- gplot+labs(title = "Misclassification error of training and testing data")+theme(plot.title = element_text(hjust = 0.5))
gplot <- gplot + scale_colour_manual(values = c("deepskyblue", "darkorange"))
gplot
