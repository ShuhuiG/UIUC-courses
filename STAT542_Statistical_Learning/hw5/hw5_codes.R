## Question 2

# This function is to compute sigma^{1/2}
matrix_sqrt <- function(X, symmetric = FALSE) {
  # Perform the spectral decomposition.
  # Covariance matrices are symmetric
  X_eig <- eigen(X, symmetric = symmetric)
  
  # extract the Q eigen-vector matrix and the eigen-values
  Q <- X_eig$vectors
  values <- X_eig$values
  Q_inv <- solve(Q) # Q^{-1}
  
  Q %*% diag(sqrt(values)) %*% Q_inv
}

my_SIR <- function(X, Y, H){
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  E <- t(matrix(rep(colMeans(X), n), p, n))
  sigma <- t(X-E) %*% (X-E)/n
  # compute sigma^{-1/2}
  sigma_inv <- solve(matrix_sqrt(sigma, symmetric = TRUE))
  
  Z <- (X-E) %*% sigma_inv
  
  # sort the dataset (Z, Y) by the observed Y values
  Z = Z[order(Y), ]
  Y = Y[order(Y)]
  
  nrFolds <- H
  # generate array containing fold-number for each sample (row)
  folds <- split(seq_len(n),rep(1:H, each = floor(n/H), length.out=n))
  
  zh <- matrix(NA,H, p)
  nh <- c()
  for (i in 1:H){
    zh[i, ] <- colMeans(Z[folds[[i]],])
    nh[i] <- length(folds[[i]])
  }
  
  M <- t(zh) %*% apply(zh, 2, "*", nh)/sum(nh) #z_bar equals to 0, so it does not need to be here
  
  M_eig <- eigen(M, symmetric = TRUE)
  M_eigvec <- M_eig$vectors
  trans <- sigma_inv %*% M_eigvec
  
  # scale the trans, because the function 'dr' inculdes the scale process
  result <- scale(trans, center = FALSE, scale = sqrt(colSums(trans^2)))[1:p, 1:p]
  return(result)
}

set.seed(1)
n = 1000; p = 10
x = matrix(rnorm(n*p), n, p)
b = matrix(c(1, 1, rep(0, p-2)))
y = 0.125*(x %*% b)^3 + 0.5*rnorm(n)

my_SIR(x, y, 10)
library(dr)
fit = dr(y~., data = data.frame(x, y), method = "sir", nslices=10)
fit$evectors

# a)
# generate toy data
set.seed(1)
n = 1000; p = 10
x = matrix(rnorm(n*p), n, p)
b = matrix(c(1, 0, 0, 1, rep(0, p-4)))
y = 5*sin(x %*% b) + rnorm(n)

est1 = my_SIR(x, y, 10)
library(dr)
fit1 = dr(y~., data = data.frame(x, y), method = "sir", nslices=10)
fit1$evectors

# b)
set.seed(1)
x = matrix(rnorm(n*p), n, p)
b = matrix(c(1, 1, 1,rep(0, p-3)))
y = 3*cos^2(x %*% b) + rnorm(n)

est2 = my_SIR(x, y, 10)
fit2 = dr(y~., data = data.frame(x, y), method = "sir", nslices=10)
fit2$evectors


## Question 3

rm(list=ls())

### predict 'revenue'
load("use_data.RData")
data_temp <- data[, -(1:20)]
data_use <- cbind(data_total[c(1,4,13,14)], data_temp)

data_use <- data_use[data_use$revenue != 0, ]
Y <- data_use$revenue
X <- data_use[, -c(2, 3)]

# there are two numeric variables 'budget' and 'runtime' in X, set them as numeric
X[,1] <- as.numeric(X[,1])
X[,2] <- as.numeric(X[,2])

# odd id as training, even id as testing
Y_training <- Y[data_use$id %% 2 == 1]
Y_testing <- Y[data_use$id %% 2 == 0]
X_training <- X[data_use$id %% 2 == 1, ]
X_testing <- X[data_use$id %% 2 == 0, ]

# select the variables whose sum values are more than one
# since they have meaning under this circumstance
X_testing <- X_testing[,colSums(X_training) > 1]
X_training <- X_training[,colSums(X_training) > 1]

# calculate the correlation between X and Y training data
# select the training x whose correlation with y is not less than 0.1
# select the testing x according to the previous step
r <- abs(cor(Y_training, X_training))
X_training <- X_training[, r >= 0.1]
X_testing <- X_testing[, r >= 0.1]

## lasso
library(glmnet)
X_training <- as.matrix(X_training)
X_testing <- as.matrix(X_testing)
m1.cv <- cv.glmnet(X_training, Y_training, alpha = 1, nfolds = 10)

# save lambda and fit lasso with this lambda
lambda <- m1.cv$lambda.min
m1 <- glmnet(X_training, Y_training, alpha = 1, lambda = lambda)
Y_pre1 <- predict(m1, X_testing)
error_m1 <- mean(abs(Y_pre1-Y_testing))

# plot
library(ggplot2)
data_m1 <- data.frame(x = seq(1,length(Y_pre1),1), s0 = Y_pre1, s1 = Y_testing)
gplot <- ggplot(data_m1) + geom_line(aes(x = x, y = s1, color = "true value"))
gplot <- gplot + geom_line(aes(x = x, y = s0, color = "predicted value"))
gplot <- gplot + labs(x = "sequence",y = "revenue")
gplot <- gplot + labs(title = "lasso") + theme(plot.title = element_text(hjust = 0.5))
gplot <- gplot + scale_colour_manual(values = c("deepskyblue", "darkorange"))
gplot

## random forest
library(randomForest)
m2 = randomForest(X_training, Y_training, ntree = 1000, mtry = 139/3, nodesize = 5) # mtry=p/3
Y_pre2 <- predict(m2, X_testing)
error_m2 <- mean(abs(Y_pre2-Y_testing))

# plot
data_m2 <- data.frame(x = seq(1,length(Y_pre2),1), s0 = Y_pre2, s1 = Y_testing)
gplot <- ggplot(data_m2) + geom_line(aes(x = x, y = s1, color = "true value"))
gplot <- gplot + geom_line(aes(x = x, y = s0, color = "predicted value"))
gplot <- gplot + labs(x = "sequence",y = "revenue")
gplot <- gplot + labs(title = "random forest") + theme(plot.title = element_text(hjust = 0.5))
gplot <- gplot + scale_colour_manual(values = c("deepskyblue", "darkorange"))
gplot

# plot the variable importance
varImpPlot(m2)

### predict 'vote_average'
rm(list=ls())
load("use_data.RData")
data2_temp <- data[, -(1:20)]
data2_use <- cbind(data_total[c(1,4,19,14)], data2_temp)

Y <- data2_use$vote_average
Y <- as.numeric(paste(Y))

# set vote_average>7 as 1, <7 as 0
Y <- as.numeric(Y > 7)
X <- data2_use[, -c(2, 3)]

# there are two numeric variables 'budget' and 'runtime' in X, set them as numeric
X[,1] <- as.numeric(X[,1])
X[,2] <- as.numeric(X[,2])

# odd id as training, even id as testing
Y_training <- Y[data2_use$id %% 2 == 1]
Y_testing <- Y[data2_use$id %% 2 == 0]
X_training <- X[data2_use$id %% 2 == 1, ]
X_testing <- X[data2_use$id %% 2 == 0, ]

# select the variables whose sum values are more than one
# since they have meaning under this circumstance
X_testing <- X_testing[,colSums(X_training) > 1]
X_training <- X_training[,colSums(X_training) > 1]

# calculate the correlation between X and Y training data
# select the training x whose correlation with y is not less than 0.1
# select the testing x according to the previous step
r <- abs(cor(Y_training, X_training))
X_training <- X_training[, r >= 0.05]
X_testing <- X_testing[, r >= 0.05]

# there are much more 0 than 1, so we should balance these unbalanced data
Y_training1 <- Y_training[Y_training == 1]
X_training1 <- X_training[Y_training == 1,]
Y_training0 <- Y_training[Y_training == 0]
X_training0 <- X_training[Y_training == 0,]

set.seed(1)
temp <- sample(seq(1,length(Y_training0),1), 1.8*length(Y_training1))
Y_training <- as.factor(c(Y_training1, Y_training0[temp]))
X_training <- rbind(X_training1, X_training0[temp,])

## lasso
library(glmnet)
X_training <- as.matrix(X_training)
X_testing <- as.matrix(X_testing[-c(1308,2033), ]) #the rows of 1308 and 2033 will introduce NA, so remove them
Y_testing <- Y_testing[-c(1308,2033)]
m1.cv <- cv.glmnet(X_training, Y_training, family=c("binomial"), alpha = 1, nfolds = 10)

# save lambda and fit lasso with this lambda
lambda <- m1.cv$lambda.min
m1 <- glmnet(x=X_training, y=Y_training, family=c("binomial"), alpha = 1, lambda = lambda)
Y_pre1 <- predict(m1, X_testing, type = "class")
error_m1 <- mean(Y_pre1 != Y_testing)

## SVM
library(e1071)

cost <- seq(1,31,5)
error <- c()
for(i in cost){
  m2.fit <- svm(y ~ ., data = data.frame(X_training, y=Y_training),type='C-classification', kernel='linear', scale=FALSE, cost = i)
  Y_pre2 <- predict(m2.fit, X_testing)
  error <- c(error, mean(Y_pre2 != Y_testing))
}

m2 <- svm(y ~ ., data = data.frame(X_training, y=Y_training), type='C-classification', kernel='linear',scale=FALSE, cost = cost[which.min(error)])
Y_prem2 <- predict(m2, X_testing)
error_m2 <- mean(Y_prem2 != Y_testing)

## random forest
library(randomForest)
m3 = randomForest(X_training, Y_training, ntree = 1000, mtry = 351/3, nodesize = 5)
Y_pre3 <- predict(m3, X_testing)
error_m3 <- mean(Y_pre3 != Y_testing)

## select the important predictors
## since the first one is intercept and our main focus is on the variables
## so I removed the intercept and set the sequence as 2:21
as.vector(coef(m1)@Dimnames[[1]][order(coef(m1))[2:21]])

### predict star war 8

library(randomForest)
m2 = randomForest(X_training, Y_training, ntree = 1000, mtry = 139/3, nodesize = 5) # mtry=p/3
starwar <- X_testing[1,]
starwar[1,1:139] <- 0
starwar$budget <- 245000000
starwar$runtime <- 152
starwar$Action <- 1
starwar$Fantasy <- 1
starwar$Adventure <- 1
starwar_pre <- as.numeric(predict(m2, starwar))
