bitcoin = read.table("C:/Users/guoshuhui/Desktop/542/homework/hw2/bitcoin_dataset.csv", header = T, sep=",")
bitcoin = bitcoin[,-5]
bitcoin$Date = as.numeric(bitcoin$Date)

X = bitcoin[,-2]
Y = bitcoin$btc_market_price
X_train = X[1:1460,]
X_test = X[1461:1588,]
Y_train = Y[1:1460]
Y_test = Y[1461:1588]
n_train = 1460
n_test = 128

library(leaps)
x = as.matrix(X_train)
leaps = regsubsets(x, Y_train, nvmax = 22)
sumleaps = summary(leaps, matrix = T)
msize=apply(sumleaps$which,1,sum)
n=dim(X_train)[1]
Cp = sumleaps$cp
BIC = sumleaps$bic
AIC = n*log(sumleaps$rss/n) + 2*msize

#calculate prediction error
#Cp
min_Cp = which.min(Cp)
X_train_Cp = X_train[,which(sumleaps$which[min_Cp,] == 1)-1]
X_test_Cp = X_test[,which(sumleaps$which[min_Cp,] == 1)-1]
lm_train_Cp = lm(Y_train~., data = X_train_Cp)
Yhat_test_Cp = predict(lm_train_Cp, X_test_Cp)
error_Cp = sum((Yhat_test_Cp-Y_test)^2)/n_test

#AIC
min_AIC = which.min(AIC)
X_train_AIC = X_train[,which(sumleaps$which[min_AIC,] == 1)-1]
X_test_AIC = X_test[,which(sumleaps$which[min_AIC,] == 1)-1]
lm_train_AIC = lm(Y_train~., data = X_train_AIC)
Yhat_test_AIC = predict(lm_train_AIC, X_test_AIC)
error_AIC = sum((Yhat_test_AIC-Y_test)^2)/n_test

#BIC
min_BIC = which.min(BIC)
X_train_BIC = X_train[,which(sumleaps$which[min_BIC,] == 1)-1]
X_test_BIC = X_test[,which(sumleaps$which[min_BIC,] == 1)-1]
lm_train_BIC = lm(Y_train~., data = X_train_BIC)
Yhat_test_BIC = predict(lm_train_BIC, X_test_BIC)
error_BIC = sum((Yhat_test_BIC-Y_test)^2)/n_test

#c)
X = bitcoin[,-2]
Y = bitcoin$btc_market_price
X_train = X[1:1460,]
X_test = X[1461:1588,]
Y_train = log(1+bitcoin$btc_market_price)[1:1460]
Y_test = Y[1461:1588]
n_train = 1460
n_test = 128

library(leaps)
x = as.matrix(X_train)
leaps = regsubsets(x, Y_train, nvmax = 22)
sumleaps = summary(leaps, matrix = T)
msize=apply(sumleaps$which,1,sum)
n=dim(X_train)[1]
Cp = sumleaps$cp
BIC = sumleaps$bic
AIC = n*log(sumleaps$rss/n) + 2*msize

#calculate prediction error
#Cp
min_Cp = which.min(Cp)
X_train_Cp = X_train[,which(sumleaps$which[min_Cp,] == 1)-1]
X_test_Cp = X_test[,which(sumleaps$which[min_Cp,] == 1)-1]
lm_train_Cp = lm(Y_train~., data = X_train_Cp)
Yhat_test_Cp = predict(lm_train_Cp, X_test_Cp)
Yorihat_test_Cp = exp(Yhat_test_Cp)-1
error_Cp = sum((Yorihat_test_Cp-Y_test)^2)/n_test

#AIC
min_AIC = which.min(AIC)
X_train_AIC = X_train[,which(sumleaps$which[min_AIC,] == 1)-1]
X_test_AIC = X_test[,which(sumleaps$which[min_AIC,] == 1)-1]
lm_train_AIC = lm(Y_train~., data = X_train_AIC)
Yhat_test_AIC = predict(lm_train_AIC, X_test_AIC)
Yorihat_test_AIC = exp(Yhat_test_AIC)-1
error_AIC = sum((Yorihat_test_AIC-Y_test)^2)/n_test

#BIC
min_BIC = which.min(BIC)
X_train_BIC = X_train[,which(sumleaps$which[min_BIC,] == 1)-1]
X_test_BIC = X_test[,which(sumleaps$which[min_BIC,] == 1)-1]
lm_train_BIC = lm(Y_train~., data = X_train_BIC)
Yhat_test_BIC = predict(lm_train_BIC, X_test_BIC)
Yorihat_test_BIC = exp(Yhat_test_BIC)-1
error_BIC = sum((Yorihat_test_BIC-Y_test)^2)/n_test



#Problem2
#Part I
library(MASS)
library(glmnet)
set.seed(1)
N = 400
P = 20

Beta = c(1:5/5, rep(0, P-5))
Beta0 = 0.5

# genrate X
V = matrix(0.5, P, P)
diag(V) = 1
X = as.matrix(mvrnorm(N, mu = 3*runif(P)-1, Sigma = V))

# create artifical scale of X
X = sweep(X, 2, 1:10/5, "*")

# genrate Y
y = Beta0 + X %*% Beta + rnorm(N)

# check OLS
lm(y ~ X)

# now start the Lasso 
# First we scale and center X, and record them.
# Also center y and record it. dont scale it. 
# now since both y and X are centered at 0, we don't need to worry about the intercept anymore. 
# this is because for any beta, X %*% beta will be centered at 0, so no intercept is needed. 
# However, we still need to recover the real intercept term after we are done estimating the beta. 
# The real intercept term can be recovered by using the x_center, x_scale, y2, and the beta parameter you estimated.
# There are other simpler ways to do it too, if you think carefully. 

x_center = colMeans(X)
x_scale = apply(X, 2, sd)
X2 = scale(X)

bhat = rep(0, ncol(X2)) # initialize it
ymean = mean(y)
y2 = y - ymean

# now start to write functions 
# prepare the soft thresholding function (should be just one line, or a couple of)

soft_th <- function(b, pen)
{
  if(abs(b)>pen) s = sign(b)*(abs(b)-pen)  else s=0
  return(s)
}

# initiate lambda. This is one way to do it, the logic is that I set the first lambda as the largetst gradient. 
# if you use this formula, you will need to calculate this for the real data too.

lambda = exp(seq(log(max(abs(cov(X2, y2)))), log(0.001), length.out = 100))

# you should write the following function which can be called this way 
#LassoFit(X2, y2, mybeta = rep(0, ncol(X2)), mylambda = lambda[10])

LassoFit <- function(myX, myY, mybeta, mylambda, tol = 1e-10, maxitr = 500)
{
  # initia a matrix to record the objective function value
  N = dim(myX)[1]
  f = rep(0, maxitr)
  
  for (k in 1:maxitr)
  {
    
    # compute residual
    r = myY - myX %*% mybeta
    
    # I need to record the residual sum of squares
    f[k] = mean(r*r)
    
    for (j in 1:ncol(myX))
    {
      # add the effect of jth variable back to r 
      # so that the residual is now the residual after fitting all other variables
      r_new = myY - myX %*% mybeta
      beta_new = sum(r_new*myX[,j])/N + mybeta[j]
      
      # apply the soft thresholding function to the ols estimate of the jth variable 
      mybeta[j] <- soft_th(beta_new, mylambda)
    }
    
    if (k > 10)
    {
      # this is just my adhoc way of stoping rule, you dont have to use it
      if (sum(abs(f[(k-9):k] - mean(f[(k-9):k]))) < tol) break;
    }
  }
  return (mybeta)
}

# you should test your function on a large lambda (penalty) level. 
# this should produce a very spase model. 
# keep in mind that these are not the beta in the original scale of X

LassoFit(X2, y2, mybeta = rep(0, ncol(X2)), mylambda = lambda[10], tol = 1e-7, maxitr = 500)

# now initiate a matrix that records the fitted beta for each lambda value 

beta_all = matrix(NA, ncol(X), length(lambda))

# this vecter stores the intercept of each lambda value
beta0_all = rep(NA, length(lambda))

# this part gets pretty tricky: you will initial a zero vector for bhat, 
# then throw that into the fit function using the largest lambda value. 
# that will return the fitted beta, then use this beta on the next (smaller) lambda value
# iterate until all lambda values are used

bhat = rep(0, ncol(X2)) # initialize it

for (i in 1:length(lambda)) # loop from the largest lambda value
{
  # if your function is correct, this will run pretty fast
  bhat = LassoFit(X2, y2, bhat, lambda[i])
  
  # this is a tricky part, since your data is scaled, you need to figure out how to scale that back 
  # save the correctly scaled beta into the beta matrix 
  beta_all[, i] = bhat/x_scale
  
  # here, you need to figure out a way to recalculte the intercept term in the original, uncentered and unscaled X
  beta0_all[i] = ymean - sum((x_center/x_scale)*bhat)
}


# now you have the coefficient matrix 
# each column correspond to one lambda value 
rbind("intercept" = beta0_all, beta_all)

# you should include a similar plot like this in your report
# feel free to make it look better
matplot(colSums(abs(beta_all)), t(beta_all), type="l", xlab = "L1 Norm", ylab = "Coefficients")



# The following part provides a way to check your code. 
# You do not need to include this part in your report. 

# However, keep in mind that my original code is based on formula (3)
# if you use other objective functions, it will be different, and the results will not match

# load the glmnet package and get their lambda 
library(glmnet)

# this plot should be identical (close) to your previous plot
plot(glmnet(X, y))

# set your lambda to their lambda value and rerun your algorithm 
lambda = glmnet(X, y)$lambda

beta_all = matrix(NA, ncol(X), length(lambda))
beta0_all = rep(NA, length(lambda))
bhat = rep(0, ncol(X2)) # initialize it

for (i in 1:length(lambda))
{
  bhat = LassoFit(X2, y2, bhat, lambda[i])
  beta_all[, i] = bhat/x_scale
  beta0_all[i] = ymean - sum((x_center/x_scale)*bhat)
}

# then this distance should be pretty small 
# my code gives distance no more than 0.01
max(abs(beta_all - glmnet(X, y)$beta))
max(abs(beta0_all - glmnet(X, y)$a0))



#Part II
bitcoin = read.table("C:/Users/guoshuhui/Desktop/542/homework/hw2/bitcoin_dataset.csv", header = T, sep=",")
bitcoin = bitcoin[,-5]
bitcoin$Date = as.numeric(bitcoin$Date)

X = bitcoin[,-2]
Y = bitcoin$btc_market_price
X_train = X[1:1460,]
X_test = X[1461:1588,]
Y_train = Y[1:1460]
Y_test = Y[1461:1588]
n_train = 1460
n_test = 128

X_train_center = colMeans(X_train)
X_train_scale = apply(X_train, 2, sd)
X2_train = scale(X_train)

Ymean_train = mean(Y_train)
Y2_train = Y_train - Ymean_train

# initiate lambda
lambda = exp(seq(log(max(abs(cov(X2_train, Y2_train)))), log(0.001), length.out = 100))

beta_all = matrix(NA, ncol(X_train), length(lambda))
beta0_all = rep(NA, length(lambda))
bhat = rep(0, ncol(X2_train))

for (i in 1:length(lambda))
{
  bhat = LassoFit(X2_train, Y2_train, bhat, lambda[i])
  beta_all[, i] = bhat/X_train_scale
  beta0_all[i] = Ymean_train - sum((X_train_center/X_train_scale)*bhat)
}

coef = rbind("intercept" = beta0_all, beta_all)
X_test_new = cbind(rep(1,n_test), X_test)
coef = as.matrix(coef)
X_test_new = as.matrix(X_test_new)
Yhat_test = X_test_new %*% coef

error_test <- c()
for (i in 1:100) {
  error_test[i] = sum((Y_test-Yhat_test[,i])^2)/n_test
}

plot(error_test, type = "l",ylim=c(0,100000),xlab = "lambda", ylab = "prediction error")

lambda_best = which.min(error_test)
error_test_min = error_test[lambda_best]

colnames(X_test[which(coef[,lambda_best]!= 0)-1])
coef[which(coef[,lambda_best]!= 0),lambda_best]
