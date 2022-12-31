###########################################################################
############################################################################
### Boston data set predcition problem
###################################################################
install.packages('MASS')
library(MASS)

# Data exploration
?Boston
dim(Boston) #506x14
head(Boston) # display first 6 lines 

sum(is.na(Boston)) # check for missing values 

# Goal: Create a model to predict per capita crime rate in Boston 
# First produce a pair plot to have graphical intuition about the learning problem. 
pairs(Boston)

# Create a training set and testing set 
set.seed(345)
Boston.sampler <- sample(1:nrow(Boston),nrow(Boston)/2)  
Boston.train <- Boston[Boston.sampler,] # use 50 % of the data for training
Boston.test <- Boston[-Boston.sampler,] # # use 50 % of the data for testing

###################################################################################
# Fit a multiple linear model chosen by best subset selection on the training set
###################################################################################
install.packages('leaps')
library(leaps)
bestsubsets <- regsubsets(crim~., data=Boston.train, nvmax=13)
results = summary(bestsubsets) # 13 best models fitted where each model has a different number 
# of predictors

# Extract Adjusted r squared, cp , bic, rsq, and for different models. 
Cp <-  results$cp
adjR <- results$adjr2
BIC <- results$bic
rsq <- results$rsq

A <- cbind(Cp,adjR,BIC,rsq) # bind values to create a matrix

apply(A[,c(1,3)],MARGIN = 2,FUN = which.min) # find which model has lowest Cp, and BIC . 

# 5th model has lowest Cp and 3rd model has lowest BIC 

apply(A[,c(2,4)],MARGIN = 2,FUN = which.max) # find which model has highest adjusted r squared and r squared

# 6th model has highest adjusted R squared 

# Generate plots of the above selection criteria against the number of predictors 
par(mfrow=c(1,3))
plot(Cp, main= 'Cp', xlab='predictors', ylab='Cp values', type = 'l', col='blue', lwd=2)
points(5,Cp[5], pch=8, col= 'red', cex=2)
plot(BIC, main= 'BIC', xlab='predictors', ylab='BIC values', type = 'l', col='blue', lwd=2)
points(3,BIC[3], pch=8, col= 'red', cex=2)
plot(adjR, main= 'AdjRsq', xlab='predictors', ylab='Adj Rsquared values', type = 'l', col='blue', lwd=2)
points(6,adjR[6], pch=8, col='red', cex=2)

#Taking the model with the lowest Cp, that is the 5th model (with 5 predictors and intercept)
Beta_vec <- coef(bestsubsets,5) # estimated beta vector
test.predictions <- model.matrix(crim~., data= Boston.test)[,names(Beta_vec)]%*%Beta_vec # test predictions

# Other method:  write a function to make predictions using regsubsets
predict.regsubsets <- function(object, newdata, id, ...) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

predict.regsubsets(bestsubsets,Boston.test,5)
# check if predicted values between the two above methods are the same 
predict.regsubsets(bestsubsets,Boston.test,5)== test.predictions
# They are indeed true

#Testing error for best subset selection multiple linear regression
MSE.TEST <- mean((Boston.test$crim-test.predictions)^2)
MSE.TEST # 40.42511   

#########################################################################################
# Now using lasso regression where lambda is tuned by cross validation on the training set
##########################################################################################

install.packages('glmnet')
library(glmnet)

x.boston.matrix <- model.matrix(crim~., data= Boston.train)[,-1]
x.boston.matrix.test <- model.matrix(crim~., data= Boston.test)[,-1]

lasso.boston <- cv.glmnet(x.boston.matrix,Boston.train$crim, alpha=1)
lambda.boston <- lasso.boston$lambda.min # 0.01739447 optimal value of lambda according to cross validation mean squared error


test.lasso.pred <- predict(lasso.boston, newx=x.boston.matrix.test, s= lambda.boston)

# Test MSE 
MSE.TEST.LASSO <- mean((Boston.test$crim-test.lasso.pred)^2)
MSE.TEST.LASSO # 43.14157

############################################################################################
# Now using ridge  regression where lambda is tuned by cross validation on the training set
#############################################################################################

ridge.boston <- cv.glmnet(x.boston.matrix,Boston.train$crim, alpha=0)
lambda.boston.ridge <- lasso.boston$lambda.min # 0.01739447 optimal value of lambda according to cross validation mean squared error

par(mfrow=c(1,1))
plot(ridge.boston, xvar='lambda')
test.ridge.pred <- predict(ridge.boston, newx=x.boston.matrix.test, s= lambda.boston.ridge)

# Test MSE 
MSE.TEST.ridge <- mean((Boston.test$crim-test.ridge.pred)^2)
MSE.TEST.ridge # 42.5929

################################################
# Now using principal component regression 
################################################

library(pls)
set.seed(234)

Pcr.boston <- pcr(crim~., data= Boston.train, scale=TRUE, validation='CV')
summary(Pcr.boston)

# Plot CV Mean squared prediction error against the number of principal components
validationplot(Pcr.boston, val.type = 'MSEP')


#Extract the number of principal components with lowest cross validation 
# error 
min.pcr <- which.min(MSEP(Pcr.boston)$val[1,1,])-1
# The model with 13 components minimise the CV

# Now, we can generate predictions using the predict function for 13 principal components 
PRC.BOSTON <- predict(Pcr.boston, Boston.test,ncomp = 13)

# Testing MSE 
test.MSE.Boston.pcr <- mean((Boston.test$crim - PRC.BOSTON)^2)
test.MSE.Boston.pcr # 50.62118

# In conclusion, one observes that the model that had the highest performance
# on the testing set is multiple linear regression using best subset selection and selecting the best model using Mallow Cp . One notices that 
# the higher test error is given by principal component regression although dimensionality of the inputs space has not been changed. 


