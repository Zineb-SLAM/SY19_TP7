#***************************************#
#              Libraries                #
#***************************************#

library('nnet')
library('MASS')
library('e1071')
library('pls')
library('caret')
library('FNN')
library("kknn")
library("tree")
library("randomForest")

# Convenient Function
printf <- function(...) invisible(print(sprintf(...)))

# Performance measure
perfMeasure = function(pred, class, trace = FALSE) {
  t = table(pred, class)
  if (trace) {
    print(t)
  }
  p = 1 - sum(diag(t)) / length(class)
  return(p)
}

#***************************************#
#              Load Data                #
#***************************************#
setwd("~/Desktop/TP2")
data = load('data_expressions.RData')
X.clean = X[, colSums(X)!=0] / 255
p = ncol(X.clean)
n = nrow(X.clean)

#***************************************#
#              Visualize                #
#***************************************#

I<-matrix(X[10,],60,70)
I1 <- apply(I, 1, rev)
image(t(I1),col=gray(0:255 / 255))

#***************************************#
#        Seperate train and test        #
#***************************************#
set.seed (1)
train_indexes = sample(1:n, size = round(2*n/3))
train_set.X = X.clean[train_indexes, ]
train_set.Y = as.matrix(y[train_indexes, ])

test_set.X  = X.clean[-train_indexes, ]
test_set.Y  = as.matrix(y[-train_indexes, ])

train_set = as.data.frame(cbind(train_set.Y, train_set.X))
test_set  = as.data.frame(cbind(test_set.Y, test_set.X))

train_set = data.frame(train_set.X, y= as.factor(train_set.Y))
test_set  = data.frame(test_set.X, y= as.factor(test_set.Y))

#*************************************#
#                KNN                  #
#*************************************#
model.kknn = train.kknn(y~., data= train_set, kmax = 30)
model.kknn.best.k = model.kknn$best.parameters$k

model.knn.pred = knn(train_set.X, test_set.X, train_set.Y ,k=model.kknn$best.parameters$k)

n_test = nrow(test_set.X)
#Vector with prediction of the class of each test observation
knn_pred = matrix(, nrow=n_test , ncol=1) 

for(i in 1:n_test)
{
    #Class of each neighbor of the observation i
    iline = train_set$y[attr(model.knn.pred, 'nn.index')[i,]]

    # Count votes for each class
    count1 = length(which(iline[] == 1)) 
    count2 = length(which(iline[] == 2)) 
    count3 = length(which(iline[] == 3)) 
    count4 = length(which(iline[] == 4)) 
    count5 = length(which(iline[] == 5)) 
    count6 = length(which(iline[] == 6)) 

    #Vector of votes
    votes       = c(count1, count2, count3, count4, count5, count6)

    #I need to select the class with the highest number of neighbors
    max_vote = which.max(votes)

    knn_pred[i] = max_vote
}

perfMeasure(knn_pred, test_set$y, TRUE)
# 26.3% error


#*******************#
#        LDA        #
#*******************#
model.lda = lda(y ~ ., data = train_set)
model.lda.predicted = predict(model.lda, newdata = test_set)
perfMeasure(model.lda.predicted$class, test_set$y, TRUE)
# 20.8% error

#*********************************#
#        FDA VISUALISATION        #
#*********************************#
U = model.lda$scaling
Z = train_set.X %*% U

dim1 = 1
dim2 = 2
plot(Z[train_set.Y==1,dim1],Z[train_set.Y==1,dim2], xlim=range(Z[,dim1]),ylim=range(Z[,dim2]), xlab = "1st FDA Vector", ylab = "2nd FDA Vector")
points(Z[train_set.Y==2,dim1],Z[train_set.Y==2,dim2],pch=2,col=2)
points(Z[train_set.Y==3,dim1],Z[train_set.Y==3,dim2],pch=3,col=3)
points(Z[train_set.Y==4,dim1],Z[train_set.Y==4,dim2],pch=4,col=4)
points(Z[train_set.Y==5,dim1],Z[train_set.Y==5,dim2],pch=5,col=5)
points(Z[train_set.Y==6,dim1],Z[train_set.Y==6,dim2],pch=6,col=6)


#*****************************#
#        BUILD FDA SET        #
#*****************************#
# This set will be used later for dimension reduction purposes

U = model.lda$scaling
train_set.fda.X = train_set.X %*% U
train_set.fda = data.frame(train_set.fda.X)
train_set.fda["y"] = factor(train_set.Y)

test_set.fda.X = test_set.X %*% U
test_set.fda = data.frame(test_set.fda.X)
test_set.fda["y"] = factor(test_set.Y)

#************************#
#        LDA + PCA       #
#************************#

pca = prcomp(train_set.X, center = TRUE, scale = TRUE )

fit_lda_pca = function() {
  model.lda.pca = lda(y ~ ., data = train_set.pca)
  model.lda.pca.predicted = predict(model.lda.pca, newdata = test_set.pca)
  p = perfMeasure(model.lda.pca.predicted$class, test_set.pca$y)
  return(p)
}

nb_comp = 30 # ARBITRARY NUMBER TO BE DETERMINED USING CV

train_set.pca.X = as.data.frame(pca$x[,1:nb_comp])
train_set.pca = data.frame(train_set.pca.X)
train_set.pca["y"] = train_set.Y

test_set.pca.X = predict(pca, newdata = test_set.X)[,1:nb_comp]
test_set.pca = data.frame(test_set.pca.X)
test_set.pca["y"] = test_set.Y

fit_lda_pca()

##########################
# Cross Validation 

nb_folds = 10
accs = matrix(Inf, 100, 1)
for (M in 1:100) {
  printf("Nb Components %d", M)
  a.train_set.pca.X = as.data.frame(pca$x[,1:M])
  a.train_set.pca = data.frame(a.train_set.pca.X)
  a.train_set.pca["y"] = train_set.Y
  
  a.test_set.pca.X = predict(pca, newdata = test_set.X)[,1:M]
  a.test_set.pca = data.frame(a.test_set.pca.X)
  a.test_set.pca["y"] = test_set.Y
  
  folds = createFolds(a.train_set.pca$y, k = nb_folds)
  
  acc = 0;
  for (k in 1:nb_folds) {
    validation_indexes = folds[[k]]
    train_set.pca.X = a.train_set.pca.X[-validation_indexes,]
    train_set.pca = a.train_set.pca[-validation_indexes,]
    
    test_set.pca.X = a.train_set.pca.X[validation_indexes,]
    test_set.pca = a.train_set.pca[validation_indexes,]
    
    acc = acc + fit_lda_pca()
  }
  
  acc = acc / nb_folds
  accs[M] = acc
}
min(accs)
which.min(accs)

plot(accs, xlab = "Number of Principal Components", ylab = "Error")
points(x = which.min(accs), y = min(accs), col = "red", pch = 16)
abline(h = min(accs), col="red")
abline(v = which.min(accs), col="red")

nb_comp = which.min(accs) 

train_set.pca.X = as.data.frame(pca$x[,1:nb_comp])
train_set.pca = data.frame(train_set.pca.X)
train_set.pca["y"] = train_set.Y

test_set.pca.X = predict(pca, newdata = test_set.X)[,1:nb_comp]
test_set.pca = data.frame(test_set.pca.X)
test_set.pca["y"] = test_set.Y

fit_lda_pca()
# About 20%

#********************************#
#        Neural Network          #
#********************************#
train_set$y = factor(train_set$y)
test_set$y = factor(test_set$y)

model.nnet = nnet(y ~ ., data=train_set, size=2, MaxNWts = 20000)
model.nnet.predicted = predict(model.nnet, test_set, type="class")
perfMeasure(model.nnet.predicted, test_set$y)

#**************************************#
#        Neural Network + FDA          #
#**************************************#
fit_nnet_fda = function(size, decay) {
  model.nnet = nnet(y ~ ., data=a.train_set.fda, size=size, decay = decay, MaxNWts = 20000, trace = FALSE)
  model.nnet.predicted = predict(model.nnet, a.test_set.fda, type="class")
  return(perfMeasure(model.nnet.predicted, a.test_set.fda$y, FALSE))
}

train_set.fda$y = factor(train_set.fda$y)
test_set.fda$y = factor(test_set.fda$y)

a.train_set.fda = train_set.fda
a.test_set.fda = test_set.fda
fit_nnet_fda(50, 0)

##########################
# Cross Validation 

nb_folds = 5
accs = matrix(Inf, 100, 1)
for (M in 1:100) {
  printf("Nb Neurons %d", M)
  folds = createFolds(train_set.fda$y, k = nb_folds)
  
  acc = 0;
  for (k in 1:nb_folds) {
    validation_indexes = folds[[k]]
    
    a.train_set.fda = train_set.fda[-validation_indexes,]
    a.test_set.fda = train_set.fda[validation_indexes,]
    
    acc = acc + fit_nnet_fda(M, 0)
  }
  
  acc = acc / nb_folds
  accs[M] = acc
}
min(accs)
which.min(accs)

plot(accs, xlab = "Number of Neurons", ylab = "Error")
points(x = which.min(accs), y = min(accs), col = "red", pch = 16)
abline(h = min(accs), col="red")
abline(v = which.min(accs), col="red")

a.train_set.fda = train_set.fda
a.test_set.fda = test_set.fda
fit_nnet_fda(which.min(accs), 0)
# more than 26% error

#*****************************************************#
#        Neural Network + FDA + Regularization        #
#*****************************************************#

##########################
# Cross Validation 

nb_folds = 10
accs = matrix(Inf, 10, 1)
for (M in 1:10) {
  printf("Lambda %d", M)
  folds = createFolds(train_set.fda$y, k = nb_folds)
  
  acc = 0;
  for (k in 1:nb_folds) {
    validation_indexes = folds[[k]]
    
    a.train_set.fda = train_set.fda[-validation_indexes,]
    a.test_set.fda = train_set.fda[validation_indexes,]
    
    acc = acc + fit_nnet_fda(50, M)
  }
  
  acc = acc / nb_folds
  accs[M] = acc
}
min(accs)
which.min(accs)

plot(accs, xlab = "Regularization Parameter", ylab = "Error")
points(x = which.min(accs), y = min(accs), col = "red", pch = 16)
abline(h = min(accs), col="red")
abline(v = which.min(accs), col="red")

a.train_set.fda = train_set.fda
a.test_set.fda = test_set.fda
fit_nnet_fda(50, which.min(accs))
# 24%

#*****************************#
#        Decision Tree        #
#*****************************#

model.tree = tree(as.factor(y) ~ ., train_set)
summary(model.tree)
plot(model.tree)
text(model.tree, pretty = 0)

model.tree.predicted = predict(model.tree, test_set, type="class")
perfMeasure(model.tree.predicted, test_set$y, TRUE)
# 45%

Size<-cv.tree(model.tree)$size
DEV<-rep(0,length(Size))
for(i in (1:10)){
  cv.credit=cv.tree(model.tree)
  DEV<-DEV+cv.credit$dev
}
DEV<-DEV/10
plot(cv.credit$size,DEV,type='b')

model.tree.pruned = prune.tree(model.tree,best=3)
model.tree.pruned.predicted = predict(model.tree.pruned, test_set, type="class")
perfMeasure(model.tree.pruned.predicted, test_set$y, TRUE)
# 60%

#**********************************#
#        Decision Tree + FDA       #
#**********************************#

model.tree = tree(as.factor(y) ~ ., train_set.fda)
summary(model.tree)
plot(model.tree)
text(model.tree, pretty = 0)

model.tree.predicted = predict(model.tree, test_set.fda, type="class")
perfMeasure(model.tree.predicted, test_set$y, TRUE)
# 26.4%

Size<-cv.tree(model.tree)$size
DEV<-rep(0,length(Size))
for(i in (1:10)){
  cv.credit=cv.tree(model.tree)
  DEV<-DEV+cv.credit$dev
}
DEV<-DEV/10
plot(cv.credit$size,DEV,type='b')

model.tree.pruned = prune.tree(model.tree,best=8)
model.tree.pruned.predicted = predict(model.tree.pruned, test_set.fda, type="class")
perfMeasure(model.tree.pruned.predicted, test_set$y)
# Same results

#****************************#
#        Random Forest       #
#****************************#
model.random_forest = randomForest(as.factor(y) ~ ., data=train_set)
model.random_forest.predicted = predict(model.random_forest,newdata=test_set,type='response')
perfMeasure(model.random_forest.predicted, test_set$y, TRUE)
# 25%

#**********************************#
#        Random Forest + FDA       #
#**********************************#
model.random_forest = randomForest(as.factor(y) ~ ., data=train_set.fda)
model.random_forest.predicted = predict(model.random_forest,newdata=test_set.fda,type='response')
perfMeasure(model.random_forest.predicted, test_set$y, TRUE)
# 16.7%

#********************************#
#             SVM                #
#********************************#

# In thus part we discuss another approach of classification : The support vector machine with the kernel option.
# We will use a support vector approach to predict the facial expression using face features expressions


#The e1071 library includes the tune() function, to perform cross-validation (by default, it performs 10-fold cross-validation) to 
#determine the best tuning parameter
# The argument scale=TRUE tells the svm() function to scale each feature to have mean zero or standard deviation one; we thought it would be more...
# This dataset has a very large number of features compared to observations. This means that likely we could use a linear kernel. 
# in Other words we want to compare SVMs with a linear kernel,
model.tune = tune(svm, y~., data= train_set, kernel= "linear", ranges= list(c(0.01, 0.1, 1, 10), gamma= c(0.1, 1, 10)))
summary(model.tune)
#We see that cost=0.1 results in the lowest cross-validation error rate. 
#The tune() function stores the best model obtained, which can be accessed as follows: model.tune$best.parameters
#We can now perform the svm with those best tuning parameters
model.svm = svm(y~., data= train_set, kernel= 'linear', gamma= model.tune$best.parameters$gamma, cost= model.tune$best.parameters$Var1)

# 6 classes
summary(model.svm) 
#This tells us, for instance, that a linear kernel was used with cost=... , 
#and that there were ... support vectors, .. in class 1 and ... in class 2, ....

# We can now plot the support vector classifier obtained:
plot(x= model.svm, data= train_set, formula= X1~X2)
# We(Note that here the second feature is plotted on the x-axis and the first feature is plotted on the y-axis, 
#The region that is assigned to each class is shown on the right (light blue for 1, ...). The decision boundary between the two classes is linear 
#(because we used the argument kernel="linear"). 
#We see that in this case only 6 observations are misclassified. 


# Training errors = we only made 6 errors on the training data
confusion_matrix_svm.train = table(model.svm$fitted, train_set$y)
errors_svm.train = sum(model.svm$fitted != train_set$y)/length(train_set$y)
perfMeasure(model.svm$fitted, test_set$y, TRUE)


#*************************************#
#             ONE Vs ONE              #
#*************************************#

#However we are most interested not in the support vector classifier’s performance on the training observations, 
#but rather its performance on the test observations.
# If the response has more than two levels, then the svm() function will perform multi-class classification using the one-versus-one approach
yhat= predict(model.svm, newdata= test_set)
confusion_matrix_svm.test= table(yhat, test_set$y)
errors_svm.test = sum(yhat!=test_set$y)/length(test_set$y)
plot(x= model.svm, data= test_set, formula= X1~X2)
#==> The test Error is higher than the train_error which is normal


# ROC plot
pred = prediction(yhat, svm.test$y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
#or confusionMatrix(yhat, svm.test$y)

#*************************************************#
#            WHAT ABOUT OTHER KERNELS             #
#*************************************************#

# WHAT ABOUT RADIAL ? BAD IDEA!!!! 
model.tune.out = tune(svm, y~., data= svm.train, kernel= "radial", ranges= list(c(0.01, 0.1, 1, 10, 100), gamma= c(0.1, 1, 10)))
model.svm      = svm(y~., data=svm.train, kernel= 'radial', gamma= model.tune.out$best.parameters$gamma , cost= model.tune.out$best.parameters$Var1)

plot(x= model.svm, data= svm.train, formula= X1~X2)
table(model.svm$fitted, svm.train$y)
yhat = predict(model.svm, newdata=svm.test)
table(predict= yhat, truth= svm.test$y)

#==> CONFIRMS THAT WE SHOULD USE LINEAR!!! 





