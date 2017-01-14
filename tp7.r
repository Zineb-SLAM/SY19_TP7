#***************************************#
#              Libraries                #
#***************************************#

library('nnet')
library('MASS')
library('e1071')
library('pls')
library('caret')
library('ROCR')
library('FNN')
library("kknn")


#***************************************#
#              Load Data                #
#***************************************#

data = load('data_expressions.RData')
X.clean = X[, colSums(X)!=0]
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
train = sample(1:n, round(2*n/3))
train.X = X.clean[train, ]
train.Y = as.matrix(y[train, ])

test.X  = X.clean[-train, ]
test.Y  = as.matrix(y[-train, ])

train_set = as.data.frame(cbind(train.Y, train.X))
test_set  = as.data.frame(cbind(test.Y, test.X))



#*************************************#
#                KNN                  #
#*************************************#
train_set      = data.frame(train.X, y= as.factor(train.Y))
test_set       = data.frame(test.X, y= as.factor(test.Y))

model.kknn = train.kknn(y~., data= train_set, kmax = 30)
model.kknn.best.k = model.kknn$best.parameters$k


model.knn.pred= knn(train.X, test.X, train.Y ,k=model.kknn$best.parameters$k)

n_test = nrow(test.X)
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

confusion_matrix_knn = table(predict=knn_pred , truth= test_set$y)
errors_knn = sum(knn_pred!=test_set$y)/length(test_set$y)
# 26.3% errors

#*************************#
#        PCA & FDA        #
#*************************#
# We can use the PCA on all the data since the class responses do not intervene

n_folds = 5
compnt_max = 120
n_train = nrow(train_set)
Errors = rep(0,compnt_max)
folds_i = sample(rep(1:n_folds, length.out = n_train))

# We can use the PCA on all the data since the class responses do not intervene
model.pca      = prcomp(train.X, center = TRUE, scale. = TRUE)

for(c in 2:compnt_max)
{
	sum = 0
    X.pca          = model.pca$x[, 1:c]

    for(k in 1:n_folds)
    {
    	
        test_i         = which(folds_i == k)

        cv.train.X     = X.pca[-test_i,]
        cv.train.Y     = as.matrix(train.Y[-test_i,])
        cv.train_set   = as.data.frame(cbind(cv.train.Y, cv.train.X))  

        validation.X   = X.pca[test_i,]
        validation.Y   = as.matrix(train.Y[test_i,])
        validation_set = as.data.frame(cbind(validation.Y, validation.X))           

        model.pca.lda  = lda(V1~., data= cv.train_set)      
        model.pca.lda.pred = predict(model.pca.lda, newdata= validation_set)

        # confusion Matrix    
        M                  = table(validation.Y, as.matrix(model.pca.lda.pred$class))
        sum                = sum + 1 - sum(diag(M))/dim(validation_set) #those that are not in my class  
        sum

    }
    Errors[c] = sum
}

which.min(Errors)


#***************************************#
#        Dimesion Reduced Data          #
#***************************************#

# We can use the PCA on all the data since the class responses do not intervene
model.pca  = prcomp(train.X, center = TRUE, scale = TRUE)

X.dimreduced = model.pca.train$x[, 1:37]
train.X = X.dimreduced
train.Y = as.matrix(y[train, ])

#We should project the principal component of the train  on the test feature ... with model.pca$rotation ??


#********************************#
#        Neural Network          #
#********************************#

model.nn1 = nnet(y~., data = train_set, size=2, linout = TRUE, decay=0, MaxNWts = 10000)
model.nn1.pred =  predict(nn1, newdata= test_set)
model.nn1.pred

model.nn2 = nnet(V1~., data = train_set, size=10, linout = TRUE, decay=1, MaxNWts = 10000)
model.nn2.pred =  predict(nn1, newdata= test_set)
model.nn2.pred


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





