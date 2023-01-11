#SUPERVISED LEARNING

# load the data
load(file = "class_data.RData")
#converting dependent variable to a categorical format since it's two classes: 0 & 1
y<-as.factor(y)
set.seed(22)

#Storing scaled version of x in a separate Data frame for future use (Ridge, Lasso, PCA, KNN)
scl_x = scale(x)

#checking for number of missing values in observations
sum(is.na(x))
print("There are no missing values in the data provided")

#Checking for imbalance in class
zeros<-0
ones<-0

for(i in y){
  if (i==1) {ones=ones +1 }
  else {zeros = zeros +1}
}
print(zeros/nrow(x)*100)
print(ones/nrow(x)*100)
print("Thus, data is well balanced")

#Checking for features with zero variance across all data
library(caret)
names(x)[nearZeroVar(x)]

##RANDOM FOREST

#Importing necessary variables
library(randomForest)
library(caret)
library(ROCR)
library(Boruta)
library(pROC)
library(ROCit)

# Number of folds for K folds cross validation
K=5 

#Initializing Outer fold Ids
fold =rep(0,dim(x)[1])

for(i in seq(1,length(fold),by=5)){
  fold[i]<-1
  fold[i+1] <- fold[i] + 1
  fold[i+2] <- fold[i] + 2
  fold[i+3] <- fold[i] + 3
  fold[i+4] <- fold[i] + 4
}

#Initializing inner fold Ids
ifold =rep(0,320)

for(i in seq(1,length(ifold),by=4)){
  ifold[i]<-1
  ifold[i+1] <- ifold[i] + 1
  ifold[i+2] <- ifold[i] + 2
  ifold[i+3] <- ifold[i] + 3
}

#Initializing variable to store outer fold classification error
outer_fold_error = rep(0,K)
rf_auc= rep(0,K)

#Initializing variable to store best mtry value for each fold
bestmtry =rep(0,K)
final_ntree = rep(0,K)

for (i in 1:K){
  
  #Splitting data into training and validation sets
  train__ <- x[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test__ <- x[Outer_folds==i,]
  test_y <- y[Outer_folds==i]
  
  #Utilizing Boruta algorithm on training data for feature selection
  boruta_output <- Boruta(train__,train_y)
  
  #Print significant variables including tentative
  boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
  
  #Only using selected features for hyperparameter tuning and model fitting
  train_x<-train__[,boruta_signif]
  test_x<-test__[,boruta_signif]
  
  #Tuning function for finding best mtry value
  gg<- tuneRF(train_x,train_y,stepFactor = 1.5, improve = 0.01, plot=FALSE,doBest = TRUE)
  
  #Storing mtry value corresponding to lowest OOB error estimate
  bestmtry[i] <- gg$mtry
  
  #Shuffling fold Ids to eliminate any bias in diving data into folds
  inner_folds=sample(ifold)
  
  k=4
  
  #Initializing variable to store inner fold classification error
  rf.fold.error = matrix(0,nrow=k,ncol=10)
 
  #Inner K fold cross validation for finding best ntree value 
  for(j in 1:k){
    #Dividing training set further into training and validation set
    train2x <-train_x[inner_folds!=j,]
    train2y <- train_y[inner_folds!=j]
  
    val_x <- train_x[inner_folds==j,]
    val_y <- train_y[inner_folds==j]

    #Iterating no of trees from 100 to 1000
    for(t in seq(100,1000,by=100)){
    
      #Fitting random forest model    
      fit=randomForest(x=train2x,y=train2y,mtry=bestmtry[i],ntree = t, importance=TRUE)
  
      #making and storing predictions for validation set
      pred=predict(fit,val_x)
  
      #computing and storing fold error
      rf.fold.error[j,t/100] <- mean(pred!=val_y)
    }
  }
  #Storing ntree value corresponding to lowest cv error
  final_ntree_ind <-which(rf.fold.error ==min(rf.fold.error),arr.ind = TRUE)
  final_ntree[i] <-final_ntree_ind[1,2]*100
  
  #Refitting model with best ntree and mtry values
  fit_new_rf = randomForest(x=train_x,y=train_y,mtry=bestmtry[i],ntree =final_ntree[i], importance=TRUE)
  
  #Making predictions with optimized model and calcuting cv error and auc score
  new_pred = predict(fit_new_rf,test_x)
  outer_fold_error[i] <-mean(new_pred != test_y)
  rf_auc[i] <-auc(test_y, as.numeric(new_pred))
  
  #Plotting ROC curve
  ROCit_obj <- rocit(score=as.numeric(new_pred),class=test_y)
  plot(ROCit_obj)
}
#Taking a mean of validation error across K folds
rf_test_error <- mean(outer_fold_error)
rf_test_error

varImpPlot(fit_new_rf)

#Checking auc score of random forest model
rf_auc_mean <- mean(rf_auc)
rf_auc_mean


## SVM
library(e1071)
library(readxl)

#Initializing variables to store fold errors and best parameter values  
svm.fold.error = rep(0,K)
svm_auc =rep(0,K)
best.cost = rep(0,K)
best.gamma = rep(0,K)

#K fold cross validation loop
for(i in 1:K){
    
  #Splitting data into training and validation sets
  train__ <- x[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test__ <- x[Outer_folds==i,]
  test_y <- y[Outer_folds==i]
  
  #Utilizing Boruta algorithm on training data() for feature selection
  boruta_output <- Boruta(train__,train_y)
  
  #Print significant variables including tentative
  boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
  
  #Only using selected features for hyperparameter tuning and model fitting & scaling them
  train_x<-scale(train__[,boruta_signif])
  test_x<-scale(test__[,boruta_signif])
  
  #Using tuning function from library e1071 with inbuilt 10 fold cross validation  
  svm_fit<-tune.svm(x=train_x, y=train_y, kernel='radial',gamma=c(0.01,0.1,1,10,20),cost=c(0.01,0.1,1,10,100))
  
  #Storing best gamma and cost values obtained after tuning
  best.cost[i] <- svm_fit$best.parameters$cost
  best.gamma[i] <- svm_fit$best.parameters$gamma

  combine = data.frame(train_x,train_y)
  
  #Fitting svm model on training data with tuned parameters
  model = svm(train_y ~. , kernel = 'radial', cost =svm_fit$best.parameters$cost, gamma = svm_fit$best.parameters$gamma, data = combine)
  
  #making and storing predictions for validation set
  preds = predict(model,test_x)
  
  #Computing and storing validation error and auc score for each fold
  svm.fold.error[i] <- mean(preds!=test_y)
  svm_auc[i] <-auc(test_y, as.numeric(preds))
  
  #Plotting ROC curve for each fold
  ROCit_obj <- rocit(score=as.numeric(preds),class=test_y)
  plot(ROCit_obj)
}
best.cost
best.gamma

#Taking a mean of cv error across K folds
svm_cv_error <-mean(svm.fold.error)
svm_cv_error

#Checking mean auc score of svm model
svm_auc_mean <-mean(svm_auc)
svm_auc_mean

##KNN with original data
library(class)
knn_error = data.frame()
knn_auc = data.frame()
K=5

#K fold cross validation loop 
for(i in 1:K){
  
  #Splitting data into training and validation sets
  train__ <- scl_x[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test__ <- scl_x[Outer_folds==i,]
  test_y <- y[Outer_folds==i]
  
  #Utilizing Boruta algorithm on training data() for feature selection
  boruta_output <- Boruta(train__,train_y)
  
  #Print significant variables including tentative
  boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
  
  #Only using selected features for knn
  train_x<-train__[,boruta_signif]
  test_x<-test__[,boruta_signif]
  
  #Iterating through k nearest neighbours for each fold
  neighbours=dim(train_x)[1]
  for(j in 1:neighbours){
    knnmodel = knn(train=train_x, cl=train_y ,test=test_x, k=j, prob =TRUE)
    knn_error[i,j]=mean(knnmodel!=test_y)
    knn_auc[i,j] <- roc(test_y,attributes(knnmodel)$prob)$auc
  }
}
#storing mean cv error and auc score for each value of k
mean_fold_error = apply(knn_error,2,mean)
knn_which.neighbour.min=which.min(mean_fold_error)

mean_fold_auc = apply(knn_auc,2,mean)
knn_which.neighbour.max=which.max(mean_fold_auc)

#Printing lowest cv error and highest auc score where no of neighbours is equal to the variable number displayed in name
print(mean_fold_error[knn_which.neighbour.min])
print(mean_fold_auc[knn_which.neighbour.max])


##KNN with PCA scores
knn_pca_auc = data.frame()
knn_error_pca = data.frame()
pca.out=prcomp(x, scale=TRUE)
x_pca=pca.out$x

#Calculating variance of each principal component
pr.var=pca.out$sdev^2
pve=pr.var/sum(pr.var)

plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')

plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
abline(h=0.95,col='red',lty=2)

#Storing loaded scores of principal components and using as data for knn
pca_use=(as.matrix(pca.out$x))[,1:280]

K=5

for(i in 1:K){
  
  #Splitting data into training and validation set
  train_x <- pca_use[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test_x <- pca_use[Outer_folds==i,]
  test_y <- y[Outer_folds==i]
  
  neighbours=dim(train_x)[1]
  for(j in 1:neighbours){
    knnmodel_pca = knn(train=train_x, cl=train_y ,test=test_x, k=j, prob=TRUE)
    knn_error_pca[i,j]=mean(knnmodel_pca!=test_y)
    knn_pca_auc[i,j] <- roc(test_y,attributes(knnmodel_pca)$prob)$auc
  }
}
#storing mean validation error and auc score for each value of k
mean_fold_error_pca = apply(knn_error_pca,2,mean)
mean_fold_pca_auc = apply(knn_pca_auc,2,mean)

#Printing lowest validation error where no of neighbours is equal to the variable number displayed in name
knn_pca_which.neighbour.min=which.min(mean_fold_error_pca)
knn_pca_which.neighbour.max=which.max(mean_fold_pca_auc)


##LOGISTIC REGRESSION
library(caret)
library(mlbench)
library(glmnet)

correlationMatrix <- cor(x)
#find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)

#Removing highly correlated variables from x
lr_x <- x[,-c(highlyCorrelated)]

fold_error_logistic=rep(0,K)
log_auc = rep(0,K)

for(j in 1:K){
  
  #Splitting into training and validation set
  train_x <- lr_x[Outer_folds!=j,]
  train_y <- y[Outer_folds!=j]
  
  test_x <- lr_x[Outer_folds==j,]
  test_y <- y[Outer_folds==j]
  
  #finding statistically significant variables having p-values less than 0.01
  d=1
  imp_index= c()
  for(i in 1:length(train_x)){
    splitx=train_x[i]
    names(splitx)="X"
    glm.fit = glm(train_y~X,data=splitx,family=binomial)
    if(summary(glm.fit)$coefficients[2,4]<0.01){
      imp_index[d] = i
      d=d+1
    }}
  
  train_x <- train_x[,imp_index]
  test_x <- test_x[,imp_index]
  
  train=data.frame(train_x,train_y)
  test=data.frame(test_x,test_y)
  
  #Fitting logistic regression model to training set
  glm.fit=glm(train_y~.,data=train,family=binomial)

  #Making predictions for validation set and computing cv fold error
  glm.probs=predict(glm.fit,newdata=test,type='response')
  glm.pred=ifelse(glm.probs>0.5,1,0)
  fold_error_logistic[j] =mean(glm.pred!=test_y)
  log_auc[j]<- auc(test_y,as.numeric(glm.pred))
}

mean_cv_error_logistic = mean(fold_error_logistic)
mean_cv_error_logistic

auc_logistic_mean =mean(log_auc)
auc_logistic_mean

# LASSO REGRESSION
library(Rcpp)
K=5
#Initializing variable to store best lamda value
best_lam = rep(0,K)
lasso.fold.error = rep(0,K)
lasso_auc =rep(0,K)

Outer_folds=sample(fold)

#scaling the observations
scl_x<-scale(lr_x)

for(i in 1:K){
  
  #Using scaled observation values for splitting into training and validation set
  train_x <- scl_x[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test_x <- scl_x[Outer_folds==i,]
  test_y <- y[Outer_folds==i]
  
  #Using in-built cross validation function for lasso regression to find best lamda
  cv.lasso <- cv.glmnet(as.matrix(train_x), as.matrix(train_y),nfolds=5, family="binomial") #alpha=1 for lasso
  # plot cv.error v.s a sequence of lambda values; the two dashed lines correspond to the best lambda value and the largest value of lambda such that error is within 1 standard error of the best lambda
  plot(cv.lasso)
  
  #Using lamda corresponding to min cv error
  best_lam[i] <- cv.lasso$lambda.min
  
  #c<-coef(cv.lasso,s=best_lam[i],exact=TRUE)
  #inds<-which(c!=0)
  #variables<-row.names(c)[inds]
  #variables <-variables[-1]
  #test_x <-test_x[,c(variables)]
  
  #Fitting logistic regression model corresponding to best lamda value chosen above
  fitnew.lasso=glmnet(train_x,train_y,lambda=5,alpha=1,family="binomial")
  predictions_test=predict(fitnew.lasso,s=best_lam[i],newx=as.matrix(test_x), type ='class')
  
  glm.pred=ifelse(predictions_test>0.5,1,0)
  lasso.fold.error[i] = mean(glm.pred != test_y)
  lasso_auc[i]<- auc(test_y,as.numeric(glm.pred))
}
lasso_cv_error <-mean(lasso.fold.error)
lasso_cv_error

lasso_auc_mean <-mean(lasso_auc)
lasso_auc_mean

# RIDGE REGRESSION

#Initializing variable to store best lamda value
best_lam = rep(0,K)
ridge.fold.error = rep(0,K)
ridge_auc = rep(0,K)
Outer_fold = sample(fold)

for(i in 1:K){
  
  #Using scaled observation values for splitting into training and validation set
  train_x <- scl_x[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test_x <- scl_x[Outer_folds==i,]
  test_y <- y[Outer_folds==i]
  
  #Using in-built cross validation function for ridge regression to find best lamda
  cv.ridge <- cv.glmnet(as.matrix(train_x), as.matrix(train_y), family='binomial',nfolds=5,alpha=0) #alpha=0 for ridge regression
  # plot cv.error v.s a sequence of lambda values; the two dashed lines correspond to the best lambda value and the largest value of lambda such that error is within 1 standard error of the best lambda
  plot(cv.ridge)
  
  #Using lamda corresponding to 1se of best lamda(more conservative with fewer variables)
  best_lam[i] <- cv.ridge$lambda.1se
  
  #Fitting logistic regression model corresponding to best lamda value chosen above
  fitnew.lasso=glmnet(x=as.matrix(train_x),y=as.matrix(train_y),lambda=best_lam[i],alpha=0,family="binomial")
  predictions_test=predict(fitnew.lasso,s=best_lam[i],newx=as.matrix(test_x), type ='class')
  
  glm.pred=ifelse(predictions_test>0.5,1,0)
  ridge.fold.error[i] = mean(glm.pred != test_y)
  ridge_auc[i]<- auc(test_y,as.numeric(predictions_test))
}
ridge_cv_error <-mean(ridge.fold.error)
ridge_cv_error

ridge_auc_mean <-mean(ridge_auc)
ridge_auc_mean

##LDA
K=5
library(MASS)
fold_error_lda=rep(0,K)
fold_auc_lda = rep(0,K)

for(j in 1:K){
  
  #Splitting into training and validation set
  train_x <- lr_x[Outer_folds!=j,]
  train_y <- y[Outer_folds!=j]
  
  test_x <- lr_x[Outer_folds==j,]
  test_y <- y[Outer_folds==j]
  
  #finding statistically significant variables having p-values less than 0.01
  d=1
  imp_index= c()
  for(i in 1:length(train_x)){
    splitx=train_x[i]
    names(splitx)="X"
    glm.fit = glm(train_y~X,data=splitx,family=binomial)
    if(summary(glm.fit)$coefficients[2,4]<0.01){
      imp_index[d] = i
      d=d+1
    }}
  
  train_x <- train_x[,imp_index]
  test_x <- test_x[,imp_index]

  train=data.frame(train_x,train_y)
  test=data.frame(test_x,test_y)
  #Fitting LDA model to data
  fitnew.lda=lda(train_y~.,data=train)
  
  #Making predictions for validation set and computing cv fold error
  lda_predictions_test=predict(fitnew.lda,newdata=test)
  #predictions_train=predict(fitnew.lda,newx=as.matrix(testx))

  fold_error_lda[j]=mean(lda_predictions_test$class!=test_y)
  fold_auc_lda[j] <- auc(test_y,as.numeric(lda_predictions_test$class))
}

mean_cv_error_lda = mean(fold_error_lda)
mean_cv_error_lda

lda_auc_mean <-mean(fold_auc_lda)
lda_auc_mean

##QDA
fold_error_qda=rep(0,K)
fold_auc_qda = rep(0,K)

for(j in 1:K){
  
  #Splitting into training and validation set
  train__ <- lr_x[Outer_folds!=j,]
  train_y <- y[Outer_folds!=j]
  
  test_x <- lr_x[Outer_folds==j,]
  test_y <- y[Outer_folds==j]
  
  #Selecting only top 50 variables with highest correlation with dependent variable
  cor(train_x,as.numeric(train_y))
  indices=order(abs(cor(train__,as.numeric(train_y))),decreasing =TRUE)[1:50]
  train_x <- train__[,indices]
  
  train=data.frame(train_x,train_y)
  test=data.frame(test_x,test_y)
  
  #Fitting QDA model to data
  fitnew.qda=qda(train_y~.,data=train)
  
  #Making predictions for validation set and computing cv fold error
  qda_predictions_test=predict(fitnew.qda,newdata=test)
  fold_error_qda[j]=mean(qda_predictions_test$class!=test_y)
  fold_auc_qda[j] <- auc(test_y,as.numeric(qda_predictions_test$class))
  
}

mean_cv_error_qda = mean(fold_error_qda)
mean_cv_error_qda

qda_auc_mean <-mean(fold_auc_qda)
qda_auc_mean


#DECISION TREES
library(tree)
library(rpart)

#Initializing variables to store classification error values and auc scores
K=5
fold_error_tree = rep(0,K)
fold_error_preprun = rep(0,K)
dt_auc = rep(0,K)
dt_auc_simple = rep(0,K)


for (i in 1:K){
  
  #Splitting data into training and validation sets
  train_x <- x[Outer_folds!=i,]
  train_y <- y[Outer_folds!=i]
  
  test_x <- x[Outer_folds==i,]
  test_y <- y[Outer_folds==i]

  red_data_train <- cbind(train_x,train_y)
  red_data_test <- cbind(test_x, test_y)

  fit <- rpart(train_y ~., data = red_data_train, method = 'class')
  #rpart.plot(fit, extra = 106)
  tree.pred2 <-predict(fit, red_data_test, type = 'class')
  fold_error_tree[i] =mean(tree.pred2!=test_y)
  dt_auc_simple[i] <-auc(test_y, as.numeric(tree.pred2))

  #PRE PRUNING
  model_preprun1 <- rpart(train_y ~ ., data = red_data_train, method = "class", control = rpart.control(cp = 0.01, minsplit = 2))
  xerror <-model_preprun1$cptable[,4]
  cp_ind <-which(xerror == min(xerror),arr.ind = TRUE)
  
  #Storing cp related to lowest xerror
  lowest_cp <-model_preprun1$cptable[,1][cp_ind]
  plotcp(model_preprun1, minline=TRUE) #Lowest Cp= 0.15
  
  #Building new tree with selected Cp
  model_preprun2 <- rpart(train_y ~ ., data = red_data_train, method = "class", control = rpart.control(cp = lowest_cp, minsplit = 2))
  tree.pred5 <-predict(model_preprun2, red_data_test, type = 'class')
  fold_error_preprun[i] <- mean(tree.pred5!=test_y)
  
  dt_auc[i] <-auc(test_y, as.numeric(tree.pred5))
}
simple_tree_error <- mean(fold_error_tree)
simple_tree_error

dt_auc_mean_simple = mean(dt_auc_simple)
dt_auc_mean_simple

preprun_error <-mean(fold_error_preprun)
preprun_error

dt_auc_mean = mean(dt_auc)
dt_auc_mean

##################################################################

#UNSUPERVISED LEARNING BELOW

##################################################################
library(cluster)
library(stats)
library(dplyr)
library(ggplot2)
library(ggfortify)
library(factoextra)
library(hopkins)
library(clValid)
library(fpc)
install.packages("fpc")
##################################################################
##################################################################
##################################################################
#importing the raw data
load(file = "cluster_data.RData")
raw_data=as.matrix(y)
#we are nomalizing the data
means=apply(raw_data,2,mean)
sds=apply(raw_data,2,sd)
means
sort(sds,decreasing=T)
sds
max(sds)
which.max(sds)#col.no=380
#next highest col.no=381
#as the std dev and means vary a lot across columns, it is imperative that we scale the data
data.nor=scale(raw_data,scale=sds,center = means)
data.nor
#calculate the distance
distance=dist(data.nor)
distance[1:1000]
#########################################################
##applying pca###########################################
y_pca<-prcomp(data.nor, center=TRUE, scale=TRUE)
summary(y_pca)
#plotting for pca
fviz_eig(y_pca)
#variable plot for pca
fviz_pca_var(y_pca, col.var="black")
#establishing no of pc using kaiser criterion
eig.val<-get_eigenvalue(y_pca)
eig.val
#there are 152 variables with number of eigenvalues which are higher than 1
pca_use<-prcomp(y, center=FALSE, scale=TRUE, rank= 152) 
#final pca to be used
pca_f=as.matrix(pca_use$x)
##################################################################
#kmeans clustering################################################

#finding the optimum value of 'K' for analysis using elbow method   
fviz_nbclust(pca_f, kmeans, method = "wss", k.max = 20)+labs(subtitle ="Elbow Method")
#elbow method suggests us to use k=11
#finding the optimum value of 'K' for analysis using silhouette method 
fviz_nbclust(pca_f, kmeans, method='silhouette')
#silhouette method suggests us to use k=2
#finding the optimum value of 'K' for analysis using gap statistics
fviz_nbclust(pca_f, kmeans, method='gap_stat')
#gap method suggests us to use k=6

#first we choose k=11 as optimal no of clusters and create a cluster plot
k_means11=kmeans(pca_f,11,iter.max=20,nstart=20)
k_means11
table(k_means11$cluster)
#next we choose k=2 as optimal no of clusters and create a cluster plot
k_means9=kmeans(pca_f,2,iter.max=20,nstart=20)
k_means9
table(k_means9$cluster)
#next we choose k=6 as optimal no of clusters and create a cluster plot
k_means6=kmeans(pca_f,6,iter.max=20,nstart=20)
k_means6
table(k_means6$cluster)

#we are choosing k=6 as no of clusters
#plotting the result
autoplot(k_means6,pca_f,frame=T)

#a 2 dim cluster will not give a better picture
#using dunn index to check the quality of cut
dunn(dist(pca_f),k_means6$cluster)
#one of the reasons we have such poor performance in k-means clustering as it is susceptible to outliers

plot(pca_f, pch=16, col="blue")
#indeed there are a lot of outliers
##################################################################
#hierarchical clustering################################################

#finding the optimum value of 'K' for analysis using elbow method   
fviz_nbclust(pca_f,hcut , method = "wss", k.max = 20)+labs(subtitle ="Elbow Method")
#difficult to discern using elbow method
#finding the optimum value of 'K' for analysis using silhouette method 
fviz_nbclust(pca_f,hcut, method='silhouette')
#silhouette method suggests us to use k=8
#finding the optimum value of 'K' for analysis using gap statistics
fviz_nbclust(pca_f, kmeans, method='gap_stat')
#gap method suggests us to use k=10

#now we proceed to do hierarchical clustering

hc_comp=hclust(dist(pca_f),method="complete")
hc_s=hclust(dist(pca_f),method="single")
plot(hc_comp)
plot(hc_s)
#we are choosing complete linkage as the clusters are more pronounced
#Now we cut the tree at K=10 determined from gap_stat method
hc_10=cutree(hc_comp,10)
table(hc_10)
hc_8=cutree(hc_comp,8)
table(hc_8)
#plot to visualize 
plot(hc_comp, col = hc_10)

#we have a total of 976 observations in cluster 1 only
#this is a problem
#let us confirm this using dunn index
#using dunn index to check the quality of cut, more is better
dunn(dist(pca_f),hc_10,method = "euclidean")#0.31

##################################################################
#PAM - Partitioning Around Medoids###############################
#this method is less sensitive to outliers and is based on k-means clustering

fviz_nbclust(pca_f, FUNcluster=cluster::pam,method='wss', k.max = 20)#k=3
fviz_nbclust(pca_f, FUNcluster=cluster::pam,method='silhouette', k.max = 20)#k=2
fviz_nbclust(pca_f, FUNcluster=cluster::pam, method="gap_stat", k.max  = 20)+ theme_classic()#k=16

pam_3=pam(pca_f,k=3)
summary(pam_3)
pam_2=pam(pca_f,k=2)
summary(pam_2)
pam_16=pam(pca_f,k=16)
summary(pam_16)

#based on average silhouette width 0.064, we are choosing k=3 

#dunn index
dunn(dist(pca_f),pam_2$cluster)

pam3<-eclust(pca_f, "pam", k=3)
fviz_silhouette(pam3)

##################################################################
#DB Scan###############################
#again, this method is less sensitive to outliers

#we take a random value of eps and min points initially
d_scan=dbscan(pca_f,eps=20,2)
d_scan
#there is a lot of noise
#now we choose optimal eps calue 
dbscan::kNNdistplot(pca_f,k=2)
abline(h=30,lty=2,col="red")
#optimal eps value is around a distance of 30
d_scan=dbscan(pca_f,eps=30,2)
help(dbscan)
d_scan
#removing the noise from data
table(d_scan$cluster)
index=d_scan$cluster
data_refined=pca_f[index==1,]

fviz_nbclust(data_refined, kmeans,method='wss', k.max = 20)#k=15
fviz_nbclust(data_refined, kmeans,method='silhouette', k.max = 20)#k=4
fviz_nbclust(data_refined, kmeans, method="gap_stat", k.max  = 20)+ theme_classic()#k=8

#first we choose k=15 as optimal no of clusters and create a cluster plot
nk_means15=kmeans(data_refined,15,iter.max=20,nstart=20)
nk_means15
table(nk_means15$cluster)
#next we choose k=4 as optimal no of clusters and create a cluster plot
nk_means4=kmeans(data_refined,4,iter.max=20,nstart=20)
nk_means4
table(nk_means4$cluster)
#next we choose k=8 as optimal no of clusters and create a cluster plot
nk_means8=kmeans(data_refined,8,iter.max=20,nstart=20)
nk_means8
table(nk_means8$cluster)

#dunn index
dunn(dist(data_refined),nk_means15$cluster)

k15<-eclust(data_refined, "kmeans", k=15)
fviz_silhouette(k15)


