library(mice)
library(VIM)
library(corrplot)

setwd("C:/Users/Ganesh/Desktop/My Files/ganesh_nielsen/Analytics_Vidhya/Loan_Prediction")

train<-read.csv("train.csv")
test<-read.csv("test.csv")
test$Loan_Status="test"

Combined<-rbind(train,test)

##############################NA imputation#############################################
# str(Combined)
# pMiss <- function(x){sum(is.na(x))/length(x)*100}
# apply(Combined,2,pMiss)
# md.pattern(Combined)
# aggr_plot <- aggr(Combined, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(Combined), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
# Combined_imputed <- mice(Combined,m=1,maxit=50,seed=500)

Combined_imputed <- preProcess(Combined, method = c("medianImpute","center","scale"))
Combined_imputed<-predict(Combined_imputed, Combined)

# stripplot(as.matrix(Combined), pch = 20, cex = 1.2)
###########################################################################
#one hot encoding

library(dummies) 
Combined_final <- dummy.data.frame(Combined_imputed, names=c("Gender","Married","Dependents","Education","Self_Employed","Property_Area"), sep="_")
Combined_final<- subset(Combined_final, select=-c(Gender_,Married_,Dependents_,`Education_Not Graduate`,Self_Employed_,Property_Area_Rural,Gender,Married_Yes,Self_Employed_No))

test_final<-as.data.table(Combined_final)[Loan_Status =="test",]
test_final$Loan_Status=NULL
train_final<-as.data.table(Combined_final)[Loan_Status !="test",]
# train_final$Loan_Status = ifelse(train_final$Loan_Status=="N", 0, 1)
train_final$Loan_Status= factor(train_final$Loan_Status)

train_final=as.data.frame(train_final)
test_final=as.data.frame(test_final)
###########################################################################################
#Check correlation
corMatrix <- cor(train_final[1:nrow(train_final),][sapply(train_final[1:nrow(train_final),], is.numeric)])
corrplot(corMatrix, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)             
##########################################################################################################


predictors=colnames(subset(train_final,select=-c(Loan_ID,Loan_Status)))
outcomeName="Loan_Status"

set.seed(200)
#K Fold cross validation
fitControl <- trainControl(method = "cv",number = 5,savePredictions = 'final',classProbs = T)

#LOGISTIC REGRESSION
model_lr<-train(train_final[,predictors],train_final[,outcomeName],method='glm',trControl=fitControl,tuneLength=6)
#RANDOM FOREST
model_rf<-train(train_final[,predictors],train_final[,outcomeName],method='rf',trControl=fitControl,tuneLength=3)
#KNN
model_knn<-train(train_final[,predictors],train_final[,outcomeName],method='knn',trControl=fitControl,tuneLength=15)

summary(model_knn)




train_final$OOF_pred_rf<-model_rf$pred$Y[order(model_rf$pred$rowIndex)]
train_final$OOF_pred_knn<-model_knn$pred$Y[order(model_knn$pred$rowIndex)]
train_final$OOF_pred_lr<-model_lr$pred$Y[order(model_lr$pred$rowIndex)]

train_final$OOF_pred_rf<-ifelse(train_final$OOF_pred_rf>0.5,1,0)
train_final$OOF_pred_knn<-ifelse(train_final$OOF_pred_knn>0.5,1,0)
train_final$OOF_pred_lr<-ifelse(train_final$OOF_pred_lr>0.5,1,0)

cm=table(train_final$Loan_Status,train_final$OOF_pred_rf)

Accuracy=(cm[1]+cm[4])/(sum(cm))
Accuracy
Precision = cm[1] / (sum(cm[1,]))
Precision
Recall = cm[1] / (sum(cm[,1]))
Recall
F1_Score = (2 * Precision * Recall ) / (Precision + Recall)
F1_Score



##predictign for test data
test_final$OOF_pred_rf<-predict(model_rf,test_final[predictors],type='prob')$Y
test_final$OOF_pred_knn<-predict(model_knn,test_final[predictors],type='prob')$Y
test_final$OOF_pred_lr<-predict(model_lr,test_final[predictors],type='prob')$Y
test_final$Loan_Status<-(test_final$OOF_pred_rf+test_final$OOF_pred_knn+test_final$OOF_pred_lr)/3


test_final$OOF_pred_rf<-ifelse(test_final$OOF_pred_rf>0.5,1,0)
test_final$OOF_pred_knn<-ifelse(test_final$OOF_pred_knn>0.5,1,0)
test_final$OOF_pred_lr<-ifelse(test_final$OOF_pred_lr>0.5,1,0)
test_final$Loan_Status<-ifelse(test_final$Loan_Status>0.5,"Y","N")


write.csv(subset(test_final,select=c("Loan_ID","Loan_Status")),"Submission_3alg.csv",row.names=F)



###################Non caret packages###################3
#LOGISTIC REGRESSION
model_lr = glm(formula = train_final[,outcomeName]~.,family = binomial,data = train_final[,predictors])
prob_pred = predict(model_lr, type = 'response', newdata = test_final[,predictors])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
test_final$OOF_pred_lr2=y_pred

#DECISION TREES
library(rpart)
model_dt = rpart(formula = train_final[,outcomeName]~.,data = train_final[,predictors])
y_pred = predict(model_dt, newdata = test_final[,predictors], type = 'class')
test_final$OOF_pred_dt2=y_pred
fancyRpartPlot(model_dt)


write.csv(test_final,"test_final.csv",row.names = F)

#RANDOM FOREST
library(randomForest)
model_rf = randomForest(x = train_final[,predictors],y = train_final[,outcomeName],ntree = 500)
y_pred = predict(model_rf, newdata = test_final[,predictors], type = 'class')
test_final$OOF_pred_rf2=y_pred



