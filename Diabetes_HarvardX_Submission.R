#load required packages

library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)
library(corrr)
library(psych)
library(corrplot)
library(MASS)
library(scales)
library(readr)

#read data
#dataset source : https://www.kaggle.com/mathchi/diabetes-data-set and saved in github repository.

urlfile="https://raw.githubusercontent.com/kanin-limniyakul/Diabetes-ML/main/diabetes.csv"

dat<-read_csv(url(urlfile))

dat$Outcome <- as.factor(dat$Outcome)

#Exploratory Data Analysis

str(dat) # Data Structure

# create feature description table

var <- c("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Oucome")
des <- c("No of times pregnant", "Plasma glucose concentration a 2 hours in an oral glucose tolerance test", "Diastolic blood pressure (mm Hg)","Triceps skin fold thickness (mm)", "2-Hour serum insulin (mu U/ml)","Body mass index (weight in kg/(height in m)^2)", "Diabetes pedigree function","Age (years)","Factor class (0 or 1), 0= no diabetes, 1 =  diabetes ")

descriptive<- data_frame( Name = var, Description = des)
head <- head(dat)
knitr::kable (descriptive , caption = "Features and Outcomes Descriptions" )
knitr::kable (head , caption = "The first 6 rows of the dataset" )

#summary outcome to create bar graph

datn<- dat %>% group_by(Outcome) %>% summarize(Cases = n())
datn<- as.data.frame(datn)
datn %>% ggplot(aes(x = Outcome, y= Cases, fill = Outcome)) + geom_bar(stat = "identity") +geom_text(aes(label = Cases), nudge_y = 25)

#create heatmap correlation plot

temp <-dat

temp$Outcome <- as.numeric(temp$Outcome) #temporary convert to numeric for heatmap
corrplot(cor(temp[1:9]),        # Correlation matrix
         method = "shade", # Correlation plot method
         type = "full",    # Correlation plot style (also "upper" and "lower")
         diag = TRUE,      # If TRUE (default), adds the diagonal
         tl.col = "black", # Labels color
         bg = "white",     # Background color
         title = "",       # Main title
         col = NULL)       # Color palette

#create features plot

featurePlot(x = dat[, 1:8], 
            y = dat$Outcome, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,2 ), 
            auto.key = list(columns = 2))

#Check near zero variance feature

nzv <- nearZeroVar(dat[,1:8], saveMetrics = TRUE)
knitr::kable(nzv, caption = "Checking Near Zero Variance Predictors")

# Pre-Processing data

#standardized data

preProcValues <- preProcess(dat[1:8], method = c("center", "scale"))
dat_n <- predict(preProcValues, dat)
norm_dat <- head(dat_n)
knitr::kable(norm_dat, column_spec= (width = "5cm"), caption = " The first six rows of scaled dataset")

#density plot after standardized

featurePlot(x = dat_n[, 1:8], 
            y = dat$Outcome, 
            plot = "density", 
            ## Pass in options to densityplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,2 ), 
            auto.key = list(columns = 2))

#data partitioning 80% train 20% test

set.seed(2022)
test_index <- createDataPartition(dat$Outcome, times = 1, p = 0.2, list = FALSE)
test_set<- dat_n[test_index, ]
train_set<- dat_n[-test_index, ]
ntest <- nrow(test_set)
ntrain <- nrow(train_set)
partition <- data.frame(Name = c("train", "test"), n = c(ntrain,ntest))
knitr::kable (partition , caption = "train/test split")

## Logistic Regression

#train the model
train_glm_all <- train(Outcome ~ ., method = "glm", data = train_set)
summary(train_glm_all)


#Remove insignificant features
train_glm_all_update <- train(Outcome ~ .-SkinThickness-Insulin -Age, method = "glm", data = train_set)

#making predictions
glm_all_preds <- predict(train_glm_all, test_set)
glm_all_preds_update <- predict(train_glm_all_update, test_set)

#calculate accuracy
glm_acc <- mean(glm_all_preds == test_set$Outcome) #glm accuracy
glm_acc_update <- mean(glm_all_preds == test_set$Outcome)

#Table compare train and test accuracy
result_glm <- data_frame(Method = c("Training Accuracy", "Training Accuracy-Update","Test Accuracy", "Test Accuracy-Update"), Accuracy =     c(train_glm_all$results$Accuracy,train_glm_all_update$results$Accuracy,glm_acc,glm_acc_update))
knitr::kable(result_glm, caption = "Accuracy Table (Logistic Regression)")

#Confusion Matrix
confusionMatrix(data = factor(glm_all_preds), reference = factor(test_set$Outcome))

## Linear Discriminat Analysis (LDA)

#train the model

model <- lda(Outcome~., data=train_set) 
plot(model)

train_results <- predict(model, train_set)

lda_train_acc <- mean(train_results$class == train_set$Outcome) #lda train accuracy

predicted <- predict(model, test_set) #prediction


#plot LD1 vs outcome

par(mfrow=c(1,1))
plot(predicted$x[,1], predicted$class, col = test_set$Outcome, xlab = "LD1", ylab = "Predicted Class")

#LDA Accuracy calculation

lda_pred_acc <- mean(predicted$class == test_set$Outcome) #lda accuracy


#LDA Summary table

result_lda <- data_frame(Method = c("Training Accuracy", "Test Accuracy"), Accuracy = c(lda_train_acc,lda_pred_acc))
knitr::kable(result_lda, caption = "Accuracy Table (LDA)")


confusionMatrix(data = factor(predicted$class), 
                reference = factor(test_set$Outcome))

## Quadratic Discriminat Analysis (QDA)

qmodel <- qda(Outcome~., data=train_set) #train the model


qtrain_results <- predict(qmodel, train_set)

qda_train_acc <- mean(qtrain_results$class == train_set$Outcome)#qda train accuracy

qpredicted <- predict(qmodel, test_set)

#plot prob vs predicted class

par(mfrow=c(1,1))
plot(qpredicted$posterior[,2], qpredicted$class, col = test_set$Outcome, xlab = "Predicted Posterior Probability", ylab = "Predicted Class")

qda_pred_acc <- mean(qpredicted$class == test_set$Outcome) #qda accuracy

#qda summary

result_qda <- data_frame(Method = c("Training Accuracy", "Test Accuracy"), Accuracy = c(qda_train_acc,qda_pred_acc))
knitr::kable(result_qda, caption = "Accuracy Table (QDA)")
confusionMatrix(data = factor(qpredicted$class), 
                reference = factor(test_set$Outcome))

## K-Nearest Neighbor Model(KNN)

set.seed(2022)
train_knn <- train(Outcome ~ .,
                   method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(3, 51, 2)))

knitr::kable(train_knn$bestTune, caption = "k best tune")

plot(train_knn)

#Prediction and Accuracy

knn_preds <- predict(train_knn, test_set) 
knn_acc <- mean(knn_preds == test_set$Outcome) 


#Table compare train and test accuracy


result_KNN <- data_frame(Method = c("Training Accuracy","Test Accuracy"), Accuracy = c(train_knn$results$Accuracy[which.max(train_knn$results$Accuracy)],knn_acc))
knitr::kable(result_KNN, caption = "Accuracy Table Comparison (KNN)")
confusionMatrix(data = factor(knn_preds), 
                reference = factor(test_set$Outcome))

#cross-validated KNN 10-fold, 10%

set.seed(2022)

train_knn_cv <- train(Outcome ~ .,
                      method = "knn",
                      data = train_set,
                      tuneGrid = data.frame(k = seq(3, 51, 2)),
                      trControl = trainControl(method = "cv", number = 10, p = 0.9))
knitr::kable(train_knn_cv$bestTune, caption = "k_cv best tune")
plot(train_knn_cv)

#prediction

knn_cv_preds <- predict(train_knn_cv, test_set)
knn_cv_acc <- mean(knn_cv_preds == test_set$Outcome)

#Table compare train and test accuracy

result_KNN_cv <- data_frame(Method = c("Training Accuracy","Test Accuracy"), Accuracy = c(train_knn_cv$results$Accuracy[which.max(train_knn_cv$results$Accuracy)],knn_cv_acc))
knitr::kable(result_KNN_cv, caption = "Accuracy Table Comparison (KNN_Cross Validation)")
confusionMatrix(data = factor(knn_cv_preds), 
                reference = factor(test_set$Outcome))

#Classification Tree Model

set.seed(2022)

train_rpart <- train(Outcome ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.06, 0.002)),
                     data = train_set)

knitr::kable(train_rpart$bestTune, caption = "Best complexity parameter tuning")

plot(train_rpart)

#plot decision tree
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel)

#prediction and accuracy
rpart_preds <- predict(train_rpart, test_set)
tree_acc <- mean(rpart_preds == test_set$Outcome)
tree_acc

#summarize results
result_CT <- data_frame(Method = c("Training Accuracy","Test Accuracy"), Accuracy = c(train_rpart$results$Accuracy[which.max(train_rpart$results$Accuracy)],tree_acc))
knitr::kable(result_CT, caption = "Accuracy Table Comparison (Classification Tree)")

confusionMatrix(data = factor(rpart_preds), 
                reference = factor(test_set$Outcome))

#random forest model

set.seed(2002)

train_rf <- train(Outcome ~ .,
                  data = train_set,
                  method = "rf",
                  ntree = 500,
                  tuneGrid = data.frame(mtry = seq(1:8)))

plot(train_rf)


knitr::kable(train_rf$bestTune, caption = "Best mtry RF model")

# Important Variables plot

plot(varImp(train_rf))

#RF prediction

rf_preds <- predict(train_rf, test_set)
rf_acc <- mean(rf_preds == test_set$Outcome)


#summarize results
result_rf <- data_frame(Method = c("Training Accuracy","Test Accuracy"), Accuracy = c(train_rf$results$Accuracy[which.max(train_rf$results$Accuracy)],rf_acc))
knitr::kable(result_rf, caption = "Accuracy Table Comparison (Random Forest)")

confusionMatrix(data = factor(rf_preds), 
                reference = factor(test_set$Outcome))

#ensemble model

ensemble <- cbind(logistic = glm_all_preds == 1,  
                  KNN = knn_preds == 1,  KNN_CV = knn_cv_preds == 1,
                  Classification_Tree = rpart_preds == 1,
                  Random_Forest = rf_preds ==1,
                  LDA = predicted$class  ==1,
                  QDA = qpredicted$class  ==1)


ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, 1, 0)

test_out <- ifelse(test_set$Outcome == 0, 0,1)


#ensembel model table

ensemble <- ensemble %>% cbind(ensemble_prediction = ensemble_preds, test_outcome = test_out)

knitr::kable(head(ensemble), caption = "Ensemble Model")



#calculate accuracy

ensemble_acc<- mean(ensemble_preds == test_set$Outcome)


# summary table

summary <- data.frame(Model = c("Logistic Regression", "LDA","QDA","KNN","KNN_CV","Classification Tree", "Random Forest","Ensemble"), Accuracy = c(glm_acc, lda_pred_acc,qda_pred_acc,knn_acc,knn_cv_acc,tree_acc,rf_acc, ensemble_acc)) 

knitr::kable(summary, caption = "Accuracy Table Comparison (All Models)")


