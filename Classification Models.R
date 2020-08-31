# [Neelan,Muthurajah]
# [20195484]
# [Masters of Management Analytics]
# [Section 2]
# [MMA 869]
# [Aug 16th 2020]


# Submission to Question [7], Part 2 [a,b,c]

#Imports R-packages that are needed to run code below. If user does not have package installed, pacman will install package for the user. 
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","partykit","ROCR","MASS","lift","glmnet","e1071", "readxl","rpart","randomForest","xgboost","dplyr","ggplot2","tidyr","lubridate","MLmetrics","tidyverse","factoextra","partykit","fastDummies") 

#Import OJ data file 
oj <- read.csv(file.choose(),header=T, sep=",")

#Explore data structure
head(oj)

#Remove First Column as its an index column and not needed 
oj<-oj%>%
  dplyr::select(3:19,2)

#Summarize the data 
summary(oj)

#Analyze variables with whole numbers as min and max. These should be categorical variables. 
hist(oj$StoreID)
hist(oj$SpecialCH)
hist(oj$SpecialMM)

#Convert character variables/categorical features (identified above) to factors
oj$Store7<-as.factor(oj$Store7)
oj$Purchase<-as.factor(oj$Purchase)
oj$StoreID<-as.factor(oj$StoreID)
oj$SpecialCH<-as.factor(oj$SpecialCH)
oj$SpecialMM<-as.factor(oj$SpecialMM)

#Store 7 & STORE are captured with store ID so it can be removed (redundant features)
#Sale price (SalePriceMM & SalePriceCH), PctDiscMM, PctDiscCH & ListPriceDiff were calculated features from other features in the dataset so they will be removed for now. 
#Will revisit potentially re-adding these calculated fields back in if the performance metrics of the classfiers below are subpar
oj<-oj%>%
  dplyr::select(1:9,18)

#Standardize numerical features 
oj_numeric<-oj%>%
  dplyr::select(3:6,9)

oj_numeric<-scale(oj_numeric)

#select factor variables
oj_factors<-oj%>%
  dplyr::select(1,7,8,10)

#One hot encode store ID 
oj_storeid<-oj%>%
  dplyr::select(2)

oj_storeid<-dummy_columns(oj_storeid)

oj_storeid<-oj_storeid%>%
        dplyr::select(2:6)

#Combine Final Dataset for modeling 
oj<-cbind(oj_numeric,oj_storeid,oj_factors)

#Distribution of Minute Maid vs CH Purchases
#60% are CH vs 40% are MM. User defined thresholds below to predict MM purchases will be 0.4 to reflect this observed split. 
ggplot(oj) + geom_bar(aes(x = Purchase))

Distribution<-oj%>%
  group_by(Purchase)%>%
  summarize(Count=n())

#Split data into training and test sets. The two datasets were mutually exclusive. The test dataset would serve as future validation data. 
set.seed(123)
TrainingIndex<-createDataPartition(oj$Purchase, p=0.7, list=FALSE)#Split data into 70% train and 30% test 
trainingset<-oj[TrainingIndex,]
testingset<-oj[-TrainingIndex,]


####################################################################################################################################

#####LOGISTIC REGRESSION########

#Determine how R is numbering Purchase variable. In this case CH is 0 and MM is 1
contrasts(oj$Purchase)

#Logistic Regression 
set.seed(11)
model_logistic<-glm(Purchase ~ ., family="binomial", data=trainingset)
summary(model_logistic) 

#Re-run logistic regression model using stepwise to determine most impactful features
model_logistic_stepwiseAIC<-stepAIC(model_logistic,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC) 

###Determine probabilities and classify instances using user defined threhold value of 40% or 0.4
logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=testingset,type="response") #Predict probabilities of class 1 or MM. Changed this line to be either model_logisitc or model_logistic_stepwise
logistic_classification<-rep("CH",320) #Creates row of 320 CH's to reflect test data
logistic_classification[logistic_probabilities>=0.4]="MM" #Predict classification using 0.4 threshold  
logistic_classification<-as.factor(logistic_classification)#converts to factor for confusion matrix code to run without issue

###Confusion matrix & F1 Score Calculation 
confusionMatrix(logistic_classification,testingset$Purchase) 
F1_Score(logistic_classification,testingset$Purchase)

####ROC Curve
logistic_ROC_prediction <- prediction(logistic_probabilities, testingset$Purchase)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") #Create AUC data
logistic_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
logistic_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - average, below 60% - not much value

####################################################################################################################################

#####DECISION TREES########

#Minsplit restricts growth of tree
#mincriterion is confidence level 
set.seed(10)
ctree_tree<-ctree(Purchase~.,data = trainingset, control = ctree_control(mincriterion = 0.9,mtry=5,minsplit = 5,maxdepth=10,minbucket=5))

plot(ctree_tree)

###Determine probabilities and classify instances using user defined threhold value of 40% or 0.4
ctree_probabilities<-predict(ctree_tree,newdata=testingset,type="prob") #Predict probabilities
ctree_probabilities

ctree_classification<-rep("CH",320) #Creates row of 320 CH's to reflect test data
ctree_classification[ctree_probabilities[,2]>=0.4]="MM" #Predict classification using 0.4 threshold  
ctree_classification<-as.factor(ctree_classification)#converts to factor for confusion matrix code to run without issue

###Confusion matrix & F1 Score Calculation 
confusionMatrix(ctree_classification,testingset$Purchase)
F1_Score(ctree_classification,testingset$Purchase)

####ROC Curve
DT_ROC_prediction <- prediction(ctree_probabilities[,2], testingset$Purchase)
DT_ROC <- performance(DT_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(DT_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(DT_ROC_prediction,"auc") #Create AUC data
DT_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
DT_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - average, below 60% - not much value


####################################################################################################################################

#####RANDOM FOREST########

set.seed(12)
model_forest <- randomForest(Purchase~., data=trainingset, 
                             type="classification",
                             importance=TRUE,
                             ntree = 500,           # hyperparameter: number of trees in the forest
                             mtry = 10,             # hyperparameter: number of random columns to grow each tree
                             nodesize = 20,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 20,         # hyperparameter: maximum number of leafs of a tree
                             cutoff = c(0.5, 0.5)   # hyperparameter: how the voting works; (0.5, 0.5) means majority vote
) 

#Summary of random forest model
model_forest

###Determine probabilities and classify instances using user defined threhold value of 40% or 0.4
forest_prediction<-predict(model_forest,newdata=testingset, type="response")
forest_probabilities<-predict(model_forest,newdata=testingset,type="prob") #Displays an array with 2 columns: Probability of being in class 0 or CH vs probability of class 1 or MM
forest_classification<-rep("CH",320) #Creates row of 320 CH's to reflect test data
forest_classification[forest_probabilities[,2]>=0.4]="MM" #Predict classification using 0.4 threshold  
forest_classification<-as.factor(forest_classification)#converts to factor for confusion matrix code to run without issue

###Confusion matrix & F1 Score Calculation 
confusionMatrix(forest_classification,testingset$Purchase)
F1_Score(forest_classification,testingset$Purchase)

####ROC Curve
RF_ROC_prediction <- prediction(forest_probabilities[,2], testingset$Purchase)
RF_ROC <- performance(RF_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(RF_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(RF_ROC_prediction,"auc") #Create AUC data
RF_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
RF_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - average, below 60% - not much value

#Variable importance plot for the random forest model 
varImpPlot(model_forest)

####################################################################################################################################

#####XG BOOST########

#Convert CH to 0 and MM to 1 for both training and test sets 
trainingset$Purchase<-ifelse(trainingset$Purchase=="CH",0,1)
testingset$Purchase<-ifelse(testingset$Purchase=="CH",0,1)

set.seed(14)
ojdata_matrix <- model.matrix(Purchase~ ., data = oj)[,-1]

x_train <- ojdata_matrix[ TrainingIndex,]
x_test <- ojdata_matrix[ -TrainingIndex,]

y_train <-trainingset$Purchase
y_test <-testingset$Purchase

model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(y_train), 
                       eta = 0.1,      # hyperparameter: learning rate 
                       max_depth = 20, # hyperparameter: size of a tree in each boosting iteration
                       nround=50,      # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic"
)

###Determine probabilities and classify instances using user defined threhold value of 40% or 0.4
XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="response") #Predict probabilities of class 1 or MM
XGboost_classification<-as.factor(ifelse(XGboost_prediction>=0.4,1,0))#Predict classification using 0.4 threshold + converts to factor for confusion matrix code to run without issue

###Confusion matrix & F1 Score Calculation 
confusionMatrix(XGboost_classification,as.factor(y_test),positive="1")
F1_Score(XGboost_classification,testingset$Purchase)

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - average, below 60% - not much value


