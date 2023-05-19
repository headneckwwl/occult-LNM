

library(pROC)


rad<-read.csv('huizong_train_fordelong.csv',header = T)
head(rad)

roccl3D <- roc(rad$label, rad$cl3D)

plot(roccl3D,
     legacy.axes = TRUE,
     main="ROC曲线最佳阈值点",
     thresholds="best", # 基于youden指数选择roc曲线最佳阈值点
     print.thres="best") # 在roc曲线上显示最佳阈值点


# 获取最佳阈值
roc_result <- coords(rocobj, "best")
# 计算在最佳阈值下混淆矩阵各项的值
TP <- dim(aSAH[as.numeric(aSAH$outcome)==2 & aSAH$s100b > roc_result$threshold, ])[1]
FP <- dim(aSAH[as.numeric(aSAH$outcome)==1 & aSAH$s100b > roc_result$threshold, ])[1]
TN <- dim(aSAH[as.numeric(aSAH$outcome)==1 & aSAH$s100b <= roc_result$threshold, ])[1]
FN <- dim(aSAH[as.numeric(aSAH$outcome)==2 & aSAH$s100b <= roc_result$threshold, ])[1]

TPR <- TP / (TP + FN)
TNR <- TN / (TN + FP)
ACC <- (TP + TN) / (TP + TN + FP + FN)


#######################



