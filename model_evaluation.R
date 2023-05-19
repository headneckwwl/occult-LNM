install.packages("rmda")
library(pROC)



############################ROC曲线，delong检测  test
data<-read.csv('huizong_external_fordelong.csv',header = T)
head(data)

roc3D<-roc(data$label,data$threeD,plot=T)
roc3D    

roc2D<-roc(data$label,data$two,plot=T)
roc2D


rocmerge<-roc(data$label,data$Merge,plot=T)
rocmerge

rocrad<-roc(data$label,data$Rad,plot=T)
rocrad

roccl3D<-roc(data$label,data$cl3D,plot=T)
roccl3D

roccl<-roc(data$label,data$cl,plot=T)
roccl

rocclrad<-roc(data$label,data$clrad,plot=T)
rocclrad



roc.test(roc3D,rocrad,method='delong')





roc(data$label,data$threeD,plot=T,print.thres=F,print.auc=F,ci=T,col='red')
roc(data$label,data$twoD,plot=T,print.thres=F,print.auc=F,ci=T,add=T,col='green')
roc(data$label,data$Merge,plot=T,print.thres=F,print.auc=F,ci=T,add=T,col='pink')
roc(data$label,data$Rad,plot=T,print.thres=F,print.auc=F,ci=T,add=T,col='blue')
legend(0.8,0.5,'3D DCNN,AUC=0.87',lty = 1,lwd = 1,col = 'red',bty = 'n')
legend(0.8,0.4,'2D DCNN,AUC=0.842',lty = 1,lwd = 1,col = 'green',bty = 'n')
legend(0.8,0.3,'2D_3D_Rad,AUC=0.821',lty = 1,lwd = 1,col = 'pink',bty = 'n')
legend(0.8,0.2,'Radiomics,AUC=0.78',lty = 1,lwd = 1,col = 'blue',bty = 'n')



#################DCA曲线 DCA curves


library(rmda)

dataall<-read.csv('results.csv',header = T)

threeDmodel<-decision_curve(label~threeD,data = dataall)
cl3DDmodel<-decision_curve(label~cl3D,data = dataall)
clradmodel<-decision_curve(label~clrad,data = dataall)
clmodel<-decision_curve(label~cl,data = dataall)

plot_decision_curve(list(threeDmodel,cl3DDmodel,clmodel,clradmodel),curve.names = c('3D','Clirad_3D','cli','clrad'),
                    confidence.intervals = F,
                    standardize = F,
                    legend.position ='topright')





##########################校准曲线 calibration curves
library(rms)



datatrain<-read.csv('huizong_train_fordelong.csv',header = T)

dd<-datadist(datatrain)
options(datadist='dd')


fit3D<- lrm(label~threeD,data = datatrain,x=T,y=T)
summary(fit3D)
cal_3D<- calibrate(fit3D,cmethod="KM")
plot(cal_3D,xlim = c(0,1.0),ylim = c(0,1.0)) 


fit2D<- lrm(label~twoD,data = datatrain,x=T,y=T)
cal_2D<- calibrate(fit2D,cmethod="KM")
plot(cal_2D,xlim = c(0,1.0),ylim = c(0,1.0)) 


fitmerge<- lrm(label~Merge,data = datatrain,x=T,y=T)
cal_merge<- calibrate(fitmerge,cmethod="KM")
plot(cal_merge,xlim = c(0,1.0),ylim = c(0,1.0))


fitrad<- lrm(label~Rad,data = datatrain,x=T,y=T)
cal_rad<- calibrate(fitrad,cmethod="KM")
plot(cal_rad,xlim = c(0,1.0),ylim = c(0,1.0))



