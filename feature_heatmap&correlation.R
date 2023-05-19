setwd("F:\\hypopharyngeal_laryngeal\\18featureheatmap")
getwd()

library(pheatmap)
install.packages("modelsummary")
library(modelsummary)
library(showtext)
install.packages("showtext")
font_add('times','times.ttf')
showtext_auto()


################################heatmap 

rad<-read.csv('2D_features.csv',header = T,row.names = 1)
clinc<-read.csv('label.csv',header = T,row.names = 1)

rad<-as.data.frame(rad)
clinc<-as.data.frame(clinc)



rad1<- log2(rad+18)


pheatmap(rad,scale = "column",annotation_row = clinc,cluster_rows  = T,cluster_cols = T)
pheatmap(rad1)


##############################################相关性分析 corrleation analysis

library(limma)
library(corrplot)


merge_features<-read.csv('merge_features2.csv',header = T,row.names = 1)

#merge_features=t(merge_features)
M=cor(merge_features)
res1=cor.mtest(merge_features, conf.level = 0.95)

#??????????ͼ??
pdf(file="cor.pdf", width=8, height=8)
corrplot(M,
         order="original",
         method = "circle",
         type = "upper",
         tl.cex=0.8, pch=T,
         #p.mat = res1$p,
         insig = "label_sig",
         pch.cex = 1.6,
         sig.level=0.05,
         number.cex = 1,
         col=colorRampPalette(c("blue", "white", "red"))(50),
         tl.col="black")
dev.off()


##############################################################测试



