###################### Image Extract Program ############################################
setwd('')
labels <- read.table("batches.meta.txt")
images.rgb <- list()
images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 5 binary files
for (f in 1:5) {
  to.read <- file(paste("data_batch_", f, ".bin", sep=""), "rb")
  for(i in 1:num.images) {
    l <- readBin(to.read, integer(), size=1, n=1, endian="big")
    r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    index <- num.images * (f-1) + i
    images.rgb[[index]] = data.frame(r, g, b)
    images.lab[[index]] = l+1
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}


################### Creating Dataframe ############################
set.seed(1)
df <- list()
df.labels <- list()
for (index in 1:50000)
  
{
  images = unlist(images.rgb[[index]])
  labels = unlist(images.lab[[index]])
  df[[index]] = images
  df.labels[[index]] = labels
  
}

data.image = as.data.frame(do.call(rbind,df))
data.label = as.data.frame(do.call(rbind,df.labels))
colnames(data.label) <- "TrueLab"



###################### Splitting the data ########################
set.seed(1)
size <- sample(1:nrow(data.image), nrow(data.image)*0.75) 
train.x <- data.image[size,]
test.x <- data.image[-size,]
train.y <- data.label[c(size),]
test.y <- data.label[-c(size),]


###################### Applying PCA on train #############################
set.seed(1)
pc.x <- prcomp (train.x , scale =TRUE)
pr.var =pc.x$sdev ^2
pve=pr.var/sum(pr.var)
plot(pve [1:250], xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1) ,type="b")
plot(cumsum (pve[1:250] ), xlab=" Principal Component ", ylab ="Cumulative Proportion of Variance Explained ", ylim=c(0,1) , type="b")


cumsum (pve[1:250])*100

train.data <- data.frame(pc.x$x)
train.data <- data.frame(pc.x$x[, 1:105])
train = data.frame(pc.x$x[, 1:105], train.y)


############### PCA on test ################# 
set.seed(1)
test.data <- predict(pc.x, newdata=test.x)
test.data <- as.data.frame(test.data)
test.data <- test.data[,1:105]
test= data.frame(test.data, test.y)

############ QDA ##############
set.seed(1)
library(MASS)
qda1<-qda(train.y~., data =train)
pred<-predict(qda1,test)
CM = table(pred$class,test$test.y)
CM
acc = (sum(diag(CM)))/sum(CM)
acc


################ Random Forests ###########
set.seed(1)
library(randomForest)
rf <- round((105)^0.5)
fit.rf = randomForest(factor(train.y) ~ ., data=train, mtry = rf, ntree=300, importance = TRUE)
fitrf.pred <-predict(fit.rf, newdata=test.data)
Accuracy <- mean(fitrf.pred == test.y)
Accuracy
importance(fit.rf)
varImpPlot(fit.rf)
#################### KNN #######################
set.seed(1)
library(class)
knn.fit=knn(train.data,test.data,train.y,k=10)
count=mean(knn.fit==test.y)
count
summary(knn.fit)
table(knn.fit, test.y)

