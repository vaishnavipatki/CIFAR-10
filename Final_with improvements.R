####################### ISEN 613 Project: Image Object Detection ########################
#Instructions:
# 1. Copy paste the path of the working directory in the setwd function
# 2. Make \\ in place of all \ in the path.
# 3. Run the code

#Set Working Directory
setwd('C:\\Users\\kastu\\Desktop\\Important data\\TAMU\\03_Courses_Fall 2019\\ISEN 613_Data Analysis\\01_project\\batch_bin\\batch_bin')

#########################################################################################
############################# Image Extraction: Training Data ###########################
#########################################################################################
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

#################### Creating Dataframe: Training Data ############################
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

############################################################################################
###################### Finding best parameters from Training Data ##########################
############################################################################################

###################### Splitting the data ########################
set.seed(1)
size <- sample(1:nrow(data.image), nrow(data.image)*0.75) 
train.x <- data.image[size,]
train.y <- data.label[c(size),]

test.x <- data.image[-size,]
test.y <- data.label[c(-size),]

###################### Applying PCA on train and test #############################
set.seed(1)
pc.x <- prcomp (train.x , scale =TRUE)
pr.var =pc.x$sdev ^2
pve=pr.var/sum(pr.var)
plot(cumsum (pve[1:500] ), xlab=" Principal Component ", ylab ="Cumulative Proportion of Variance Explained ", ylim=c(0,1) , type="b")
cumsum (pve[1:500])*100

train.data <- data.frame(pc.x$x)
test.data <- predict(pc.x, newdata=test.x)
test.data <- as.data.frame(test.data)

##################################################################
########################## Minimum Error QDA #####################
##################################################################
set.seed(1)
library(MASS)
lowerlim = 105
upperlim = 250
n <- length(lowerlim:upperlim)
Accuracy <- rep(NA, n)
for (k in lowerlim:upperlim)
{
  #k=500
  train.data1 <- data.frame(pc.x$x[, 1:k])
  train1 = data.frame(pc.x$x[, 1:k], train.y)
  test.data1 <- test.data[ ,1:k]
  test1= data.frame(test.data, test.y)
  qda1 <- qda(train.y~., data =train1)
  pred<-predict(qda1,test1)
  CM = table(pred$class,test1$test.y)
  acc = (sum(diag(CM)))/sum(CM)
  #Accuracy = acc*100
  #Accuracy
  Accuracy [k-(lowerlim - 1)] = acc*100
}
plot(lowerlim:upperlim, Accuracy, type="b", xlab = "Number of PCs", ylim = c(50.0, 52.0))
best = which.max(Accuracy) + 104
maxAcc = max(Accuracy)
best
maxAcc

######################################################################################
############################## Extract code : Final Test Data ########################
######################################################################################
images.rgb.final <- list()
images.lab.final <- list()
num.images.final = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 5 binary files
to.read <- file(paste("test_batch.bin"), "rb")
for(i in 1:num.images.final) 
{
  l <- readBin(to.read, integer(), size=1, n=1, endian="big")
  r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
  g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
  b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
  images.rgb.final[[i]] = data.frame(r, g, b)
  images.lab.final[[i]] = l+1
}
close(to.read)
remove(l,r,g,b,i, to.read)

###################### Create dataframe: Final Test Data ##########################
df_final <- list()
df.labels_final <- list()
for (index in 1:10000)
{
  images_final = unlist(images.rgb.final[[index]])
  labels_final = unlist(images.lab.final[[index]])
  df_final[[index]] = images_final
  df.labels_final[[index]] = labels_final
}

data.image_final = as.data.frame(do.call(rbind,df_final))
data.label_final = as.data.frame(do.call(rbind,df.labels_final))
colnames(data.label_final) <- "TrueLabTest"

#########################################################################################
########################### Final training and test dataframes ##########################
#########################################################################################

set.seed(1)
size <- sample(1:nrow(data.image), nrow(data.image)*0.75) 
train.x <- data.image[size,]
train.y <- data.label[c(size),]

set.seed(1)
size.final <- sample(1:nrow(data.image_final), nrow(data.image_final)*1) 
final.x <- data.image_final[size.final,]
final.y <- data.label_final[c(size.final),]

##########################################################################################
################################ PCA on Final Test Dataset ###############################
##########################################################################################

train.data <- data.frame(pc.x$x)
final.data <- predict(pc.x, newdata=final.x)
final.data <- as.data.frame(final.data)

##########################################################################################
################################## FINAL QDA #############################################
##########################################################################################
set.seed(1)
library(MASS)
k= best
train.data1 <- data.frame(pc.x$x[, 1:k])
train1 = data.frame(pc.x$x[, 1:k], train.y)
final.data <- final.data[ ,1:k]
final= data.frame(final.data, final.y)
qda1 <- qda(train.y~., data =train1)
pred<-predict(qda1,final)
CM = table(pred$class,final$final.y)
acc = (sum(diag(CM)))/sum(CM)
Accuracy = acc*100
Accuracy
#Accuracy [k-(lowerlim - 1)] = acc*100