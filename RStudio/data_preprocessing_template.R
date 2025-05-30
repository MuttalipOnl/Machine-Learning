dataset = read.csv('Social_Network_Ads.csv')
# dataset = dataset[, 3:5]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 1:2] = scale(training_set[, 1:2])
# test_set[, 1:2] = scale(test_set[, 1:2])