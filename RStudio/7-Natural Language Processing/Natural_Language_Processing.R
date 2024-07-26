# dataset = read.delim('Restaurant_Reviews.tsv', 
#                          quote = '', stringsAsFactors = FALSE, fileEncoding = "ASCII")
dataset_ori = read.delim('Restaurant_Reviews.tsv', 
                     quote = '', stringsAsFactors = FALSE, fileEncoding = "ASCII")

#install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus, content_transformer(tolower)) # W>w
corpus = tm_map(corpus, removeNumbers) #40>
corpus = tm_map(corpus, removePunctuation) #...>
corpus = tm_map(corpus, removeWords, stopwords()) #this>
corpus = tm_map(corpus, stemDocument) #loved>love
corpus = tm_map(corpus, stripWhitespace) #a,i,for>

dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_ori$Liked

dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

y_pred = predict(classifier, newdata = test_set[-692])

cm = table(test_set[, 692], y_pred)

