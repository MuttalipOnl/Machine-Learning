dataset = read.delim('Restaurant_Reviews.tsv', 
                     quote = '', stringsAsFactors = FALSE)

#install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus, content_transformer(tolower)) # W>w
corpus = tm_map(corpus, removeNumbers) #40>
corpus = tm_map(corpus, removePunctuation) #...>
corpus = tm_map(corpus, removeWords, stopwords()) #




