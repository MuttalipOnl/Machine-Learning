dataset = read.csv('mall.csv')
X = dataset[4:5]

dendrogram = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distances')

hc = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
y_hc = cutree(hc, 5, )

library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters of clients"),
         xlab = "Annual Income",
         ylab = "Spending Score")