# Random Forest Regression

# Veriyi ice aktarma
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Random Forest Regression' yi veri kumesine yerlestirme
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 100)

# Test seti sonuclarini tahmin etme
# y_pred = predict(regressor, data.frame(Level = 6.5))

# Egitim setini gosterme
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Years of Experience') +
  ylab('Salary') 

