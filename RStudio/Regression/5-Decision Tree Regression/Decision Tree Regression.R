# Decision Tree Regression

# Veriyi ice aktarma
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Decision Tree Regression' yi veri kumesine yerlestirme
# install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Test seti sonuclarini tahmin etme
y_pred = predict(regressor, data.frame(Level = 6.5))

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

