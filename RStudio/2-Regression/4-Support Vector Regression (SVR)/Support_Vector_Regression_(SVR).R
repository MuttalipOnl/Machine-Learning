# SVR 

# Veriyi ice aktarma
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Svr'yi veri kumesine yerlestirme
# install.packages('e1071')
library(e1071)
regressor =svm(formula = Salary ~ .,
               data = dataset,
               type = 'eps-regression')

# Test seti sonuclarini tahmin etme
y_pred = predict(regressor, data.frame(Level = 6.5))

# Egitim setini gosterme
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Years of Experience') +
  ylab('Salary')      



