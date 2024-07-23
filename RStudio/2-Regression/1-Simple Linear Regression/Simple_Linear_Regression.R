# Simple Linear Regression

# Veriyi ice aktarma
datasets = read.csv('Salary_Data.csv')

# Veri k??mesini e??itim k??mesi ve test k??mesine ayirma
library(caTools)
set.seed(123)
split = sample.split(datasets$Salary, SplitRatio = 2/3)
training_set = subset(datasets, split == TRUE)
test_set = subset(datasets, split == FALSE)

# Dogrusal Regresyonun Egitim Setine Yerlestirilmesi
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

# Test seti sonuclarini tahmin etme
y_pred = predict(regressor, newdata = test_set)

# Egitim setini g??sterme
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             color = 'red') +

  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
    xlab('Years of Experience') +
    ylab('Salary')


# Test setini g??sterme
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             color = 'red') +
  
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

