# Multiple Linear Regression

# Veriyi ice aktarma
datasets = read.csv('50_Startups.csv')

# Kategorik verileri kodlama
datasets$State = factor(datasets$State, 
                          levels = c('New York', 'California', 'Florida'),
                          labels = c(1, 2, 3))


# Veri kumesini egitim kumesi ve test kumesine ayirma
library(caTools)
set.seed(123)
split = sample.split(datasets$Profit, SplitRatio = 0.8)
training_set = subset(datasets, split == TRUE)
test_set = subset(datasets, split == FALSE)

# ??oklu Dogrusal Regresyonun Egitim Setine Yerlestirilmesi
regressor = lm(formula = Profit ~ ., 
               data = training_set)
# Test seti sonu??lar??n?? tahmin etmek
y_pred = predict(regressor, newdata = test_set)



