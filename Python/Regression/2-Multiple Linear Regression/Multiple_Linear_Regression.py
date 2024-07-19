import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test) # kat sayı alma
np.set_printoptions(precision=2) # alt çizgi yazdırma,, vrgülden sonra iki ondallık sayı içern herhangi sayısal değeri gösterme.
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1)) # yatay olarak, iki vektörü ve hatta dizileri birleştiren bir işlevdir.

# Soru1:
# Çoklu doğrusal regresyon modelini kullanarak örneğin, Ar-Ge Harcaması = 160000, Yönetim Harcaması = 130000, Pazarlama Harcaması = 300000 ve Eyalet = Kaliforniya olan bir girişimin kârını tahmin etmek için nasıl kullanırım?
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

#Dolayısıyla modelimiz, Ar-Ge'ye 160.000, İdari İşlere 130.000 ve Pazarlamaya 300.000 dolar harcayan bir Kaliforniya girişiminin karının 181.566,92 dolar olacağını öngörüyor.

# Soru2:
# Katsayıların son değerleri ile son regresyon denklemi y = b0 + b1 x1 + b2 x2 + ... nasıl elde edilir?
print(regressor.coef_)
print(regressor.intercept_)



