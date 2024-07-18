# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer  # eksik verileri doldurma
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # eksik verileri doldurma
imputer.fit(X[:, 1:3])  # eksik verileri doldurma
X[:, 1:3] = imputer.transform(X[:, 1:3])  # eksik verileri doldurma

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer  # kategorik verileri sayısal verisine çevirme
from sklearn.preprocessing import OneHotEncoder  # kategorik verileri sayısal verisine çevirme
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # kategorik verileri sayısal verisine çevirme
X = np.array(ct.fit_transform(X))  # kategorik verileri sayısal verisine çevirme

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder  # kategorik verileri sayısal verisine çevirme
le = LabelEncoder()  # kategorik verileri sayısal verisine çevirme
y = np.array(le.fit_transform(y))  # kategorik verileri sayısal verisine çevirme
y = le.fit_transform(y)  # kategorik verileri sayısal verisine çevirme

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler  # veriyi ölçeklendirme
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print("Özellik Nesnesi(X)")
print(X)
print("\n")
print("Bağımlı Değişken(y)")
print(y)
print("\n")
print("X Training Set")
print(X_train)
print("\n")
print("X Test Set")
print(X_test)
print("\n")
print("y Training Set")
print(y_train)
print("\n")
print("y Test Set")
print(y_test)

