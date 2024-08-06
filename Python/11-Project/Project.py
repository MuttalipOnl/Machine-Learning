import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time

tf.__version__

# Veri setlerini yükleme
training_set = pd.read_csv('UNSW_NB15_training-set.csv')
testing_set = pd.read_csv('UNSW_NB15_testing-set.csv')

# Kategorik sütunlar
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['proto', 'service', 'state']
# Eğitim ve test veri setleri için ayrı ayrı LabelEncoder kullanarak kategorik verileri sayısallaştırma
for col in categorical_columns:
    le = LabelEncoder()
    training_set[col] = le.fit_transform(training_set[col])
    # Test verisini de aynı encoder ile dönüştürebilmek için ayrı bir encoder kullanmalıyız
    # Dikkat: test setini eğitirken eğitim setinden öğrendiğimiz bilgileri kullanmalıyız
    le_test = LabelEncoder()
    le_test.fit(list(training_set[col].unique()) + list(testing_set[col].unique()))  # Tüm olası sınıfları kapsar
    testing_set[col] = le_test.transform(testing_set[col])
    
# Özellikleri ölçeklendirme
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_columns = [col for col in training_set.columns if col not in ['id', 'attack_cat']]
training_scaled = training_set.copy()
testing_scaled = testing_set.copy()

# Sadece özellikleri ölçeklendirin
training_scaled[feature_columns] = scaler.fit_transform(training_set[feature_columns])
testing_scaled[feature_columns] = scaler.transform(testing_set[feature_columns])

X_train = training_scaled[feature_columns]
y_train = training_scaled['attack_cat']

# SMOTE işlemi öncesi grafiği
plt.figure(figsize=(8, 6)) # Grafik Boyutu
sns.countplot(x=y_train, palette='viridis') # bar grafiği sns
plt.title('SMOTE Öncesi Sınıf Dağılımı')
plt.xlabel('Sınıflar')
plt.xticks(rotation=90)
plt.ylabel('Örnek Sayısı')
plt.show()

# SMOTE ile veri dengeleme
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# SMOTE işlemi sonrası grafiği
plt.figure(figsize=(8, 6)) 
sns.countplot(x=y_resampled, palette='viridis') 
plt.title('SMOTE Sonrası Sınıf Dağılımı')
plt.xlabel('Sınıflar')
plt.xticks(rotation=90)
plt.ylabel('Örnek Sayısı')
plt.show()

# Sınıf etiketlerini sayısallaştırma
le_attack_cat = LabelEncoder()
le_attack_cat.fit(training_set['attack_cat'])
training_scaled['attack_cat'] = le_attack_cat.transform(training_scaled['attack_cat'])
testing_scaled['attack_cat'] = le_attack_cat.transform(testing_scaled['attack_cat'])

X_train = training_scaled[feature_columns]
y_train = training_scaled['attack_cat']
X_test = training_scaled[feature_columns]
y_test = training_scaled['attack_cat']

# Model performansını değerlendirme fonksiyonu
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
def model(model, X_train, y_train, X_test, y_test):
    start_train_time = time.time()
    model.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time
    
    start_test_time = time.time()
    y_pred = model.predict(X_test)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
   
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model: {model.__class__.__name__}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Test Time: {test_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_attack_cat.classes_, yticklabels=le_attack_cat.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model.__class__.__name__}')
    plt.show()

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import catboost as cb
# XGBoost modeli
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(y_test.unique()), random_state=42)
model(xgb_model, X_train, y_train, X_test, y_test)

# Random Forest Classifier
rfc_model = RandomForestClassifier(random_state=42)
model(rfc_model, X_train, y_train, X_test, y_test)

# CatBoost Classifier
catboost_model = cb.CatBoostClassifier(verbose=0, random_state=42)
model(catboost_model, X_train, y_train, X_test, y_test)

# AdaBoost Classifier
adaboost_model = AdaBoostClassifier(random_state=42)
model(adaboost_model, X_train, y_train, X_test, y_test)

# Artificial Neural Network (ANN)
# Modelin Tanımlanması:
ann_model = tf.keras.Sequential([
    # Giris katmanı ve ilk gizli katmanı ekleme(6 genel)
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
    # İkinci gizli katmanı ekleme
    tf.keras.layers.Dense(units=64, activation='relu'),
    # Çıkış katmanı ekleme
    tf.keras.layers.Dense(len(le_attack_cat.classes_), activation='softmax')
])

# Modelin Derlenmesi:
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin Eğitilmesi:
start_train_time = time.time()
ann_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
end_train_time = time.time()
training_time_ann = end_train_time - start_train_time

# Tahminler Yapılması:
start_test_time = time.time()
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=-1)
end_test_time = time.time()
test_time_ann = end_test_time - start_test_time

accuracy_ann = accuracy_score(y_test, y_pred_ann)
recall_ann = recall_score(y_test, y_pred_ann, average='weighted')
precision_ann = precision_score(y_test, y_pred_ann, average='weighted')
f1_ann = f1_score(y_test, y_pred_ann, average='weighted')


print("\nModel: ANN")
print(f"Training Time: {training_time_ann:.4f} seconds")
print(f"Test Time: {test_time_ann:.4f} seconds")
print(f"Accuracy: {accuracy_ann:.4f}")
print(f"Recall: {recall_ann:.4f}")
print(f"Precision: {precision_ann:.4f}")
print(f"F1 Score: {f1_ann:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ann))
cm_ann = confusion_matrix(y_test, y_pred_ann)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', xticklabels=le_attack_cat.classes_, yticklabels=le_attack_cat.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: ANN')
plt.show()

"""
Diğer Modeller: XGBoost, Random Forest, CatBoost ve AdaBoost gibi modellerin eğitim ve 
tahmin süreçleri genellikle tek bir fonksiyon çağrısıyla yönetilir (fit, predict). 

ANN (Yapay Sinir Ağı): Yapay sinir ağlarının eğitim süreci daha karmaşıktır ve 
genellikle birden fazla adım içerir. Bu adımlar şunları içerir:

Modeli Tanımlama: Modelin yapısının (katmanlar ve nöronlar) belirlenmesi.
Modeli Derleme: Kaybı ve optimizasyon yöntemini tanımlama.
Modeli Eğitme: Eğitim verisiyle modelin eğitilmesi (fit).
Tahmin Yapma: Test verisiyle tahminler yapma (predict).

ReLU, nöronun giriş değerini 0'dan küçükse 0 yapar, 0'dan büyükse olduğu gibi bırakır.
"""

# Convolutional Neural Network (CNN)
# Modelin Tanımlanması:
cnn_model = tf.keras.Sequential([
    # Giriş şekli (örnek sayısı, özellik sayısı, kanal sayısı) (CNN için 1D girdi)
    tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(len(le_attack_cat.classes_), activation='softmax')
])

# Modelin Derlenmesi:
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin Eğitilmesi:
start_train_time = time.time()
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
end_train_time = time.time()
training_time_cnn = end_train_time - start_train_time

# Tahminler Yapılması:
start_test_time = time.time()
y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=-1)
end_test_time = time.time()
test_time_cnn = end_test_time - start_test_time

accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
recall_cnn = recall_score(y_test, y_pred_cnn, average='weighted')
precision_cnn = precision_score(y_test, y_pred_cnn, average='weighted')
f1_cnn = f1_score(y_test, y_pred_cnn, average='weighted')

print("\nModel: CNN")
print(f"Training Time: {training_time_cnn:.4f} seconds")
print(f"Test Time: {test_time_cnn:.4f} seconds")
print(f"Accuracy: {accuracy_cnn:.4f}")
print(f"Recall: {recall_cnn:.4f}")
print(f"Precision: {precision_cnn:.4f}")
print(f"F1 Score: {f1_cnn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cnn))
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=le_attack_cat.classes_, yticklabels=le_attack_cat.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: CNN')
plt.show()
