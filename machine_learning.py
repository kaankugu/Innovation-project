import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Veri Yükleme
file_path = "Smart_Hydrogel_Expanded_Dataset_Expanded.xlsx"
df = pd.read_excel(file_path)

# 2. Sütun isimlerini kontrol et ve düzenle
df.columns = df.columns.str.strip()  # Boşlukları temizle
df.columns = df.columns.str.lower()  # Büyük harfleri küçük yap

# 3. Özelliklerin Seçimi
# İlaç Salınımı oranını tahmin etmek için pH, Şişme oranı, Diffüzyon katsayısı gibi bağımsız değişkenleri seçeceğiz
X = df[['ph', 'swelling ratio (g/g)', 'diffusion coefficient (cm²/s)']]  # Bu sütunları kontrol et
y = df['drug release (%)']  # Hedef değişken: İlaç salınım oranı

# 4. Veriyi Eğitim ve Test Setlerine Ayırma (80-20 oranında)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Veriyi Ölçekleme (özellikle regresyon için önemli)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Eğitim verisini ölçeklendir
X_test = scaler.transform(X_test)  # Test verisini aynı şekilde ölçeklendir

# ------------------------------------------
# Random Forest Hiperparametre Optimizasyonu
# ------------------------------------------

# Random Forest için hiperparametre ızgarası
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest modeli ve GridSearchCV
rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                              param_grid=rf_param_grid, 
                              cv=3, 
                              scoring='neg_mean_squared_error', 
                              verbose=2, 
                              n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdır
print(f"Best Random Forest Hyperparameters: {rf_grid_search.best_params_}")

# En iyi model ile tahmin yap
rf_best_model = rf_grid_search.best_estimator_
rf_best_predictions = rf_best_model.predict(X_test)

# Performansı değerlendir
rf_best_mae = mean_absolute_error(y_test, rf_best_predictions)
rf_best_rmse = np.sqrt(mean_squared_error(y_test, rf_best_predictions))

print(f"Best Random Forest MAE: {rf_best_mae}")
print(f"Best Random Forest RMSE: {rf_best_rmse}")

# ------------------------------------------
# XGBoost Hiperparametre Optimizasyonu
# ------------------------------------------

# XGBoost için hiperparametre ızgarası
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.7, 0.8, 1.0]
}

# XGBoost modeli ve GridSearchCV
xgb_grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42), 
                               param_grid=xgb_param_grid, 
                               cv=3, 
                               scoring='neg_mean_squared_error', 
                               verbose=2, 
                               n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdır
print(f"Best XGBoost Hyperparameters: {xgb_grid_search.best_params_}")

# En iyi model ile tahmin yap
xgb_best_model = xgb_grid_search.best_estimator_
xgb_best_predictions = xgb_best_model.predict(X_test)

# Performansı değerlendir
xgb_best_mae = mean_absolute_error(y_test, xgb_best_predictions)
xgb_best_rmse = np.sqrt(mean_squared_error(y_test, xgb_best_predictions))

print(f"Best XGBoost MAE: {xgb_best_mae}")
print(f"Best XGBoost RMSE: {xgb_best_rmse}")

# ------------------------------------------
# Sonuçları Görselleştirme
# ------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Gerçek Değerler (Test Seti)', color='blue')
plt.plot(rf_best_predictions, label='Random Forest Tahminleri (Test Seti)', color='red', alpha=0.7)
plt.plot(xgb_best_predictions, label='XGBoost Tahminleri (Test Seti)', color='green', alpha=0.7)
plt.legend()
plt.xlabel('Test Seti Örnekleri')
plt.ylabel('İlaç Salınımı Oranı (%)')
plt.title('Test Seti Performans Karşılaştırması')
plt.grid()
plt.show()

import joblib

# 1. Modeli Kaydetme (XGBoost modeli)
joblib.dump(xgb_best_model, 'best_xgboost_model.pkl')

# 2. Kaydedilen Modeli Yükleme (Modeli yeniden yüklemek için)
loaded_model = joblib.load('best_xgboost_model.pkl')

# 3. Yüklenen model ile tahmin yapma
loaded_predictions = loaded_model.predict(X_test)

# Performansı değerlendirme
loaded_mae = mean_absolute_error(y_test, loaded_predictions)
loaded_rmse = np.sqrt(mean_squared_error(y_test, loaded_predictions))

print(f"Loaded XGBoost MAE: {loaded_mae}")
print(f"Loaded XGBoost RMSE: {loaded_rmse}")
