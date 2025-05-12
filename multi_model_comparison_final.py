
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# Veri yükle
df = pd.read_excel("Smart_Hydrogel_Expanded_Dataset_Expanded.xlsx")
df.columns = df.columns.str.strip().str.lower()

X = df[['ph', 'swelling ratio (g/g)', 'diffusion coefficient (cm²/s)']]
y = df['drug release (%)']

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Veriyi ölçekle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelleri tanımla
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost (Light)": xgb.XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, subsample=0.8, objective='reg:squarederror', random_state=42),
    "XGBoost (Strong)": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=1.0, objective='reg:squarederror', random_state=42)
}

# Sonuçları buraya yazacağız
results = []

# Her modeli sırayla eğit
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    preds = np.clip(preds, 0, 100)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append({
        "Model": name,
        "MAE (Loss)": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Accuracy (R²)": f"{r2:.4f} ({r2 * 100:.2f}%)"
    })

# Tabloyu yazdır
df_results = pd.DataFrame(results)

print("\nFinal Comparison Table:")
print(df_results.to_markdown(index=False))

# Grafikle görselleştir

# Grafikle görselleştir
plt.figure(figsize=(10, 4))
accuracies = [float(r.split()[0]) * 100 for r in df_results["Accuracy (R²)"]]
plt.bar(df_results["Model"], accuracies, color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.title("Model Accuracy Comparison")
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Her approach için sonucu açık açık yazdır
print("\nModel-specific Performance Summary:")
for result in results:
    print(f"- {result['Model']}: Accuracy = {result['Accuracy (R²)']}, MAE = {result['MAE (Loss)']}, RMSE = {result['RMSE']}")
