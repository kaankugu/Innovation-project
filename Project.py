import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

df = pd.read_excel("Smart_Hydrogel_Expanded_Dataset_Expanded.xlsx")


time = df.iloc[:, 0].values
temperature = df.iloc[:, 1].values
pH = df.iloc[:, 2].values
drug_release = df.iloc[:, 3].values
swelling_ratio = df.iloc[:, 4].values
diff_coeff = df.iloc[:, 5].values

L = 0.1 
x = np.linspace(0, L, 100)

def fick_1d_concentration(x, D, t, C0=1.0):
    return C0 * (1 - np.exp(-D * np.pi**2 * t / (L**2)) * np.cos(np.pi * x / L))

plt.figure(figsize=(10, 6))
for i in [0, 15, 20, 35, 40]:
    t = time[i] * 3600 
    D = diff_coeff[i]
    C_profile = fick_1d_concentration(x, D, t)
    plt.plot(x, C_profile * 100, label=f'pH={pH[i]}, t={time[i]}h')

plt.xlabel('Distance (cm)')
plt.ylabel('Drug Concentration (%)')
plt.title('Drug Diffusion Profile in Hydrogel')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(pH, swelling_ratio, c='blue', label='Experimental')
plt.xlabel('pH')
plt.ylabel('Swelling Ratio (g/g)')
plt.title('Swelling Behavior of Hydrogel vs pH')
plt.grid()
plt.legend()
plt.show()


k_deg = 0.05
degradation = np.exp(-k_deg * time)

plt.figure(figsize=(8, 5))
plt.plot(time, degradation, label='Degradation Profile', color='black')
plt.xlabel('Time (h)')
plt.ylabel('Remaining Hydrogel (%)')
plt.title('Hydrogel Degradation Over Time')
plt.grid()
plt.legend()
plt.show()


print(df.head())










