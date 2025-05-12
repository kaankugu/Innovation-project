
import joblib
import numpy as np

# Load the pre-trained XGBoost model
model = joblib.load("best_xgboost_model.pkl")

# Example input data: [pH, Swelling Ratio (g/g), Diffusion Coefficient (cm²/s)]
sample_input = np.array([[7.4, 1.8, 0.00000019]])

# Make prediction and clip the result to 0–100%
predicted_release = model.predict(sample_input)
predicted_release = np.clip(predicted_release, 0, 100)

# Display the result
print(f"Predicted Drug Release Percentage (clipped): {predicted_release[0]:.2f}%")
