from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# ✅ Sample training data
X_train = np.array([[5000, 35, 2], [3000, 42, 1], [7000, 29, 3]])
y_train = np.array([1, 0, 1])  # Approved (1) or Rejected (0)

# ✅ Train & Save Scaler
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "C:/Users/Priti Priya/Downloads/loan_api/scaler.pkl")

# ✅ Train & Save Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "C:/Users/Priti Priya/Downloads/loan_api/loan_model.pkl")

print("✅ Scaler and model saved successfully!")
