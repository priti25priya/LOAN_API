import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ Sample training data
X_train = np.array([
    [5849, 0, 128, 360, 1], 
    [4583, 1508, 66, 360, 1], 
    [3000, 0, 120, 360, 1], 
    [2583, 2358, 141, 360, 1], 
    [6000, 0, 90, 180, 0]
])  # Columns: [ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History]

y_train = np.array([1, 0, 1, 1, 1])  # Loan approved (1) or rejected (0)

# ✅ Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ✅ Define class to store model objects
class LoanApprovalModel:
    def __init__(self, model_name, model, scaler):
        self.model_name = model_name
        self.model = model
        self.scaler = scaler
    
    def predict(self, X_new):
        X_transformed = self.scaler.transform(X_new)
        return self.model.predict(X_transformed)

# ✅ Train model and store as an object
model_list = []

rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Store trained model in the object list
model_list.append(LoanApprovalModel("RandomForest", rf_model, scaler))

# ✅ Save trained model & scaler
joblib.dump(rf_model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained successfully and stored as an object!")
