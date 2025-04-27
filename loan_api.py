from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# ✅ Ensure the correct paths for your model and scaler
scaler_path = "C:/Users/Priti Priya/Downloads/loan_api/scaler.pkl"
model_path = "C:/Users/Priti Priya/Downloads/loan_api/loan_model.pkl"

if os.path.exists(scaler_path) and os.path.exists(model_path):
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
else:
    raise FileNotFoundError("Scaler or model file not found. Ensure the correct path.")

# ✅ Define request model
class LoanData(BaseModel):
    features: list

@app.post("/predict")
def predict_loan_status(data: LoanData):
    try:
        loan_features = np.array(data.features).reshape(1, -1)

        # ✅ Scale input before prediction
        loan_scaled = scaler.transform(loan_features)
        loan_prediction = model.predict(loan_scaled)[0]
        loan_status = "Approved" if loan_prediction == 1 else "Rejected"

        return {"Loan Status Prediction": loan_status}
    except Exception as e:
        return {"error": str(e)}
