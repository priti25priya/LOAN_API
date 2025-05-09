from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

# ✅ Initialize FastAPI instance
app = FastAPI()

# ✅ Model and Scaler Paths
SCALER_PATH = "C:/Users/Priti Priya/Downloads/loan_api/scaler.pkl"
MODEL_PATH = "C:/Users/Priti Priya/Downloads/loan_api/loan_model.pkl"

# ✅ Function to load model & scaler safely
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"❌ Error: {path} not found! Ensure the correct file path.")

try:
    scaler = load_model(SCALER_PATH)
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"❌ Critical: {str(e)}")

# ✅ Define Pydantic model for request validation
class LoanData(BaseModel):
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_term: float
    credit_history: float

@app.post("/predict")
def predict_loan_status(data: LoanData):
    """Predicts loan approval status based on given financial details"""
    try:
        # ✅ Convert input data into a NumPy array
        loan_features = np.array([
            data.applicant_income, 
            data.coapplicant_income, 
            data.loan_amount, 
            data.loan_term, 
            data.credit_history
        ]).reshape(1, -1)

        # ✅ Scale input before prediction
        loan_scaled = scaler.transform(loan_features)
        loan_prediction = model.predict(loan_scaled)[0]
        loan_status = "Approved ✅" if loan_prediction == 1 else "Rejected ❌"

        return {"Loan Status Prediction": loan_status}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error during prediction: {str(e)}")

@app.get("/")
def home():
    return {"message": "Loan Approval Prediction API is running!"}

print("✅ API setup complete! Ready for loan predictions.")
