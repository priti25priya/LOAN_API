# 🏦 Loan Prediction System

A machine learning-powered API that predicts loan approval based on applicant data. Built using **FastAPI**, **XGBoost**, and **scikit-learn**, this project demonstrates end-to-end deployment of a predictive model.

---

## 🚀 Features

- FastAPI-based RESTful API for loan prediction
- XGBoost model trained on structured applicant data
- Scaler normalization for consistent input handling
- Modular codebase with preprocessing and training scripts
- Postman support for API testing

---

## 📁 Project Structure

loan_api.py – FastAPI app that serves loan approval predictions

preprocess.py – Handles data cleaning and feature engineering

train_model.py – Trains the XGBoost model and saves it using Joblib

test_api.py – Script to test the API with sample inputs

loan_model.pkl – Serialized XGBoost model for prediction

scaler.pkl – Scaler object used to normalize input features

README.md – Project documentation and usage instructions

---

# 🏦 Loan Prediction System

A FastAPI-based machine learning API that predicts loan approval based on applicant details. Built using XGBoost, pandas, and scikit-learn.

---

## 🚀 Getting Started

### 📁 **Clone the Repository**

1. git clone https://github.com/priti25priya/LOAN_API.git
2. cd loan-prediction-system


---

### 📦 **Install Dependencies**

pip install fastapi uvicorn xgboost pandas scikit-learn joblib

---

### 🧠 **Train the Model**

python train_model.py
This will create:
1. loan_model.pkl - Trained XGBoost model
2. scaler.pkl - Scaler for input normalization

---

### 🚀 **Launch the API**

uvicorn loan_api:app --reload

---

### 🧪 **Test the API with Postman**

Endpoint: POST http://127.0.0.1:8000/predict
