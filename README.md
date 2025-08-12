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

├── loan_api.py # FastAPI app to serve predictions
├── preprocess.py # Data cleaning and feature engineering
├── train_model.py # Model training and saving
├── test_api.py # Script to test the API 
├── loan_model.pkl # Trained XGBoost model 
├── scaler.pkl # Scaler for input normalization 
├── README.md # Project documentation


---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/priti25priya/loan-prediction-system.git
cd loan-prediction-system

### 2. Install Dependencies
pip install fastapi uvicorn xgboost pandas scikit-learn joblib

### 3. Start the FastAPI Server
uvicorn loan_api:app --reload

### 4. API Testing with Postman

<img width="761" height="338" alt="image" src="https://github.com/user-attachments/assets/b728d018-4f99-4e71-bd7f-cfce06829ebe" />





