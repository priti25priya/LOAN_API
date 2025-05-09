import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "applicant_income": 5000,
    "coapplicant_income": 0,
    "loan_amount": 128,
    "loan_term": 360,
    "credit_history": 1
}

response = requests.post(url, json=data)
print(response.json())
