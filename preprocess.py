from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Data Collection and Processing
# loading the dataset to pandas DataFrame
df = pd.read_csv(r"C:\Users\Priti Priya\Downloads\loan.csv.csv")

# Preview data
print(df.head())

# Check missing values
print(df.isnull().sum())

# Handle missing values
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)

# Convert "3+" in Dependents to numerical format
if "Dependents" in df.columns:
    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)

# Check which categorical columns exist before encoding
categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
existing_cols = [col for col in categorical_cols if col in df.columns]

# Convert categorical variables into numerical format only if they exist
df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

# Drop Loan_ID column as it's not useful for prediction
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

print("Data processed successfully!")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score

# Prepare features and target variable
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"].map({"Y": 1, "N": 0})  # Convert labels into numerical format

# Scale feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Train XGBoost model
model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate model accuracy
predictions = model.predict(X_test)
print("XGBoost Model Accuracy:", accuracy_score(y_test, predictions))
