"""
Train a final Telco Customer Churn model and save it as churn_model.joblib.

Run:
    python train_final_model.py
or:
    python train_final_model.py "Telco_customer_churn(1).xlsx"
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Use a filename supplied on the command line, or the default filename.
src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Telco_customer_churn.xlsx")

if not src.exists():
    raise FileNotFoundError(
        f"Dataset not found: {src.resolve()}\n"
        "Put the Excel file in this folder or pass its full path, e.g.\n"
        'python train_final_model.py "Telco_customer_churn(1).xlsx"'
    )

df = pd.read_excel(src)

# Convert the extended IBM Telco column names to the standard names.
rename = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
}
df = df.rename(columns=rename)

# Check that the expected target and feature columns exist.
required = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Dataset is missing expected columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Match the standard feature set used in the evaluation scripts.
df["SeniorCitizen"] = (
    df["SeniorCitizen"].astype(str).str.strip().str.lower()
    .map({"yes": 1, "no": 0, "1": 1, "0": 0})
)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

feature_cols = [c for c in required if c != "Churn"]
df = df.dropna(subset=feature_cols + ["Churn"]).copy()

# Standardize target labels to 0 = No churn, 1 = Churn.
target = df["Churn"].astype(str).str.strip().str.lower()
valid = target.isin(["yes", "no"])
df = df.loc[valid].copy()
target = target.loc[valid]
y = target.eq("yes").astype(int)
X = df[feature_cols].copy()

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Train the final model on all cleaned rows.
# scale_pos_weight compensates for the smaller churn class.
positive = int(y.sum())
negative = int((y == 0).sum())
if positive == 0 or negative == 0:
    raise ValueError("Target must contain both churn and non-churn customers.")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight=negative / positive,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X, y)

output = Path("churn_model.joblib")
joblib.dump(model, output)

print(f"Dataset used: {src.resolve()}")
print(f"Training rows: {len(X)}")
print(f"Features: {len(feature_cols)}")
print(f"Churn cases: {positive}; non-churn cases: {negative}")
print(f"Saved trained pipeline to: {output.resolve()}")
print("The saved file includes preprocessing and the trained XGBoost model.")
