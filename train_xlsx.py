"""Use the extended IBM Telco xlsx (33 columns) safely.
1) Converts it to the standard 21-column format (telco_from_xlsx.csv) and verifies it matches telco.csv.
2) Compares feature sets with 5-fold CV: clean, clean+location, and LEAKY variants (demo only)."""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SEED = 42
src = sys.argv[1] if len(sys.argv) > 1 else "Telco_customer_churn.xlsx"
x = pd.read_excel(src)

rename = {"CustomerID": "customerID", "Gender": "gender", "Senior Citizen": "SeniorCitizen", "Tenure Months": "tenure",
          "Phone Service": "PhoneService", "Multiple Lines": "MultipleLines", "Internet Service": "InternetService",
          "Online Security": "OnlineSecurity", "Online Backup": "OnlineBackup", "Device Protection": "DeviceProtection",
          "Tech Support": "TechSupport", "Streaming TV": "StreamingTV", "Streaming Movies": "StreamingMovies",
          "Paperless Billing": "PaperlessBilling", "Payment Method": "PaymentMethod",
          "Monthly Charges": "MonthlyCharges", "Total Charges": "TotalCharges", "Churn Label": "Churn"}
d = x.rename(columns=rename)
d["SeniorCitizen"] = (d["SeniorCitizen"] == "Yes").astype(int)

# ---- 1) Standard 21-column file ----
std_cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
d[std_cols].to_csv("telco_from_xlsx.csv", index=False)
print(f"Saved telco_from_xlsx.csv {d[std_cols].shape}")
if os.path.exists("telco.csv"):
    o = pd.read_csv("telco.csv").merge(d[std_cols], on="customerID", suffixes=("_o", "_x"))
    bad = [c for c in std_cols[1:] if (o[c + "_o"].astype(str).str.strip() != o[c + "_x"].astype(str).str.strip()).any()]
    print("Matches telco.csv on all 20 columns:" , "YES" if not bad and len(o) == len(d) else f"NO, differs in {bad}")

# ---- 2) Feature-set comparison (5-fold CV AUC) ----
d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
d = d.dropna(subset=["TotalCharges"]).reset_index(drop=True)
y = (d["Churn"] == "Yes").astype(int)
d["ChurnReasonKnown"] = d["Churn Reason"].notna().astype(int)      # leaky: exists only for churners
base = [c for c in std_cols if c not in ("customerID", "Churn")]
sets = {
    "Clean 19 features (what we use)": base,
    "Clean + Latitude/Longitude": base + ["Latitude", "Longitude"],
    "Clean + CLTV": base + ["CLTV"],
    "LEAKY: + Churn Score": base + ["Churn Score"],
    "LEAKY: + Churn Reason (known?)": base + ["ChurnReasonKnown"],
}
spw = float((y == 0).sum() / (y == 1).sum())
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def pipe(cols, kind):
    num = d[cols].select_dtypes("number").columns.tolist()
    cat = [c for c in cols if c not in num]
    pre = ColumnTransformer([("n", StandardScaler() if kind == "lr" else "passthrough", num),
                             ("c", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)])
    m = (LogisticRegression(C=10, max_iter=3000, class_weight="balanced") if kind == "lr" else
         XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=SEED,
                       eval_metric="logloss", scale_pos_weight=spw, n_jobs=-1))
    return Pipeline([("p", pre), ("m", m)])

print(f"\n5-fold CV ROC-AUC on {len(d)} customers (mean ± std):")
print(f"{'Feature set':34s} {'XGBoost':>16s} {'LogReg':>16s}")
for name, cols in sets.items():
    r = {}
    for kind in ("xgb", "lr"):
        s = cross_val_score(pipe(cols, kind), d[cols], y, cv=cv, scoring="roc_auc")
        r[kind] = f"{s.mean():.4f} ± {s.std():.4f}"
    print(f"{name:34s} {r['xgb']:>16s} {r['lr']:>16s}")