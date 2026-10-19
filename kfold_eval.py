"""Nested stratified K-fold on ALL data.
Outer 5 folds = honest performance estimate. Inner 3-fold GridSearchCV = hyperparameter tuning (train folds only).
Also scores untuned default models on the same outer folds to measure how much tuning really helps."""
import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

SEED, K = 42, 5
df = pd.read_csv("telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna().drop(columns="customerID")
y = (df["Churn"] == "Yes").astype(int).values
X = df.drop(columns="Churn")
num = X.select_dtypes("number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()

def pre(scale):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num),
                              ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)])

def make(name, spw):
    return {
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
                          {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [5, 20]}, False),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
                          {"m__n_estimators": [300], "m__max_depth": [8, 12], "m__min_samples_leaf": [3, 8]}, False),
        "XGBoost": (XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, scale_pos_weight=spw),
                    {"m__n_estimators": [150, 300], "m__max_depth": [3, 4], "m__learning_rate": [0.03, 0.08],
                     "m__subsample": [0.8, 1.0]}, False),
        "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                                {"m__C": [0.01, 0.1, 1, 10]}, True),
    }[name]

def score(yt, p):
    pr = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(yt, pr), precision=precision_score(yt, pr, zero_division=0),
                recall=recall_score(yt, pr), f1=f1_score(yt, pr), roc_auc=roc_auc_score(yt, p))

names = ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"]
rows = []
outer = StratifiedKFold(K, shuffle=True, random_state=SEED)
for fold, (tr, te) in enumerate(outer.split(X, y), 1):
    t0 = time.time()
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    for n in names:
        est, grid, scale = make(n, spw)
        gs = GridSearchCV(Pipeline([("p", pre(scale)), ("m", est)]), grid, cv=3, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
        rows.append(dict(fold=fold, model=n, kind="tuned", **score(yte, gs.predict_proba(Xte)[:, 1]),
                         params=str({k[3:]: v for k, v in gs.best_params_.items()})))
        base = Pipeline([("p", pre(scale)), ("m", make(n, spw)[0])]).fit(Xtr, ytr)   # library defaults
        rows.append(dict(fold=fold, model=n, kind="default", **score(yte, base.predict_proba(Xte)[:, 1]), params=""))
    print(f"fold {fold}/{K} done in {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rows)
R.to_csv("kfold_folds.csv", index=False)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
summary = {}
print("\n=== Nested 5-fold results (mean ± std over outer folds, threshold 0.5) ===")
for kind in ["tuned", "default"]:
    print(f"\n[{kind}]")
    for n in names:
        s = R[(R.model == n) & (R.kind == kind)]
        summary.setdefault(kind, {})[n] = {m: [float(s[m].mean()), float(s[m].std())] for m in metrics}
        print(f"  {n:20s} " + " | ".join(f"{m[:4]} {s[m].mean():.3f}±{s[m].std():.3f}" for m in metrics))
print("\nTuning gain in AUC (tuned - default), mean over folds, and folds where tuned won:")
gain = {}
for n in names:
    a = R[(R.model == n) & (R.kind == "tuned")].roc_auc.values
    b = R[(R.model == n) & (R.kind == "default")].roc_auc.values
    gain[n] = dict(mean_gain=float((a - b).mean()), folds_won=int((a > b).sum()))
    print(f"  {n:20s} {gain[n]['mean_gain']:+.4f}  ({gain[n]['folds_won']}/{K} folds)")
print("\nBest params chosen per outer fold (stability):")
for n in names:
    print(" ", n); [print("    fold", r.fold, r.params) for r in R[(R.model == n) & (R.kind == "tuned")].itertuples()]
json.dump({"summary": summary, "tuning_gain": gain, "k": K}, open("kfold_results.json", "w"), indent=2)
print("\nSaved kfold_results.json, kfold_folds.csv")