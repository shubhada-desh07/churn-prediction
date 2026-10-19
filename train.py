"""Churn pipeline v2: paper's method + fixes for the paper's weaknesses.
Fixes: imbalance handling, CV mean/std, bootstrap CIs + paired tests, in-fold SMOTE study,
leakage demo, soft-voting ensemble, probability calibration, confusion matrices, XGB explainer."""
import json, time, joblib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
df = pd.read_csv("telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_raw = len(df)
df = df.dropna().drop(columns="customerID")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns="Churn")
num = X.select_dtypes("number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"raw {n_raw} -> {len(df)} rows (dropped {n_raw-len(df)}) | train {len(X_tr)} | test {len(X_te)}")
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)

def pre(scale=False):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)])

def metrics(yt, p, thr=0.5):
    pr = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pr).ravel()
    return dict(accuracy=accuracy_score(yt, pr), precision=precision_score(yt, pr, zero_division=0),
                recall=recall_score(yt, pr), f1=f1_score(yt, pr), roc_auc=roc_auc_score(yt, p),
                brier=brier_score_loss(yt, p), confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))

spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
candidates = {   # imbalance handled in every model (class weights), no resampling
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
                      {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [5, 20]}, False),
    "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
                      {"m__n_estimators": [300], "m__max_depth": [8, 12], "m__min_samples_leaf": [3, 8]}, False),
    "XGBoost": (XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, scale_pos_weight=spw),
                {"m__n_estimators": [150, 300], "m__max_depth": [3, 4], "m__learning_rate": [0.03, 0.08],
                 "m__subsample": [0.8, 1.0]}, False),
    "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                            {"m__C": [0.01, 0.1, 1, 10]}, True),
}

fitted, results, probs = {}, {}, {}
for name, (est, grid, scale) in candidates.items():
    t0 = time.time()
    gs = GridSearchCV(Pipeline([("p", pre(scale)), ("m", est)]), grid, cv=cv5, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
    i = gs.best_index_
    fitted[name] = gs.best_estimator_
    probs[name] = gs.predict_proba(X_te)[:, 1]
    results[name] = dict(metrics(y_te, probs[name]), cv_auc=float(gs.cv_results_["mean_test_score"][i]),
                         cv_std=float(gs.cv_results_["std_test_score"][i]), train_seconds=time.time() - t0,
                         best_params={k[3:]: v for k, v in gs.best_params_.items()})
    print(f"{name:20s} CV AUC {results[name]['cv_auc']:.4f} +/- {results[name]['cv_std']:.4f} | test AUC {results[name]['roc_auc']:.4f}")

# ---- Soft-voting ensemble (the paper mentions one but never reports it) ----
t0 = time.time()
ens = VotingClassifier([(n, fitted[n]) for n in candidates], voting="soft")
cvs = cross_val_score(ens, X_tr, y_tr, cv=cv5, scoring="roc_auc", n_jobs=1)
ens.fit(X_tr, y_tr)
fitted["Soft-Voting Ensemble"] = ens
probs["Soft-Voting Ensemble"] = ens.predict_proba(X_te)[:, 1]
results["Soft-Voting Ensemble"] = dict(metrics(y_te, probs["Soft-Voting Ensemble"]), cv_auc=float(cvs.mean()),
                                       cv_std=float(cvs.std()), train_seconds=time.time() - t0, best_params={})
print(f"{'Soft-Voting Ensemble':20s} CV AUC {cvs.mean():.4f} +/- {cvs.std():.4f} | test AUC {results['Soft-Voting Ensemble']['roc_auc']:.4f}")

# ---- Bootstrap CIs and paired significance test on test AUC ----
rng = np.random.default_rng(SEED)
yv = y_te.values
B = 1000
idx = [rng.integers(0, len(yv), len(yv)) for _ in range(B)]
boot = {n: np.array([roc_auc_score(yv[i], probs[n][i]) for i in idx]) for n in probs}
best = max(results, key=lambda k: results[k]["cv_auc"])      # selected on CV, never on test
for n in results:
    lo, hi = np.percentile(boot[n], [2.5, 97.5])
    results[n]["auc_ci"] = [float(lo), float(hi)]
    d = boot[best] - boot[n]
    results[n]["p_vs_best"] = None if n == best else float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
print(f"\nSelected on CV: {best}")
for n in results:
    r = results[n]
    print(f"  {n:20s} AUC {r['roc_auc']:.3f} CI [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  p vs best: {r['p_vs_best']}")

# ---- Imbalance study: RF with none / class-weight / SMOTE inside folds / SMOTE BEFORE split (leaky) ----
rfp = {k: v for k, v in results["Random Forest"]["best_params"].items()}
def rf(cw=None): return RandomForestClassifier(**rfp, random_state=SEED, n_jobs=-1, class_weight=cw)
study = {}
def run(name, model, Xa=X_tr, ya=y_tr, Xb=X_te, yb=y_te, cv=True):
    s = cross_val_score(model, Xa, ya, cv=cv5, scoring="roc_auc").mean() if cv else None
    model.fit(Xa, ya); m = metrics(yb, model.predict_proba(Xb)[:, 1])
    m["cv_auc"] = None if s is None else float(s); study[name] = m
    print(f"  {name:38s} acc {m['accuracy']:.3f} R {m['recall']:.3f} P {m['precision']:.3f} AUC {m['roc_auc']:.3f}")
print("\nImbalance study (same RF settings):")
run("No imbalance handling", Pipeline([("p", pre()), ("m", rf(None))]))
run("Class weights (our choice)", Pipeline([("p", pre()), ("m", rf("balanced_subsample"))]))
run("SMOTE inside CV folds (correct)", ImbPipeline([("p", pre()), ("s", SMOTE(random_state=SEED)), ("m", rf(None))]))
Xall = pre().fit_transform(X); Xall = Xall.toarray() if hasattr(Xall, "toarray") else Xall
Xs, ys = SMOTE(random_state=SEED).fit_resample(Xall, y)
Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=SEED)
run("SMOTE BEFORE split (LEAKY, demo only)", rf(None), Xa, ya, Xb, yb, cv=False)

# ---- Calibration of the selected model ----
raw_best = fitted[best]
p_raw = raw_best.predict_proba(X_te)[:, 1]
cal = CalibratedClassifierCV(raw_best, method="isotonic", cv=5).fit(X_tr, y_tr)
p_cal = cal.predict_proba(X_te)[:, 1]
b_raw, b_cal = brier_score_loss(yv, p_raw), brier_score_loss(yv, p_cal)
use_cal = b_cal < b_raw
final, p_final = (cal, p_cal) if use_cal else (raw_best, p_raw)
print(f"\nBrier score raw {b_raw:.4f} vs calibrated {b_cal:.4f} -> using {'calibrated' if use_cal else 'raw'}")
curves = {}
for tag, p in [("raw", p_raw), ("calibrated", p_cal)]:
    fr, mp = calibration_curve(yv, p, n_bins=8, strategy="quantile")
    curves[tag] = {"predicted": mp.tolist(), "actual": fr.tolist()}

thr_table = []
for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    m = metrics(yv, p_final, t)
    thr_table.append(dict(threshold=t, precision=m["precision"], recall=m["recall"], f1=m["f1"],
                          flagged_pct=float((p_final >= t).mean())))
final_metrics = metrics(yv, p_final, 0.30)

imp = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
importance = {c: float(v) for c, v in sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])}

joblib.dump({
    "model": final, "best_name": best, "calibrated": bool(use_cal), "results": results, "study": study,
    "curves": curves, "brier": {"raw": b_raw, "calibrated": b_cal}, "thresholds": thr_table,
    "final_metrics_at_0.30": final_metrics, "importance": importance, "xgb_explainer": fitted["XGBoost"],
    "columns": X.columns.tolist(), "num": num, "cat": cat, "rows": dict(raw=n_raw, used=len(df), train=len(X_tr), test=len(X_te)),
    "options": {c: sorted(df[c].unique().tolist()) for c in cat},
    "ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in num},
}, "churn_model.joblib")
json.dump({"results": results, "study": study, "thresholds": thr_table}, open("metrics.json", "w"), indent=2, default=float)
print("Saved churn_model.joblib, metrics.json")
