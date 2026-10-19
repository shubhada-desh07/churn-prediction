import joblib, numpy as np, pandas as pd, xgboost as xgb
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

@st.cache_resource
def load():
    return joblib.load("churn_model.joblib")

art = load()
model, num, cat, opts, rng, cols = art["model"], art["num"], art["cat"], art["options"], art["ranges"], art["columns"]
res = art["results"]

st.title("📉 Customer Churn Predictor")
fm = art["final_metrics_at_0.30"]
st.caption(f"Deployed model: **{art['best_name']}**{' (isotonic-calibrated)' if art['calibrated'] else ''} · "
           f"IBM Telco data · test ROC-AUC {res[art['best_name']]['roc_auc']:.3f}")

threshold = st.sidebar.slider("Flag as 'at risk' above", 0.05, 0.95, 0.30, 0.05,
    help="Lower = catch more churners, but flag more customers who would have stayed. See the trade-off table in 'Model report'.")
st.sidebar.markdown("Probabilities are estimates from one static dataset. Validate on your own data before real use.")

def band(p):
    return "🔴 High" if p >= max(threshold + 0.2, 0.6) else ("🟠 At risk" if p >= threshold else "🟢 Low")

def explain(row_df):
    """Per-customer log-odds contributions from the XGBoost component (pred_contribs)."""
    pipe = art["xgb_explainer"]
    p, booster = pipe.named_steps["p"], pipe.named_steps["m"].get_booster()
    Xt = p.transform(row_df[cols])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    c = booster.predict(xgb.DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True)[0][:-1]
    names = [n.split("__", 1)[1] for n in p.get_feature_names_out()]
    return pd.Series(c, index=names).sort_values()

tab1, tab2, tab3, tab4 = st.tabs(["Single customer", "Batch scoring (CSV)", "Model report", "Paper gaps vs our fixes"])

# ---------------- Single customer ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    groups, row = [c1, c2, c3], {}
    for i, c in enumerate(cat):
        row[c] = groups[i % 3].selectbox(c, opts[c], key=c)
    for i, c in enumerate(num):
        lo, hi, med = rng[c]
        if c == "SeniorCitizen":
            row[c] = groups[i % 3].selectbox(c, [0, 1], key=c)
        elif c == "tenure":
            row[c] = groups[i % 3].slider("tenure (months)", int(lo), int(hi), int(med), key=c)
        else:
            row[c] = groups[i % 3].number_input(c, float(lo), float(hi) * 1.5, float(med), key=c)
    if st.button("Predict churn risk", type="primary"):
        d = pd.DataFrame([row])[cols]
        p = float(model.predict_proba(d)[0, 1])
        m1, m2 = st.columns([1, 2])
        m1.metric("Churn probability", f"{p:.1%}")
        m1.markdown(f"### {band(p)}")
        m1.progress(min(p, 1.0))
        s = explain(d)
        top = pd.concat([s.tail(4), s.head(3)]).sort_values()
        m2.write("**Why (log-odds contribution; positive = pushes toward churn):**")
        m2.bar_chart(top, horizontal=True)
        m2.caption("Explanation comes from the XGBoost component of the system; "
                   "the probability shown comes from the deployed (calibrated) model.")

# ---------------- Batch ----------------
with tab2:
    st.write("Upload a CSV with the training columns (`customerID` and `Churn` optional).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        d = pd.read_csv(up)
        d["TotalCharges"] = pd.to_numeric(d.get("TotalCharges"), errors="coerce")
        miss = [c for c in cols if c not in d.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            d = d.dropna(subset=cols).copy()
            d["churn_probability"] = model.predict_proba(d[cols])[:, 1]
            d["risk"] = d["churn_probability"].apply(band)
            d = d.sort_values("churn_probability", ascending=False)
            k = int((d["churn_probability"] >= threshold).sum())
            st.success(f"{len(d)} customers scored · {k} flagged at or above {threshold:.0%}")
            show = [c for c in ["customerID", "tenure", "Contract", "MonthlyCharges"] if c in d.columns]
            st.dataframe(d[show + ["churn_probability", "risk"]], height=380)
            st.download_button("Download scored CSV", d.to_csv(index=False), "scored_customers.csv")

# ---------------- Model report ----------------
with tab3:
    r = art["rows"]
    st.caption(f"Data: {r['raw']} raw rows → {r['used']} used (only rows with invalid TotalCharges dropped) · "
               f"train {r['train']} / test {r['test']} (stratified 80/20)")
    st.subheader("1. Model comparison (test set, threshold 0.5)")
    t = pd.DataFrame(res).T
    t["CV AUC (mean ± std)"] = t.apply(lambda x: f"{x.cv_auc:.3f} ± {x.cv_std:.3f}", axis=1)
    t["Test AUC [95% CI]"] = t.apply(lambda x: f"{x.roc_auc:.3f} [{x.auc_ci[0]:.3f}, {x.auc_ci[1]:.3f}]", axis=1)
    t["p vs best (bootstrap)"] = t["p_vs_best"].apply(lambda v: "—" if v is None or pd.isna(v) else f"{v:.3f}")
    st.dataframe(t[["accuracy", "precision", "recall", "f1", "CV AUC (mean ± std)", "Test AUC [95% CI]",
                    "p vs best (bootstrap)"]].style.format({c: "{:.3f}" for c in ["accuracy", "precision", "recall", "f1"]}))
    st.caption("A large p-value means the model is statistically indistinguishable from the selected one. "
               "Precision/recall are for the churn class. Always predicting 'stays' gives ~73.5% accuracy.")

    st.subheader("2. Confusion matrix")
    pick = st.selectbox("Model", list(res.keys()), index=list(res.keys()).index(art["best_name"]))
    cm = res[pick]["confusion"]
    st.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                 index=["Actual: stays", "Actual: churns"], columns=["Predicted: stays", "Predicted: churns"]))

    st.subheader("3. Imbalance handling study (same Random Forest settings)")
    s = pd.DataFrame(art["study"]).T[["accuracy", "precision", "recall", "roc_auc"]]
    st.dataframe(s.style.format("{:.3f}"))
    st.caption("The LEAKY row oversamples before splitting, so test rows contain synthetic copies of training rows. "
               "It inflates every metric, and its profile resembles the published paper's reported numbers.")

    st.subheader("4. Calibration (does '40%' really mean 40%?)")
    cv_ = art["curves"]
    cdf = pd.DataFrame({"perfect": cv_["calibrated"]["predicted"],
                        "raw model": np.interp(cv_["calibrated"]["predicted"], cv_["raw"]["predicted"], cv_["raw"]["actual"]),
                        "calibrated": cv_["calibrated"]["actual"]}, index=np.round(cv_["calibrated"]["predicted"], 2))
    st.line_chart(cdf)
    st.caption(f"Brier score (lower is better): raw {art['brier']['raw']:.4f} → calibrated {art['brier']['calibrated']:.4f}.")

    st.subheader("5. Threshold trade-off (deployed model, test set)")
    th = pd.DataFrame(art["thresholds"]).set_index("threshold")
    th["flagged_pct"] = th["flagged_pct"] * 100
    st.dataframe(th.rename(columns={"flagged_pct": "% customers flagged"}).style.format("{:.3f}"))

    st.subheader("6. Global feature importance (AUC drop when shuffled)")
    st.bar_chart(pd.Series(art["importance"]).head(10)[::-1], horizontal=True)

    import json, os
    if os.path.exists("kfold_results.json"):
        kf = json.load(open("kfold_results.json"))
        st.subheader(f"7. Nested {kf['k']}-fold cross-validation (all data, tuning inside each training fold)")
        for kind, label in [("tuned", "Tuned (grid search in inner loop)"), ("default", "Library defaults")]:
            tbl = pd.DataFrame(
                {a: {m: f"{v[0]:.3f} ± {v[1]:.3f}" for m, v in met.items()} for a, met in kf["summary"][kind].items()}).T
            st.markdown(f"**{label}**"); st.dataframe(tbl)
        st.markdown("**Tuning gain (AUC, tuned − default)**")
        st.dataframe(pd.DataFrame(kf["tuning_gain"]).T.rename(columns={"mean_gain": "mean AUC gain", "folds_won": "folds tuned won"}))
        st.caption("Mean ± std across outer folds at threshold 0.5 (class-weighted, uncalibrated), so use AUC for comparisons.")

# ---------------- Viva tab ----------------
with tab4:
    st.markdown("""
### Weaknesses in the reference paper, and what this system does
| Paper weakness | Our handling |
|---|---|
| 5,634 rows claimed after cleaning, but only 11 rows are invalid (7,043 → 7,032) | Row counts reported at every step (see Model report) |
| Confusion matrices belong to another dataset (labelled Malignant/Benign, 114 samples) | Real matrices from the 1,407-customer test set, per model |
| Reported scores (84.7% acc, 0.939 AUC) not reproducible with an honest pipeline | Leak-free sklearn Pipelines; honest result ≈ 80% accuracy, ≈ 0.84 AUC |
| Likely oversampling before splitting (scores match our leaky demo) | Demonstrated the leak side by side; SMOTE only inside CV folds |
| No significance testing; ranked models 0.3 points apart | Bootstrap 95% CIs and paired p-values |
| Only single-split scores | 5-fold CV mean ± std for every model |
| Class imbalance addressed by metrics only | Class weights / scale_pos_weight inside every model |
| Soft-voting ensemble claimed but never reported | Built and evaluated (no significant gain over XGBoost) |
| Fixed 0.5 threshold, no probability quality check | Isotonic calibration (Brier) + threshold trade-off table + slider |
| Black-box ensembles, explainability left as future work | Per-customer explanations (XGBoost contributions) + global importance |
| No deployable output | Streamlit app: single, batch scoring, downloadable results |

### Limitations we did NOT fix (say these honestly)
- Dataset has no timestamps, so **temporal validation / concept drift cannot be tested**.
- **No external dataset** validation; results hold for IBM Telco only.
- One train/test split (CIs from bootstrap, not repeated splits).
- Features are static account attributes; no usage, complaints or network-quality data.
- Per-customer explanations come from the XGBoost component, not the calibrated wrapper.
- Predictions show association, not that an intervention would retain a customer.
""")