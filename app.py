import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

MODEL_PATH = "churn_model.joblib"

# These must match the feature columns used by train_final_model.py.
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Supports both the standard IBM dataset headers and the extended Excel headers.
RENAME_COLUMNS = {
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

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Do not overwrite an existing standard column if both names are present.
    for old, new in RENAME_COLUMNS.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            df["SeniorCitizen"].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
        )

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize whitespace in text values without changing their meaning.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Normalize common Yes/No capitalization variants.
    yes_no_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda value: value.title() if isinstance(value, str)
                and value.strip().lower() in {"yes", "no"} else value
            )

    if "gender" in df.columns:
        df["gender"] = df["gender"].map(
            lambda value: value.title() if isinstance(value, str)
            and value.strip().lower() in {"female", "male"} else value
        )

    return df

def clean_and_validate_upload(raw_df):
    """Clean recoverable issues, validate values, and return a transparent report."""
    df = normalize_columns(raw_df)
    report = {
        "uploaded": len(df),
        "exact_duplicates": int(df.duplicated().sum()),
        "missing_columns": [col for col in FEATURE_COLS if col not in df.columns],
        "imputed_cells": 0,
        "normalized_cells": 0,
        "invalid_numeric": 0,
        "excluded_rows": 0,
        "scored": 0,
        "changes": [],
    }

    if report["missing_columns"]:
        return df, report

    allowed = {
        "gender": {"Female", "Male"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "MultipleLines": {"No phone service", "No", "Yes"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "OnlineSecurity": {"No internet service", "No", "Yes"},
        "OnlineBackup": {"No internet service", "No", "Yes"},
        "DeviceProtection": {"No internet service", "No", "Yes"},
        "TechSupport": {"No internet service", "No", "Yes"},
        "StreamingTV": {"No internet service", "No", "Yes"},
        "StreamingMovies": {"No internet service", "No", "Yes"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "PaperlessBilling": {"Yes", "No"},
        "PaymentMethod": {
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        },
    }

    # Count text formatting changes, then normalize known categories case-insensitively.
    for col in CATEGORICAL_COLS:
        before = df[col].copy()
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        if col in allowed:
            lower_to_canonical = {value.lower(): value for value in allowed[col]}
            df[col] = df[col].map(
                lambda value: lower_to_canonical.get(value.lower(), value)
                if isinstance(value, str) else value
            )
        report["normalized_cells"] += int(
            ((before != df[col]) & before.notna() & df[col].notna()).sum()
        )

    # Numeric conversion: text that cannot be parsed becomes missing and may be imputed.
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    original_numeric = {}
    for col in numeric_cols:
        original_numeric[col] = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Values outside plausible bounds cannot be safely guessed, so exclude those rows.
    invalid_range = (
        (df["SeniorCitizen"].notna() & ~df["SeniorCitizen"].isin([0, 1]))
        | (df["tenure"].notna() & ~df["tenure"].between(0, 72))
        | (df["MonthlyCharges"].notna() & (df["MonthlyCharges"] < 0))
        | (df["TotalCharges"].notna() & (df["TotalCharges"] < 0))
    )
    # Distinguish missing/uncastable numeric values from impossible numeric values.
    invalid_numeric_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        invalid_numeric_mask |= original_numeric[col].notna() & df[col].isna()
    report["invalid_numeric"] = int(invalid_numeric_mask.sum())

    # Truly unknown non-empty categories are not guessed; those rows are excluded.
    invalid_category = pd.Series(False, index=df.index)
    for col, valid_values in allowed.items():
        invalid_category |= df[col].notna() & ~df[col].isin(valid_values)

    # SeniorCitizen is a binary field: unknown text (e.g. "maybe") is not imputed.
    bad_senior = original_numeric["SeniorCitizen"].notna() & df["SeniorCitizen"].isna()
    invalid_range |= bad_senior

    # Missing values and unparseable numeric text are imputed using the uploaded
    # batch's median (numeric) or mode (categorical), only when a defensible value exists.
    # For reliable production use, fit these imputers on training data and reuse them.
    excluded_mask = invalid_range | invalid_category
    report["excluded_rows"] = int(excluded_mask.sum())
    eligible = df.loc[~excluded_mask].copy()

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        missing = eligible[col].isna()
        if missing.any():
            usable = eligible.loc[~missing, col]
            if not usable.empty:
                fill_value = float(usable.median())
                eligible.loc[missing, col] = fill_value
                report["imputed_cells"] += int(missing.sum())
                report["changes"].append(f"{col}: filled {int(missing.sum())} missing/invalid value(s) with batch median ({fill_value:g})")
            else:
                # No defensible value available for this column.
                eligible = eligible.loc[eligible[col].notna()].copy()

    for col in CATEGORICAL_COLS:
        missing = eligible[col].isna()
        if missing.any():
            modes = eligible[col].mode(dropna=True)
            if not modes.empty:
                fill_value = modes.iloc[0]
                eligible.loc[missing, col] = fill_value
                report["imputed_cells"] += int(missing.sum())
                report["changes"].append(f"{col}: filled {int(missing.sum())} missing value(s) with most common category ({fill_value})")
            else:
                eligible = eligible.loc[eligible[col].notna()].copy()

    # Final validation after cleaning.
    final_bad = eligible[FEATURE_COLS].isna().any(axis=1)
    eligible = eligible.loc[~final_bad].copy()
    report["scored"] = len(eligible)
    report["excluded_rows"] = report["uploaded"] - report["scored"]
    return eligible, report

def risk_band(probability, threshold):
    if probability >= max(threshold + 0.2, 0.6):
        return "🔴 High"
    if probability >= threshold:
        return "🟠 At risk"
    return "🟢 Low"

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Put churn_model.joblib in the same folder "
        "as app.py, then run the app again."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.title("📉 Customer Churn Predictor")
st.caption(
    "Predicts churn probability using the saved XGBoost pipeline. "
    "The pipeline includes preprocessing, so enter the original feature columns."
)

threshold = st.sidebar.slider(
    "Flag as 'at risk' above",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05,
)
st.sidebar.info(
    "Probabilities are estimates based on the training dataset. "
    "Validate the model on new data before using it for business decisions."
)

single_tab, batch_tab, info_tab = st.tabs(
    ["Single customer", "Batch scoring (CSV)", "Model information"]
)

with single_tab:
    st.subheader("Enter customer details")
    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)
        groups = [c1, c2, c3]
        row = {}

        options = {
            "gender": ["Female", "Male"],
            "Partner": ["Yes", "No"],
            "Dependents": ["Yes", "No"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No phone service", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No internet service", "No", "Yes"],
            "OnlineBackup": ["No internet service", "No", "Yes"],
            "DeviceProtection": ["No internet service", "No", "Yes"],
            "TechSupport": ["No internet service", "No", "Yes"],
            "StreamingTV": ["No internet service", "No", "Yes"],
            "StreamingMovies": ["No internet service", "No", "Yes"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }

        for i, col in enumerate(CATEGORICAL_COLS):
            row[col] = groups[i % 3].selectbox(col, options[col], key=f"single_{col}")

        row["SeniorCitizen"] = c1.selectbox(
            "SeniorCitizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        row["tenure"] = c2.slider("tenure (months)", 0, 72, 12)
        row["MonthlyCharges"] = c3.number_input(
            "MonthlyCharges", min_value=0.0, max_value=2000.0, value=70.0, step=1.0
        )
        row["TotalCharges"] = c1.number_input(
            "TotalCharges", min_value=0.0, max_value=100000.0, value=840.0, step=10.0
        )

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        input_df = pd.DataFrame([row])[FEATURE_COLS]
        try:
            probability = float(model.predict_proba(input_df)[0, 1])
            m1, m2 = st.columns([1, 2])
            m1.metric("Churn probability", f"{probability:.1%}")
            m1.markdown(f"### {risk_band(probability, threshold)}")
            m1.progress(min(max(probability, 0.0), 1.0))
            m2.write(
                "This is a model estimate, not a guarantee. The model does not establish "
                "that any particular customer will leave."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with batch_tab:
    st.write(
        "Upload a CSV containing the customer features used during training. "
        "`customerID` and `Churn` are optional. Extra columns are preserved in the "
        "download but are not used by the model."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            cleaned_df, report = clean_and_validate_upload(raw_df)

            st.subheader("Upload and cleaning report")
            a, b, c, d = st.columns(4)
            a.metric("Rows uploaded", report["uploaded"])
            b.metric("Rows scored", report["scored"])
            c.metric("Rows excluded", report["excluded_rows"])
            d.metric("Cells imputed", report["imputed_cells"])

            if report["missing_columns"]:
                st.error(f"Missing required feature columns: {report['missing_columns']}")
            else:
                st.caption(
                    f"Formatting changes made: {report['normalized_cells']}. "
                    f"Exact duplicate rows detected: {report['exact_duplicates']} "
                    "(duplicates are reported, not automatically deleted)."
                )
                if report["invalid_numeric"]:
                    st.warning(
                        f"{report['invalid_numeric']} value(s) could not be parsed as numbers. "
                        "Recoverable missing numeric values may be imputed; impossible values are excluded."
                    )
                if report["changes"]:
                    with st.expander("View cleaning actions"):
                        for change in report["changes"]:
                            st.write(f"• {change}")
                if report["exact_duplicates"]:
                    st.warning(
                        "Exact duplicate rows were retained because identical-looking rows may "
                        "represent separate records. Review them before operational use."
                    )

                if cleaned_df.empty:
                    st.error("No records could be safely cleaned. Review the uploaded values.")
                else:
                    scored = cleaned_df.copy()
                    scored["churn_probability"] = model.predict_proba(
                        scored[FEATURE_COLS]
                    )[:, 1]
                    scored["risk"] = scored["churn_probability"].apply(
                        lambda probability: risk_band(float(probability), threshold)
                    )
                    scored = scored.sort_values("churn_probability", ascending=False)

                    flagged = int((scored["churn_probability"] >= threshold).sum())
                    st.success(
                        f"{len(scored)} customers scored · {flagged} flagged at or above {threshold:.0%}"
                    )

                    st.subheader("Risk overview")
                    risk_counts = (
                        scored["risk"].value_counts()
                        .reindex(["🔴 High", "🟠 At risk", "🟢 Low"], fill_value=0)
                        .rename_axis("Risk level").reset_index(name="Customers")
                    )
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.write("**Customers by risk level**")
                        st.bar_chart(risk_counts.set_index("Risk level"))
                    with chart_col2:
                        st.write("**Churn probability distribution**")
                        prob_counts = scored.assign(
                            probability_band=pd.cut(
                                scored["churn_probability"],
                                bins=[0, .2, .4, .6, .8, 1.000001],
                                labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
                                include_lowest=True,
                            )
                        )["probability_band"].value_counts().sort_index()
                        st.bar_chart(prob_counts)

                    st.subheader("Explore customers by risk")
                    selected_risk = st.selectbox(
                        "Choose a risk group",
                        ["All customers", "🔴 High", "🟠 At risk", "🟢 Low"],
                    )
                    view = scored if selected_risk == "All customers" else scored[scored["risk"] == selected_risk]
                    display_cols = [
                        col for col in
                        ["customerID", "tenure", "Contract", "MonthlyCharges",
                         "TotalCharges", "churn_probability", "risk"]
                        if col in view.columns
                    ]
                    st.dataframe(view[display_cols], use_container_width=True, height=420)

                    st.subheader("Visualize customer-level values")
                    if not view.empty:
                        x_axis = st.selectbox("X-axis", ["tenure", "MonthlyCharges", "TotalCharges"], index=0)
                        y_axis = st.selectbox(
                            "Y-axis",
                            ["churn_probability", "MonthlyCharges", "TotalCharges", "tenure"],
                            index=0,
                        )
                        st.scatter_chart(
                            view, x=x_axis, y=y_axis, color="risk", size="MonthlyCharges"
                        )
                        st.write("**Selected group summary**")
                        st.dataframe(
                            view[["tenure", "MonthlyCharges", "TotalCharges", "churn_probability"]].describe().T,
                            use_container_width=True,
                        )
                    else:
                        st.info("No customers belong to this risk group.")

                    st.download_button(
                        "Download scored CSV",
                        data=scored.to_csv(index=False).encode("utf-8"),
                        file_name="scored_customers.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

with info_tab:
    st.subheader("Saved model")
    st.write("Artifact: `churn_model.joblib`")
    st.write("Algorithm: XGBoost classifier inside a scikit-learn Pipeline.")
    st.write("Preprocessing: numeric columns pass through; categorical columns are one-hot encoded.")
    st.write("Training features:")
    st.code(", ".join(FEATURE_COLS))
    st.warning(
        "This version uses the model artifact produced by train_final_model.py. "
        "It does not contain the comparison metrics, calibration curves, feature explanations, "
        "or paper-review tables expected by the older app. Those sections were removed to "
        "avoid reading keys that do not exist in this saved pipeline."
    )
