# Customer Churn Prediction System (IBM Telco)

A small end-to-end machine learning project that predicts which telecom customers are likely to leave (churn), and serves the model through a Streamlit web app.

The project follows the general method of a published churn-prediction paper (Mettle et al., 2026) but fixes several methodological problems found in it: possible data leakage, no significance testing, no calibration, and unreproducible results. See [Paper gaps and our fixes](#paper-gaps-and-our-fixes).

> **Honest headline:** on this dataset an honest pipeline reaches about **0.84 to 0.85 ROC-AUC** and about **80% accuracy**. Results above 90% accuracy on this dataset almost always come from data leakage.

---

## Contents

- [What it does](#what-it-does)
- [Project structure](#project-structure)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Using the app](#using-the-app)
- [Method](#method)
- [Results](#results)
- [Paper gaps and our fixes](#paper-gaps-and-our-fixes)
- [Limitations](#limitations)
- [Tech stack](#tech-stack)

---

## What it does

- Trains and compares **Decision Tree, Random Forest, XGBoost, Logistic Regression** and a **soft-voting ensemble**.
- Tunes hyperparameters with grid search and 5-fold cross-validation.
- Verifies performance with **nested 5-fold cross-validation**.
- Calibrates predicted probabilities and reports a threshold trade-off table.
- Provides a **Streamlit app** for single-customer prediction (with an explanation of the factors), batch CSV scoring, and a full model report.

## Project structure

```
churn_system/
├── app.py               # Streamlit web app
├── train.py             # Main training pipeline (saves the model)
├── kfold_eval.py        # Nested 5-fold evaluation (tuned vs default models)
├── telco.csv            # IBM Telco Customer Churn dataset
├── requirements.txt     # Python dependencies
├── .gitignore           # Files Git should not track (venv, caches, logs)
├── churn_model.joblib   # Trained model + metadata (created by train.py)
├── metrics.json         # Results from train.py (created by train.py)
├── kfold_results.json   # Nested CV summary (created by kfold_eval.py)
└── kfold_folds.csv      # Per-fold nested CV scores (created by kfold_eval.py)
```

## Setup

Requires Python 3.9 or newer.

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

Keep all files in **one folder** and run every command from inside it. The scripts read `telco.csv` and `churn_model.joblib` from the current directory.

## How to run

Run these in order, one at a time:

```bash
python train.py                      # ~5 min. Builds churn_model.joblib
python kfold_eval.py                 # ~3 min. Optional, adds section 7 to the app
python -m streamlit run app.py       # Opens http://localhost:8501
```

Run times depend on your CPU. If you change the training code, re-run `train.py` before starting the app, because the app loads the saved model.

## Using the app

| Tab | What it does |
|---|---|
| **Single customer** | Fill in a customer profile and get a churn probability, a risk band, and a chart of the factors pushing the prediction up or down. |
| **Batch scoring (CSV)** | Upload a CSV with the same columns as `telco.csv` (`customerID` and `Churn` are optional). Returns customers ranked by risk, with a download button. |
| **Model report** | Model comparison with confidence intervals, confusion matrices, the imbalance study, calibration, threshold trade-offs, feature importance, and nested CV results. |
| **Paper gaps vs our fixes** | Summary of the reference paper's weaknesses and how this project handles them. |

The sidebar slider sets the "at risk" threshold. Lower values catch more churners but flag more customers who would have stayed. The default is 0.30, a starting point and not an optimized value. Use the trade-off table in the Model report to choose.

**Quick sanity check** (values from our run; yours may differ slightly):

| Customer | Result |
|---|---|
| tenure 2, month-to-month, fiber optic, electronic check, no tech support | about 53% churn probability |
| tenure 60, two-year contract, DSL, automatic bank transfer, tech support | about 1.5% churn probability |

## Method

1. **Data:** IBM Telco Customer Churn, 7,043 customers and 21 columns. Churn rate is about 26.5%. The 11 rows with invalid `TotalCharges` (all tenure-0 customers) are dropped, leaving 7,032 rows.
2. **Split:** stratified 80/20 train/test (5,625 / 1,407). The test set is used once for reporting.
3. **Preprocessing:** one-hot encoding of categoricals (binary columns reduced to one column) and scaling for logistic regression only. Everything sits **inside a scikit-learn Pipeline**, so nothing leaks from the test set.
4. **Imbalance handling:** class weights (`class_weight` / `scale_pos_weight`) in every model. No resampling before splitting.
5. **Tuning:** `GridSearchCV`, 5-fold, optimizing ROC-AUC. Models are selected by cross-validation AUC, never by test results.
6. **Ensemble:** soft-voting over the four tuned models.
7. **Statistics:** 1,000-sample bootstrap confidence intervals on test AUC, and paired bootstrap p-values against the selected model.
8. **Calibration:** isotonic calibration (5-fold) of the selected model, kept if it lowers the Brier score.
9. **Explanations:** per-customer contributions from XGBoost (`pred_contribs`) and global permutation importance.
10. **Nested CV (`kfold_eval.py`):** outer 5-fold scores all 7,032 customers, inner 3-fold grid search tunes inside each training fold. Untuned default models are scored on the same folds.

### Hyperparameter grids

| Model | Searched |
|---|---|
| Decision Tree | `max_depth` {3, 5, 8}, `min_samples_leaf` {5, 20} |
| Random Forest | `n_estimators` {300}, `max_depth` {8, 12}, `min_samples_leaf` {3, 8} |
| XGBoost | `n_estimators` {150, 300}, `max_depth` {3, 4}, `learning_rate` {0.03, 0.08}, `subsample` {0.8, 1.0} |
| Logistic Regression | `C` {0.01, 0.1, 1, 10} |

## Results

Numbers below are from our runs. Small differences are expected on other machines and library versions.

### Test-set results (single 80/20 split, ROC-AUC)

| Model | Test AUC [95% CI] | p vs selected model |
|---|---|---|
| **XGBoost** (selected on CV) | 0.840 [0.819, 0.861] | n/a |
| Soft-Voting Ensemble | 0.841 [0.819, 0.862] | 0.59 |
| Random Forest | 0.836 [0.815, 0.857] | 0.04 |
| Logistic Regression | 0.834 [0.812, 0.856] | 0.12 |
| Decision Tree | 0.828 [0.805, 0.849] | 0.02 |

The top models are statistically close. XGBoost is not clearly better than logistic regression or the ensemble.

### Nested 5-fold cross-validation (ROC-AUC, mean ± std)

| Model | Tuned | Default settings |
|---|---|---|
| XGBoost | 0.849 ± 0.005 | 0.825 |
| Random Forest | 0.847 ± 0.004 | 0.823 |
| Logistic Regression | 0.845 ± 0.003 | 0.845 |
| Decision Tree | 0.831 ± 0.010 | 0.658 |

Tuning helps tree models (mainly by limiting overfitting) and does nothing for logistic regression.

### Imbalance handling (same Random Forest settings)

| Method | Accuracy | Churn recall | ROC-AUC |
|---|---|---|---|
| No imbalance handling | 79.4% | 50.5% | 0.837 |
| Class weights (used here) | 74.9% | 78.3% | 0.836 |
| SMOTE inside CV folds (correct) | 76.8% | 65.0% | 0.827 |
| SMOTE **before** the split (leaky, demo only) | 83.9% | 86.8% | 0.924 |

Only the leaky variant approaches the reference paper's reported profile.

### Calibration and thresholds

- Brier score improved from 0.170 (raw) to 0.138 (calibrated).
- At threshold 0.30 the deployed model has about **50% precision and 80% recall**; at 0.40, about **61% precision and 62% recall**.

Accuracy is a poor metric for this problem (a model that always predicts "stays" scores about 73.5%), so compare models on AUC, precision and recall.

## Paper gaps and our fixes

| Weakness in the reference paper | How this project handles it |
|---|---|
| Row count inconsistent (5,634 claimed after cleaning; only 11 rows are invalid) | Row counts are reported at every step |
| Confusion matrices appear to belong to a different dataset (labelled "Malignant/Benign", 114 samples) | Real matrices from the 1,407-customer test set |
| Reported scores (84.7% accuracy, 0.939 AUC) do not reproduce with an honest pipeline | Leak-free pipelines; honest result is about 80% accuracy and 0.84 AUC |
| Scores consistent with oversampling before splitting | Leakage demonstrated side by side; SMOTE only inside CV folds |
| No significance testing | Bootstrap confidence intervals and paired p-values |
| Single-split scores only | 5-fold CV mean ± std, plus nested 5-fold CV |
| Imbalance handled only through metrics | Class weights inside every model |
| Soft-voting ensemble claimed but not reported | Built and evaluated (no significant gain) |
| No probability calibration, fixed 0.5 threshold | Isotonic calibration, threshold trade-off table, adjustable slider |
| Explainability left as future work | Per-customer explanations and global importance |
| No deployable output | Streamlit app |

We cannot see the paper authors' code, so the leakage point is an inference from the numbers, not a proven fact.

## Limitations

- The dataset has **no timestamps**, so temporal validation and concept drift cannot be tested.
- Validated on **one dataset only**; there is no external validation.
- Main test results come from a single train/test split (confidence intervals come from bootstrapping; nested CV adds a second view).
- Hyperparameter grids are small, and several chosen XGBoost and Random Forest values sit at the grid edge, so the search is not exhaustive.
- Features are static account attributes; there is no usage, complaint or network-quality data.
- Per-customer explanations come from the XGBoost component, not the calibrated wrapper.
- The model shows association with churn, not that an intervention would retain a customer.
- The default 0.30 threshold is not cost-optimized; real deployment would need real retention costs.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, Streamlit, joblib.

## Credits

- Dataset: IBM Telco Customer Churn (publicly available, also on Kaggle).
- Reference paper: H. N.-A. Mettle, E. H. Ayerko, P. Appiahene, "Machine Learning-Based Customer Churn Prediction in Telecommunication Industry," *Applied Computational Intelligence and Soft Computing*, 2026.