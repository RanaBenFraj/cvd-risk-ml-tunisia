# ML_HOSPITAL

**Comparative Analysis of Machine Learning Algorithms for Predicting CVD Risk in a Tunisian Hospital Population**

A fully reproducible, end-to-end machine-learning pipeline: raw data → cleaning → EDA →
preprocessing → model training → final evaluation. Every notebook runs top-to-bottom
without errors and produces byte-identical numerical results given the pinned library
versions and `random_state=42` used throughout.

---

## 1. Folder structure

```
ML_HOSPITAL/
├── data/
│   ├── raw/
│   │   └── CVD_not_clean.csv        <- original raw export (never modified)
│   └── cleaned/
│       └── CVD_cleaned.csv          <- produced by 02_Data_Cleaning.ipynb
├── notebooks/
│   ├── 01_Data_Inspection.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_ML_Preprocessing.ipynb
│   ├── 05_Model_Training.ipynb
│   └── 06_Model_Evaluation.ipynb
├── ml_artifacts/
│   ├── models/
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib
│   │   ├── logistic_regression.joblib
│   │   └── xgboost.joblib
│   ├── splits/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   ├── ml_config.json
│   ├── cv_tuning_summary.csv
│   ├── test_evaluation_summary.csv
│   └── test_per_class_metrics.csv
├── figures/                          <- all figures produced by every notebook (42 PNGs)
└── README.md
```

All notebooks use **relative paths** (`Path.cwd().parent`), so the project is portable:
it works identically whether it sits at `C:\Users\<you>\Desktop\ML_HOSPITAL`, on Linux,
or anywhere else, as long as notebooks are run with their working directory set to
`notebooks/` (the Jupyter default when you open a notebook from that folder).

---

## 2. Environment setup

### Python version
Python 3.10–3.12 all work. This project was built and verified against **Python 3.12**.

### Required package versions (pinned for exact reproducibility)

```
scikit-learn==1.4.2
imbalanced-learn==0.12.3
xgboost==3.4.1
pandas
numpy
matplotlib
seaborn
joblib
jupyter
nbconvert
ipykernel
```

Install with:

```bash
pip install scikit-learn==1.4.2 imbalanced-learn==0.12.3 xgboost==3.4.1 \
            pandas numpy matplotlib seaborn joblib jupyter nbconvert ipykernel
```

(On some systems you may need `pip install --break-system-packages ...` or a virtual
environment: `python -m venv venv && venv\Scripts\activate` on Windows, then run the
`pip install` line above.)

Using different library versions will still run correctly, but tree/ensemble
tie-breaking and floating-point details can shift metrics by a very small amount —
use the pinned versions above for numbers that match this README exactly.

---

## 3. How to reproduce the project from scratch

1. Place the raw file at `data/raw/CVD_not_clean.csv` (already included).
2. Open a terminal **inside `notebooks/`** (or launch Jupyter and `cd` into it) and run
   the six notebooks **in order**, top to bottom. Either:
   - Open each `.ipynb` in Jupyter/JupyterLab and choose **Run All**, in this order:
     `01 → 02 → 03 → 04 → 05 → 06`; or
   - Execute headlessly from the command line:

```bash
cd ML_HOSPITAL/notebooks
jupyter nbconvert --to notebook --execute --inplace 01_Data_Inspection.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_Data_Cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_ML_Preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_Model_Training.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_Model_Evaluation.ipynb
```

Notebook 05 (`GridSearchCV` over 4 models) is the slowest step, taking roughly
3–6 minutes on a typical laptop CPU.

3. After running all six, you will have:
   - `data/cleaned/CVD_cleaned.csv` — the cleaned, encoded dataset (1,529 rows × 15 columns, 0 missing, 0 duplicates).
   - `ml_artifacts/splits/{X_train,X_test,y_train,y_test}.csv` — the stratified 80/20 split (seed 42; 1,223 train / 306 test rows).
   - `ml_artifacts/ml_config.json` — feature groups, target mapping, split parameters.
   - `ml_artifacts/models/*.joblib` — the four final fitted `imblearn` pipelines (imputation → [scaling] → SMOTENC → classifier), refit on the full training set with the best `GridSearchCV` hyperparameters.
   - `ml_artifacts/cv_tuning_summary.csv` — best cross-validated Macro F1 per model (training set only).
   - `ml_artifacts/test_evaluation_summary.csv` / `test_per_class_metrics.csv` — the final, decision-driving test-set metrics.
   - `figures/` — 42 PNG figures covering every visual in notebooks 01–06.

No notebook modifies the raw file. Re-running the full pipeline from a clean checkout
reproduces every number and every saved file exactly, because:
- `random_state=42` is fixed everywhere a random process occurs (train/test split, `StratifiedKFold`, `SMOTENC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`, `XGBClassifier`).
- The 80/20 split is stratified by the target, so class proportions are always identical.
- SMOTENC and scaling are only ever fit inside the training pipeline (`imblearn.pipeline.Pipeline`), never on the test set, so the test set stays leakage-free.

---

## 4. What each notebook does

| Notebook | Input | Output | Key logic |
|---|---|---|---|
| `01_Data_Inspection` | `data/raw/CVD_not_clean.csv` | figures only | Structural overview, missingness, duplicates, categorical/numerical ranges, BMI & BP plausibility checks, two-batch structural finding, target imbalance, leakage screen. **Never modifies the data.** |
| `02_Data_Cleaning` | raw CSV | `data/cleaned/CVD_cleaned.csv` | Encodes all 6 categorical columns in place (F/M, Y/N, ordinal activity/risk), recalculates BMI from Weight/Height for every row, resolves invalid Systolic ≤ Diastolic pairs (swap or set-to-missing), median-imputes all 9 numeric columns **once**, deduplicates, exports. |
| `03_EDA` | cleaned CSV | figures only | Descriptive statistics, class balance, univariate/bivariate analysis vs. target, correlation matrix, outlier screening (1.5×IQR). |
| `04_ML_Preprocessing` | cleaned CSV | `ml_artifacts/splits/*.csv`, `ml_config.json` | Defines 14 features (9 continuous + 5 binary/ordinal) and the target; stratified 80/20 split (`test_size=0.20, random_state=42`); demonstration-only SMOTENC run (discarded, not used for training) to show before/after class balance. |
| `05_Model_Training` | split CSVs, config | `ml_artifacts/models/*.joblib`, `cv_tuning_summary.csv` | Builds `imblearn.pipeline.Pipeline`s (`SimpleImputer → [StandardScaler for LR only] → SMOTENC → classifier`), tunes each of the 4 models with `GridSearchCV(scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42))`, refits on the full training set, saves the fitted pipelines. **Never touches the test set.** |
| `06_Model_Evaluation` | test split, saved models | `test_evaluation_summary.csv`, `test_per_class_metrics.csv`, figures | Loads the four saved pipelines, predicts on the untouched 306-row test set, reports per-class precision/recall/F1, Macro F1 (primary metric), Weighted F1, Macro ROC-AUC (secondary), confusion matrices, feature importances, and Logistic Regression coefficients. **No retraining.** |

---

## 5. Reference results (obtained with the pinned versions above, seed 42)

**5-fold cross-validated Macro F1 (training set, for hyperparameter selection):**

| Model | Best CV Macro F1 | Best hyperparameters |
|---|---|---|
| Random Forest | 0.6005 (±0.0188) | `max_depth=10, max_features='sqrt', min_samples_leaf=10, n_estimators=200` |
| XGBoost | 0.5838 (±0.0277) | `learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0` |
| Decision Tree | 0.5358 (±0.0296) | `criterion='gini', max_depth=7, min_samples_leaf=5` |
| Multinomial Logistic Regression | 0.5087 (±0.0389) | `C=100.0` |

**Final test-set evaluation (306 held-out rows, never seen during training/tuning):**

| Model | Accuracy | Macro Precision | Macro Recall | **Macro F1** | Weighted F1 | Macro ROC-AUC (OvR) |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.6765 | 0.5995 | 0.6028 | **0.6006** | 0.6803 | 0.7928 |
| XGBoost | 0.6699 | 0.5742 | 0.5729 | 0.5732 | 0.6664 | 0.7880 |
| Multinomial Logistic Regression | 0.5850 | 0.5446 | 0.5370 | 0.5266 | 0.6039 | 0.7158 |
| Decision Tree | 0.5458 | 0.5079 | 0.5149 | 0.5086 | 0.5511 | 0.6595 |

**Random Forest is the best-performing model by Macro F1**, with a margin of ~0.027
points over the second-best model (XGBoost).

If your run produces different numbers, first check that you used the exact pinned
library versions (see Section 2) — `scikit-learn`, `imbalanced-learn`, and `xgboost`
each have internal RNG usage and default-parameter behavior that changed across
versions, which can shift Macro F1 by a few hundredths even with `random_state=42`
fixed everywhere.

---

## 6. Known limitations (documented, not hidden)

1. **Two-segment file structure.** `01_Data_Inspection.ipynb` shows that the raw file
   appears to concatenate two cohorts (rows 0–985 and 986–1528) that differ in age
   range, BP/glucose extremes, BMI-recording reliability, and target balance. This is
   documented as a methodological limitation; models are trained/evaluated on the
   pooled mixture, and a diagnostic-only batch label (reconstructed from row position)
   is used **only** to confirm both segments appear in both splits — it is never used
   as a predictive feature.
2. **SMOTENC-based oversampling** creates synthetic minority-class (`LOW`) training
   examples by interpolation; this helps the classifier learn the minority boundary but
   adds no genuinely new clinical information.
3. **Weak individual numerical predictors** — most continuous variables correlate only
   weakly with the target (|r| < 0.2); models rely on combining several weak signals.
4. **Single train/test split** — a repeated nested cross-validation study would give
   tighter confidence intervals on the reported metrics.
5. **No causal inference** — feature importances and Logistic Regression coefficients
   describe learned statistical associations only, not causal clinical relationships.

---

## 7. Troubleshooting

- **`ModuleNotFoundError`** — reinstall the pinned versions from Section 2 inside the
  active Python environment/kernel that Jupyter is using.
- **Different Macro F1 than Section 5** — verify `scikit-learn==1.4.2`,
  `imbalanced-learn==0.12.3`, `xgboost==3.4.1` exactly (`pip show <package>`).
- **`FileNotFoundError` for the raw CSV** — confirm the working directory is
  `notebooks/` when you launch Jupyter (the notebooks use `Path.cwd().parent` to find
  `data/raw/CVD_not_clean.csv`).
- **Notebook 05 is slow** — this is expected; it runs 4 `GridSearchCV` searches
  (40 + 36 + 5 + 36 = 117 parameter combinations × 5 folds = 585 total model fits).
