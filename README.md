# ML Capstone Explorer

A structured Streamlit application for **three independent datasets**, matching the project notebooks.

| Track | Actual dataset | Task | Deployment model |
|---|---|---|---|
| Regression | `data/TelcoCustomerChurn.csv` | Estimate supplied customer lifetime value (`CLTV`) | Gradient Boosting Regressor |
| Classification | `data/default of credit card clients.csv` | Default payment next month (`Y`: 0 no, 1 yes) | Decision Tree Classifier |
| Clustering | `data/Meteorite_Landings.csv` | Group landing records by mass, year, and coordinates | K-Means |

Classification does **not** use Telco churn data. Clustering does **not** use Telco customer segmentation. Each track loads its own fitted pipeline, input schema, example records, CSV template, charts, and evaluation. The dataset name is always visible in the interface.

## Run locally

Use Python **3.11** and run commands from the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the Local URL printed by Streamlit. Keep the terminal open. No API keys or training at startup are required. Install `requirements-notebooks.txt` if you also need Jupyter and plotting dependencies.

## App structure

- **Run analysis:** select example records, enter a single record in grouped forms, or upload a CSV. Read the interpretation, inspect Visual summary, and open Records & export to download predictions.
- **Model performance:** see the method, dataset, measured evaluation, metric explanations, and training provenance. Classification includes a confusion matrix; clustering includes K selection scores.
- **Dataset & user guide:** compare the three tracks, review code definitions and cleaning rules, and consult the selected track's input reference.

Uploads must use the selected track's template: Telco account columns for regression, `X1` through `X23` for classification, and `mass (g),year,reclat,reclong` for clustering. A CSV from a different track is rejected with missing-column details. Limits: 10,000 rows and 10 MB. Extra columns remain in exports but are excluded from prediction. Exported CSVs keep original dataset column names; the UI uses readable labels.

Example records are exploratory examples from the source population, not a separate held-out test. Performance metrics are computed offline on the documented split or clustering sample.

## Training and notebook alignment

```bash
python train_models.py
```

This recreates `models/app/{cltv,credit,meteorites}.joblib`, `metrics.json`, and `meteorite_profiles.csv`. Each joblib includes its complete fitted feature transformation/preprocessor and estimator. Use the pinned scikit-learn version and load only trusted joblib artifacts.

### Regression / Telco CLTV

Uses the existing `regression_train.csv` (5,634 rows) and `regression_test.csv` (1,409 rows). Median imputation, numeric standard scaling, and categorical one-hot encoding precede Gradient Boosting (200 estimators, depth 2, learning rate 0.05, seed 42). Feature engineering adds service count and charges per tenure month. Missing Offer and InternetType receive semantic categories. The allowlist excludes identifiers, geography, churn outcomes/scores, and SatisfactionScore.

This is the previously tested deployment model, rather than an export of a notebook kernel or a claim to recreate its full model comparison. CLTV currency and horizon are unspecified. R-squared is modest; estimates should not be treated as exact customer valuations.

### Classification / Credit-card default

Follows the latest classification notebook, `notebooks/classification  (1).ipynb`. Removes the embedded text-header row and ID field, converts values to numeric, and removes 35 duplicates, leaving **29,965 accounts**. Stratified 80/20 split: **23,972 training / 5,993 test**, seed 42.

Uses the notebook's decision-tree approach and depth grid `[3,5,7,10,15,None]`, selected by five-fold weighted F1 on training rows; selected depth is **5**. Deployment cross-validation keeps preprocessing inside each fold to avoid sharing preprocessing statistics across folds. This does not rerun the full notebook's ten-model comparison.

Feature engineering computes average bill and payment amounts before clipping the raw continuous fields to training 1st/99th percentiles, matching notebook ordering. Numeric imputation/scaling and category-code imputation/encoding are saved with the estimator. X2/X3/X4 are categorical codes. Y and IDs are excluded. Negative bill balances and repayment status codes are valid. The form explains codes and months (April through September 2005); amounts use NT dollars. Blank numeric values use training medians; category blanks use training modes. Unsupported category/status codes and invalid numeric values are rejected.

The threshold defaults to 0.50 and changes the Yes/No label, not the score. Scores are not calibrated probabilities. This is an academic model using historical demographic and financial features, not a lending eligibility system.

### Clustering / Meteorite landings

Follows the K-Means branch of `notebooks/clustering.ipynb`. Of **45,716 raw rows**, **31,929** are retained. Excludes missing/nonfinite selected features, negative mass, years outside 0-2026, invalid latitude/longitude, and coordinates (0,0). The fixed 2026 cutoff matches the notebook's analysis year. All four input columns are required at inference; invalid uploaded records produce explicit errors rather than being silently dropped.

Uses `log1p(mass)` and StandardScaler on `[mass,year,reclat,reclong]`. Compares K=2..10 with n_init=10 and seed 42 using silhouette on the same 1,500 sampled records. Fits the selected **K=2** with n_init=20 on the complete cleaned population. The pipeline assigns new records with `predict`; it does not recluster per upload. Full-dataset cluster medians/counts are in `meteorite_profiles.csv`.

Cluster IDs are identities, not rankings or verified meteorite types. Euclidean distance on standardized coordinates is not great-circle distance. The coordinate plot samples at most 5,000 points for large batches, with an explicit notice; summaries and exports include every validated row. The hierarchical clustering notebook is preserved, but no approximate out-of-sample hierarchical prediction is presented.

## Measured deployment results

| Track | Metric | Result |
|---|---|---:|
| CLTV | Holdout R-squared | 0.2052 |
| CLTV | Holdout RMSE / MAE | 1058.52 / 896.50 |
| Credit default | Holdout accuracy | 0.8173 |
| Credit default | Precision / recall | 0.6518 / 0.3741 |
| Credit default | Weighted F1 / ROC-AUC | 0.7978 / 0.7446 |
| Meteorites | Selected K | 2 |
| Meteorites | Sample silhouette | 0.4881 |

`models/app/metrics.json` stores full metrics, row counts, selection scores, parameters, source-data SHA-256 values, seed, and library version. Supervised pipelines remain fitted only on training rows. Clustering is descriptive full-population analysis, not supervised holdout evaluation.

## Preservation of existing work

Original notebooks and data remain unchanged. The prior Telco-only churn/segmentation artifacts and their metrics are preserved under `models/legacy_telco/` and are **not loaded by the app**. `README.original.md` retains the initial project README. `ml_core.py` retains legacy helpers so those old joblib artifacts remain readable; the current app dispatches through `track_data.py`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover all three real saved pipelines, target/ID exclusion, single/batch consistency, different-track schema rejection, CSV round-trips, dataset cleaning and validation, distinct data provenance, each UI form/upload view, and classification threshold persistence. GitHub Actions uses Python 3.11. UI file-picker automation is not part of the test suite.

## Deploy or update Streamlit Community Cloud

1. Use repository `manaswiniyoshitha2006-netizen/ML-Evaluation` and the branch containing these files (normally `main`).
2. Set the entry point to `streamlit_app.py` and Python version to **3.11** in Advanced settings.
3. Use a general available subdomain, such as `ml-capstone-explorer`. No secrets are needed.
4. Include `models/app/`, the three source datasets, and the lowercase `requirements.txt` in GitHub.
5. After deployment, verify all three dataset labels, run an example and single record on each track, and try each downloaded template.

For an existing deployment connected to main, pushing this update should trigger an automatic update. If the old page remains visible, reboot the app from its Streamlit management menu and refresh the browser. Deployment itself and URL availability must be verified in your Streamlit workspace.

Official instructions: [deploy](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [manage updates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).
