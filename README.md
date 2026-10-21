# Telco Customer Insights — ML Evaluation

A working Streamlit application for CLTV regression, churn classification, and customer segmentation on the included 7,043-row Telco dataset.

## Run locally

Use Python 3.11 (the version used to train and test the saved models).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The fitted pipelines are included under `models/app/`. The app never trains on startup and needs no API keys. Use example customers, fill in a single customer, or download the CSV template and upload up to 10,000 rows / 10 MB. Each track supports prediction downloads. CSV headers must match the template. Extra columns are preserved in exports but ignored by models.

The dashboard includes a navy and teal visual theme, model evaluation cards, portfolio summaries, interactive value/risk distributions, and a customer segment bubble chart. Single-customer inputs are grouped into profile, billing, and services. Example mode offers reproducible samples of 10–100 customers; prediction columns appear first in the results table.

### App navigation

- **Analyze customers:** choose example customers, a single customer, or a CSV upload. Single-customer fields use readable labels and three form sections. Results include a plain-language interpretation, a visual summary, and a separate records/export tab. The last submitted customer remains available when adjusting the churn threshold.
- **Model performance:** view the fitted method, measured evaluation, metric explanations, limitations, and technical training details. Churn includes the test-set confusion matrix; clustering shows the group-count comparison.
- **User guide:** follow the workflow, learn the meaning of each output, review upload rules, and consult the input-column reference.

Segment descriptions summarize the saved profiles. The dashboard labels are friendly names; exported CSVs retain the original column names. Histograms are aggregated before plotting so a 10,000-row upload is not blocked by the chart library's row limit.

## Existing project work and scope

All original notebooks, datasets, and existing model files are preserved. The previous README is retained as `README.original.md`.

Install `requirements-notebooks.txt` instead of the runtime requirements if you also want Jupyter and the original plotting dependencies.

- `notebooks/regression.ipynb`: Telco CLTV regression; existing processed datasets and train/test partitions are retained.
- `notebooks/classification (1).ipynb`: credit-card default, **not Telco churn**.
- `notebooks/clustering.ipynb`: meteorite clustering, **not customer segmentation**.

The deployment adds separate Telco churn and customer segmentation pipelines; it does not claim these were the original notebook models. It does not replace the notebooks' broader algorithm comparisons or constitute a new full capstone report.

## Models and evaluation

Run `python train_models.py` from the repository root to reproduce all deployment artifacts. This overwrites only `models/app/` deployment files. Fit with the pinned dependencies before loading models; scikit-learn serialized models are version dependent. Load only trusted joblib files.

- **CLTV:** GradientBoostingRegressor (200 estimators, learning rate 0.05, depth 2). Uses the repository's existing `regression_train.csv` and `regression_test.csv` split. This is a newly fitted deployment model, not an exported notebook kernel object or a claim of exact notebook tuning reproduction.
- **Churn:** GradientBoostingClassifier (150 estimators, depth 2). Target `ChurnLabel` maps No=0 and Yes=1. Stratified 80/20 holdout, seed 42. Threshold defaults to 0.5 and can be changed in the app. Metrics report threshold 0.5. Historical classification scores are not calibrated future probabilities.
- **Segments:** KMeans on tenure, monthly charge, download usage, referrals, and service count. Median imputation and standard scaling precede clustering. Select K=2..6 by silhouette on a fixed 1,500-row sample, then fit the chosen K to all customers. This is descriptive full-population analysis, not held-out predictive evaluation. Profiles show segment sizes and feature means; cluster IDs have no rank or fixed meaning across retraining.

`models/app/metrics.json` contains actual measured holdout metrics, cluster selection scores, row counts, seed, library version, and source-data SHA-256. `segment_profiles.csv` contains interpretable cluster summaries. Supervised models remain fitted on training rows only.

Measured deployment results (seed 42, Python 3.11.9):

| Track | Metric | Value |
|---|---|---:|
| CLTV | Holdout R² | 0.2052 |
| CLTV | Holdout RMSE | 1058.52 |
| CLTV | Holdout MAE | 896.50 |
| Churn | Holdout accuracy | 0.8453 |
| Churn | Holdout precision | 0.7422 |
| Churn | Holdout recall | 0.6390 |
| Churn | Holdout F1 | 0.6868 |
| Churn | Holdout ROC-AUC | 0.9095 |
| Segments | Selected K | 6 |
| Segments | Sample silhouette | 0.3014 |

The six profiles distinguish low-spend customers with high referrals (0), long-tenure service-rich customers (1), shorter-tenure moderate-spend customers (2), low-spend customers with few services (3), long-tenure high-referral customers (4), and high-download customers (5). These descriptions summarize feature means, not causal explanations.

The fitted Pipeline includes feature engineering, imputers, scaling, encoding, and estimator. Numeric blanks use training medians; missing Offer and InternetType become No Offer and No Internet. Other categorical blanks use training modes. Unseen categories are ignored by the encoder. ServiceCount counts the eight add-on services; charge-per-tenure uses TotalCharges / TenureinMonths with zero tenure treated as missing. Invalid numeric text, infinities, negative values, empty uploads, and missing columns produce validation messages.

Input allowlists exclude customer identifiers, geography, CLTV, all churn outcomes, ChurnScore, CustomerStatus, and SatisfactionScore. SatisfactionScore is omitted because its collection timing is unspecified. Model inputs should reflect information available when scoring. CLTV horizon and currency are unspecified in the source; estimates have substantial error and should be interpreted alongside the metrics.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests load each real saved pipeline, compare single/batch predictions, check outcome-field exclusion, missing/unseen input handling and validation, and exercise all three Streamlit views and single-customer form submissions. GitHub Actions runs these tests with Python 3.11. The app's CSV prediction logic is tested directly; automated UI tests do not exercise the browser file picker.

## Deploy to Streamlit Community Cloud

1. Upload/commit these changes, including `models/app/`, to `manaswiniyoshitha2006-netizen/ML-Evaluation` using an account with write permission. Keep the lowercase `requirements.txt` name on GitHub (the old uppercase file was renamed).
2. Sign in to https://share.streamlit.io with the account authorized for the repository and select **Create app**.
3. Choose repository `manaswiniyoshitha2006-netizen/ML-Evaluation`, the branch containing these changes, and main file `streamlit_app.py`.
4. In advanced settings choose **Python 3.11**, then deploy. No secrets are required.
5. Confirm the example predictions work on all tracks, submit a customer form, and upload the downloaded CSV template.

Community Cloud builds dependencies from the repository's requirements file; it loads the committed fitted models. See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Apply the deliverable safely

The accompanying `streamlit-app-changes.zip` contains only added/changed files. Extract it into an up-to-date clone, review changes, stage the `Requirements.txt` → `requirements.txt` rename explicitly on case-sensitive systems, and commit on a new branch. Do not replace the data or notebooks. `delivery.patch` is also supplied for tracked text changes; the changes ZIP includes the binary trained models too. The complete updated repository ZIP provides a separate runnable copy.

No remote push or live cloud deployment is implied by these deliverables. Publishing requires a GitHub account with write access and access to the Streamlit workspace.
