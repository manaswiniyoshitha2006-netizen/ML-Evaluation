# Customer Lifetime Value & Churn Analysis — ML Evaluation Project

## 📌 Project Overview
This project applies **regression** techniques to the Telco Customer Churn dataset. It is used to find the CLTV.

## 📂 Repository Structure
```
/
├── README.md
├── requirements.txt
├── data/                  # Raw dataset + processed CSVs
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/                # Saved model files (.pkl via joblib)
└── app/                   # GUI / deployment code (optional bonus)
```

## 📊 Dataset
- **Source:** Telco Customer Churn dataset (`TelcoCustomerChurn.csv`)
- Contains customer demographics, account information, service usage, billing details, and churn outcomes.
- Missing values were treated semantically (e.g., missing `Offer` → `"No Offer"`, missing `InternetType` → `"No Internet"`) rather than statistically imputed, since they represent meaningful categories.

## ⚙️ Setup Instructions
1. Clone this repository:
```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
```
2. Create a virtual environment (recommended):
```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Launch Jupyter and run the notebooks in `notebooks/`:
```bash
   jupyter notebook
```

## 🔬 Methodology (Regression)
- **Target:** `CLTV` (Customer Lifetime Value)
- **Preprocessing:** Median imputation + `StandardScaler` for numeric features; most-frequent imputation + `OneHotEncoder` for categorical features, all wrapped in a `ColumnTransformer`/`Pipeline` to prevent data leakage.
- **Feature engineering:** Added `ServiceCount` (total add-on services) and `AvgChargePerTenureMonth`.
- **Split:** Stratified 80/20 train-test split (stratified on CLTV deciles).
- **Models compared (10):** Linear, Ridge, Lasso, ElasticNet, Polynomial (deg 2 & 3), Decision Tree, Random Forest, Gradient Boosting, SVR, KNN — each evaluated with R², RMSE, and MAE, with `GridSearchCV` (5-fold) hyperparameter tuning.

### Results Summary
| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Gradient Boosting (tuned)** | **0.2090** | **1056.00** | **895.62** |
| Random Forest (tuned) | 0.2013 | 1061.13 | 898.31 |
| Decision Tree (tuned) | 0.1978 | 1063.47 | 900.78 |
| Lasso (tuned) | 0.1378 | 1102.52 | 931.14 |
| SVR (tuned) | 0.1371 | 1102.96 | 928.81 |
| ElasticNet (tuned) | 0.1336 | 1105.21 | 931.61 |
| Ridge (tuned) | 0.1334 | 1105.34 | 931.59 |
| Linear Regression | 0.1319 | 1106.32 | 931.01 |
| KNN (tuned) | 0.0743 | 1142.42 | 958.28 |
| Polynomial (deg 2) | 0.0109 | 1180.86 | 968.09 |

**Best model:** Gradient Boosting Regressor, confirmed via 5-fold cross-validation (mean CV R² ≈ 0.2225).

## 🔬 Methodology (Classification)
> _TODO: Fill in once `classification.ipynb` is finalized — target variable, models compared, best-performing model, and key metrics (accuracy, precision, recall, F1, ROC-AUC)._

## 🔬 Methodology (Clustering)
> _TODO: Fill in once `clustering.ipynb` is finalized — clustering algorithm(s) used, number of clusters chosen, and key cluster interpretations._

## 🗂️ Saved Models
Trained model files are saved under `models/` using `joblib`:
```python
import joblib
joblib.dump(best_gb_model, "models/gradient_boosting_cltv.pkl")
```

## 🖥️ App / Deployment (Bonus)
> _TODO: Describe here if you build a GUI or deployment app under `app/`._

## 👥 Contributors
- <Add team member names here>

## 📄 License
> _Add a license if required by your course/assignment guidelines._

A couple of notes:

I pinned the versions loosely to what's current/stable — if pip freeze on your machine gives different versions, swap them in for accuracy.
The results table numbers come straight from your notebook's printed output — double check them against your final run before committing, in case reruns shift slightly.
Remember the grading note about committing at every milestone rather than one big commit — so don't dump all this in a single commit at the end.

give me correctly read me

Here's the README on its own, cleaned up and ready to paste directly into your README.md file:

markdown
# Customer Lifetime Value & Churn Analysis — ML Evaluation Project

## Project Overview
This project applies **regression**, **classification**, and **clustering** techniques to the Telco Customer Churn dataset. Each task is handled in its own notebook under `notebooks/`.

| Task | Notebook | Target Variable | Goal |
|---|---|---|---|
| Regression | `regression.ipynb` | `CLTV` | Predict Customer Lifetime Value |
| Classification | `classification.ipynb` | `ChurnLabel` | Predict customer churn |
| Clustering | `clustering.ipynb` | — | Segment customers into groups |

## Repository Structure
```
/
├── README.md
├── requirements.txt
├── data/                  # Raw dataset + processed CSVs
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/                # Saved model files (.pkl via joblib)
└── app/                   # GUI / deployment code (optional bonus)
```

## Dataset
- **Source:** Telco Customer Churn dataset (`TelcoCustomerChurn.csv`)
- Contains customer demographics, account information, service usage, billing details, and churn outcomes.
- Missing values were treated semantically rather than statistically imputed, since they represent meaningful categories (e.g., missing `Offer` → `"No Offer"`, missing `InternetType` → `"No Internet"`).

## Setup Instructions
1. Clone this repository:
```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
```
2. Create a virtual environment (recommended):
```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Launch Jupyter and run the notebooks in `notebooks/`:
```bash
   jupyter notebook
```

## Methodology — Regression
- **Target:** `CLTV` (Customer Lifetime Value)
- **Preprocessing:** Median imputation + `StandardScaler` for numeric features; most-frequent imputation + `OneHotEncoder` for categorical features, wrapped in a `ColumnTransformer` / `Pipeline` to prevent data leakage.
- **Feature engineering:** Added `ServiceCount` (total add-on services) and `AvgChargePerTenureMonth`.
- **Split:** Stratified 80/20 train-test split (stratified on CLTV deciles).
- **Models compared (10):** Linear, Ridge, Lasso, ElasticNet, Polynomial (degree 2 & 3), Decision Tree, Random Forest, Gradient Boosting, SVR, KNN — each evaluated with R², RMSE, and MAE, tuned via `GridSearchCV` (5-fold CV).

### Results Summary
| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Gradient Boosting (tuned)** | **0.2090** | **1056.00** | **895.62** |
| Random Forest (tuned) | 0.2013 | 1061.13 | 898.31 |
| Decision Tree (tuned) | 0.1978 | 1063.47 | 900.78 |
| Lasso (tuned) | 0.1378 | 1102.52 | 931.14 |
| SVR (tuned) | 0.1371 | 1102.96 | 928.81 |
| ElasticNet (tuned) | 0.1336 | 1105.21 | 931.61 |
| Ridge (tuned) | 0.1334 | 1105.34 | 931.59 |
| Linear Regression | 0.1319 | 1106.32 | 931.01 |
| KNN (tuned) | 0.0743 | 1142.42 | 958.28 |
| Polynomial (degree 2) | 0.0109 | 1180.86 | 968.09 |

**Best model:** Gradient Boosting Regressor, confirmed via 5-fold cross-validation (mean CV R² ≈ 0.2225).

## Methodology — Classification
> _TODO: Fill in once `classification.ipynb` is finalized — target variable, models compared, best-performing model, and key metrics (accuracy, precision, recall, F1, ROC-AUC)._

## Methodology — Clustering
> _TODO: Fill in once `clustering.ipynb` is finalized — algorithm(s) used, number of clusters chosen, and cluster interpretations._

## Saved Models
Trained model files are saved under `models/` using `joblib`:
```python
import joblib
joblib.dump(best_gb_model, "models/gradient_boosting_cltv.pkl")
