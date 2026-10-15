"""Run once offline; the app only loads the resulting fitted pipelines."""
import hashlib
import json
import platform
import joblib
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, silhouette_score, davies_bouldin_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ml_core import ROOT, NUMERIC, CATEGORICAL, SEGMENT_FEATURES, CustomerFeatures

def preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]),
         NUMERIC + ['ServiceCount', 'AvgChargePerTenureMonth']),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         CATEGORICAL)])

def train():
    output = ROOT / 'models' / 'app'
    output.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / 'data' / 'TelcoCustomerChurn.csv'
    raw = pd.read_csv(raw_path)
    reports = {}
    for track, target in [('cltv', 'CLTV'), ('churn', 'ChurnLabel')]:
        if track == 'cltv':
            # Preserve the notebook's existing train/test partition.
            train = pd.read_csv(ROOT / 'data' / 'regression_train.csv')
            test = pd.read_csv(ROOT / 'data' / 'regression_test.csv')
            X_train, X_test, y_train, y_test = train, test, train[target], test[target]
            estimator = GradientBoostingRegressor(random_state=42, max_depth=2, n_estimators=200, learning_rate=0.05)
        else:
            y = raw[target].map({'No': 0, 'Yes': 1})
            X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42, stratify=y)
            estimator = GradientBoostingClassifier(random_state=42, max_depth=2, n_estimators=150)
        pipe = Pipeline([('features', CustomerFeatures()), ('preprocess', preprocessor()), ('model', estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        report = {'train_rows': len(X_train), 'test_rows': len(X_test)}
        if track == 'cltv':
            report.update(r2=r2_score(y_test, pred), mae=mean_absolute_error(y_test, pred),
                          rmse=mean_squared_error(y_test, pred)**0.5)
        else:
            report.update(accuracy=accuracy_score(y_test, pred), precision=precision_score(y_test, pred),
                          recall=recall_score(y_test, pred), f1=f1_score(y_test, pred),
                          roc_auc=roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
                          confusion_matrix=confusion_matrix(y_test, pred).tolist())
        joblib.dump({'track': track, 'pipeline': pipe}, output / f'{track}.joblib', compress=3)
        reports[track] = report
    segment_pre = ColumnTransformer([('numeric', Pipeline([
        ('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), SEGMENT_FEATURES)])
    features = CustomerFeatures().fit_transform(raw)
    scaled = segment_pre.fit_transform(features)
    candidates = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(scaled)
        candidates.append({'k': k, 'silhouette': silhouette_score(scaled, model.labels_, sample_size=1500, random_state=42)})
    best = max(candidates, key=lambda row: row['silhouette'])
    pipe = Pipeline([('features', CustomerFeatures()), ('preprocess', segment_pre),
                     ('model', KMeans(n_clusters=best['k'], n_init=20, random_state=42))]).fit(raw)
    labels = pipe.predict(raw)
    profiles = features[SEGMENT_FEATURES].assign(Segment=labels).groupby('Segment').mean().round(2)
    profiles['Customers'] = pd.Series(labels).value_counts().sort_index()
    profiles.to_csv(output / 'segment_profiles.csv')
    joblib.dump({'track': 'segments', 'pipeline': pipe}, output / 'segments.joblib', compress=3)
    reports['segments'] = {'rows': len(raw), 'selected_k': best['k'], 'candidates': candidates,
                            'davies_bouldin': davies_bouldin_score(scaled, labels)}
    reports['provenance'] = {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                             'data_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                             'seed': 42, 'supervised_refit_on_holdout': False}
    (output / 'metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))

if __name__ == '__main__':
    train()
