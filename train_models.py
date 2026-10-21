"""Reproducible deployment training for the three actual capstone datasets."""
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, silhouette_score, davies_bouldin_score)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from ml_core import ROOT, NUMERIC, CATEGORICAL, CustomerFeatures
from track_data import (TRACKS, CREDIT, CREDIT_CATEGORIES, METEORITE, CreditFeatures,
                        MeteoriteFeatures, clean_credit, clean_meteorites)

OUTPUT = ROOT / 'models' / 'app'

def preprocessing(numeric, categorical):
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)])

def provenance(track):
    path = ROOT / 'data' / TRACKS[track]['file']
    return {'dataset': path.name, 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'method': TRACKS[track]['method']}

def save(track, pipeline, report):
    joblib.dump({'track': track, 'pipeline': pipeline}, OUTPUT / f'{track}.joblib', compress=3)
    report.update(provenance(track))
    return report

def train():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    train_df = pd.read_csv(ROOT / 'data' / 'regression_train.csv')
    test_df = pd.read_csv(ROOT / 'data' / 'regression_test.csv')
    reg = Pipeline([('features', CustomerFeatures()),
                    ('preprocess', preprocessing(NUMERIC + ['ServiceCount', 'AvgChargePerTenureMonth'], CATEGORICAL)),
                    ('model', GradientBoostingRegressor(random_state=42, max_depth=2, n_estimators=200, learning_rate=.05))])
    reg.fit(train_df, train_df.CLTV)
    pred = reg.predict(test_df)
    reports['cltv'] = save('cltv', reg, {'train_rows': len(train_df), 'test_rows': len(test_df),
        'r2': r2_score(test_df.CLTV, pred), 'mae': mean_absolute_error(test_df.CLTV, pred),
        'rmse': mean_squared_error(test_df.CLTV, pred)**.5})
    print('CLTV trained.', flush=True)

    credit = clean_credit()
    X_train, X_test, y_train, y_test = train_test_split(credit[CREDIT], credit.Y.astype(int),
        test_size=.2, stratify=credit.Y, random_state=42)
    classifier = Pipeline([('features', CreditFeatures()),
        ('preprocess', preprocessing([c for c in CREDIT if c not in CREDIT_CATEGORIES] + ['Avg_Bill_Amount', 'Avg_Payment_Amount'], CREDIT_CATEGORIES)),
        ('model', DecisionTreeClassifier(random_state=42))])
    # Same depth grid and weighted-F1 selection as the notebook, with preprocessing
    # inside CV to prevent preprocessing statistics from leaking across folds.
    search = GridSearchCV(classifier, {'model__max_depth': [3, 5, 7, 10, 15, None]},
                          scoring='f1_weighted', cv=5, n_jobs=1)
    search.fit(X_train, y_train)
    classifier = search.best_estimator_
    pred = classifier.predict(X_test)
    reports['credit'] = save('credit', classifier, {'rows': len(credit), 'train_rows': len(X_train),
        'test_rows': len(X_test), 'selected_parameters': search.best_params_,
        'cv_weighted_f1': search.best_score_, 'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0), 'f1': f1_score(y_test, pred),
        'weighted_f1': f1_score(y_test, pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1]),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist()})
    print('Credit classification trained.', flush=True)

    meteorites = clean_meteorites()
    prep = Pipeline([('features', MeteoriteFeatures()), ('scale', StandardScaler())])
    scaled = prep.fit_transform(meteorites)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(scaled), size=min(1500, len(scaled)), replace=False)
    candidates = []
    for k in range(2, 11):
        candidate = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        score = silhouette_score(scaled[indices], candidate.labels_[indices])
        candidates.append({'k': k, 'silhouette': score, 'inertia': candidate.inertia_})
        print(f'Meteorites K={k}, silhouette={score:.3f}', flush=True)
    chosen = max(candidates, key=lambda row: row['silhouette'])['k']
    clusterer = Pipeline([('features', MeteoriteFeatures()), ('scale', StandardScaler()),
        ('model', KMeans(n_clusters=chosen, n_init=20, random_state=42))]).fit(meteorites)
    labels = clusterer.predict(meteorites)
    profiles = meteorites[METEORITE].assign(Cluster=labels).groupby('Cluster').median()
    profiles['Records'] = pd.Series(labels).value_counts().sort_index()
    profiles.to_csv(OUTPUT / 'meteorite_profiles.csv')
    raw_count = len(pd.read_csv(ROOT / 'data' / TRACKS['meteorites']['file']))
    reports['meteorites'] = save('meteorites', clusterer, {'raw_rows': raw_count, 'rows': len(meteorites),
        'excluded_rows': raw_count - len(meteorites), 'selected_k': chosen, 'candidates': candidates,
        'silhouette': silhouette_score(scaled[indices], labels[indices]),
        'davies_bouldin': davies_bouldin_score(scaled, labels), 'analysis_year': 2026})
    reports['provenance'] = {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                             'seed': 42, 'supervised_refit_on_holdout': False}
    (OUTPUT / 'metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2), flush=True)

if __name__ == '__main__':
    train()
