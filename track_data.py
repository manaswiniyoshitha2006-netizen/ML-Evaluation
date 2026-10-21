"""Dataset-specific input contracts and fitted feature transformations."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ml_core import ROOT, FEATURES, NUMERIC, validate

CREDIT = [f'X{i}' for i in range(1, 24)]
CREDIT_CATEGORIES = ['X2', 'X3', 'X4']
CREDIT_CONTINUOUS = ['X1', 'X5'] + [f'X{i}' for i in range(12, 24)]
METEORITE = ['mass (g)', 'year', 'reclat', 'reclong']
TRACKS = {
    'cltv': {'nav': 'Regression / Customer value', 'title': 'Customer lifetime value',
             'file': 'TelcoCustomerChurn.csv', 'entity': 'customer', 'features': FEATURES,
             'numeric': NUMERIC, 'output': 'PredictedCLTV', 'method': 'Gradient Boosting Regressor',
             'description': 'Estimate the CLTV value supplied in the Telco dataset from account, billing, and service information.'},
    'credit': {'nav': 'Classification / Credit default', 'title': 'Credit-card default classification',
               'file': 'default of credit card clients.csv', 'entity': 'account', 'features': CREDIT,
               'numeric': CREDIT, 'output': 'DefaultScore', 'method': 'Decision Tree Classifier',
               'description': 'Classify default payment next month using credit limit, repayment history, bills, and payments.'},
    'meteorites': {'nav': 'Clustering / Meteorites', 'title': 'Meteorite landing clusters',
                   'file': 'Meteorite_Landings.csv', 'entity': 'meteorite', 'features': METEORITE,
                   'numeric': METEORITE, 'output': 'Cluster', 'method': 'K-Means',
                   'description': 'Explore landing groups based on mass, year, latitude, and longitude. Assign records to fitted clusters.'},
}

def numeric_frame(frame, features, allow_missing=False):
    if frame.empty:
        raise ValueError('Provide at least one record.')
    missing = sorted(set(features) - set(frame))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    out = frame[features].copy()
    for col in features:
        values = pd.to_numeric(out[col], errors='coerce')
        if (out[col].notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f'{col} must contain finite numbers.')
        if not allow_missing and values.isna().any():
            raise ValueError(f'{col} is required; blank values are not accepted for this track.')
        out[col] = values
    return out

def validate_credit(frame):
    out = numeric_frame(frame, CREDIT, allow_missing=True)
    for col in ['X1', 'X5'] + [f'X{i}' for i in range(18, 24)]:
        if (out[col] < 0).any():
            raise ValueError(f'{col} cannot be negative.')
    # Negative bills represent credit balances; negative repayment-status codes are valid.
    for col in CREDIT_CATEGORIES:
        if not out[col].dropna().isin(range(7) if col == 'X3' else (range(4) if col == 'X4' else [1, 2])).all():
            raise ValueError(f'{col} has an unsupported category code; see the input reference.')
    for col in [f'X{i}' for i in range(6, 12)]:
        if not out[col].dropna().isin(range(-2, 10)).all():
            raise ValueError(f'{col} repayment status must be an integer from -2 to 9.')
    return out

def validate_meteorites(frame):
    out = numeric_frame(frame, METEORITE)
    if (out['mass (g)'] < 0).any():
        raise ValueError('Mass cannot be negative.')
    if not out.year.between(0, 2026).all():
        raise ValueError('Year must be between 0 and the notebook analysis year 2026.')
    if not out.reclat.between(-90, 90).all() or not out.reclong.between(-180, 180).all():
        raise ValueError('Latitude must be -90 to 90 and longitude -180 to 180.')
    if (out.reclat.eq(0) & out.reclong.eq(0)).any():
        raise ValueError('Coordinates (0, 0) are excluded by the notebook location-quality rule.')
    return out

def clean_credit():
    raw = pd.read_csv(ROOT / 'data' / TRACKS['credit']['file'])
    # The first data row is an embedded description/header, not a customer.
    numeric = raw[CREDIT + ['Y']].apply(pd.to_numeric, errors='coerce')
    return numeric.loc[numeric.Y.isin([0, 1])].drop_duplicates().reset_index(drop=True)

def clean_meteorites():
    raw = pd.read_csv(ROOT / 'data' / TRACKS['meteorites']['file'])
    out = raw[METEORITE].apply(pd.to_numeric, errors='coerce')
    mask = (np.isfinite(out).all(axis=1) & out['mass (g)'].ge(0) & out.year.between(0, 2026)
            & out.reclat.between(-90, 90) & out.reclong.between(-180, 180)
            & ~(out.reclat.eq(0) & out.reclong.eq(0)))
    return raw.loc[mask].assign(**{c: out.loc[mask, c] for c in METEORITE}).reset_index(drop=True)

def examples(track):
    if track == 'credit':
        return clean_credit()[CREDIT]
    if track == 'meteorites':
        return clean_meteorites()[METEORITE]
    return pd.read_csv(ROOT / 'data' / TRACKS[track]['file'])[FEATURES]

class CreditFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        out = validate_credit(X)
        self.lower_ = out[CREDIT_CONTINUOUS].quantile(.01)
        self.upper_ = out[CREDIT_CONTINUOUS].quantile(.99)
        return self

    def transform(self, X):
        out = validate_credit(X)
        # Match notebook ordering: averages are engineered before clipping raw columns.
        out['Avg_Bill_Amount'] = out[[f'X{i}' for i in range(12, 18)]].mean(axis=1)
        out['Avg_Payment_Amount'] = out[[f'X{i}' for i in range(18, 24)]].mean(axis=1)
        out[CREDIT_CONTINUOUS] = out[CREDIT_CONTINUOUS].clip(self.lower_, self.upper_, axis=1)
        return out

class MeteoriteFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate_meteorites(X)
        return self

    def transform(self, X):
        out = validate_meteorites(X)
        out['mass (g)'] = np.log1p(out['mass (g)'])
        return out

def infer(bundle, frame, threshold=.5):
    track = bundle['track']
    clean = {'cltv': validate, 'credit': validate_credit, 'meteorites': validate_meteorites}[track](frame)
    result = frame.copy()
    pipe = bundle['pipeline']
    if track == 'credit':
        score = pipe.predict_proba(clean)[:, list(pipe.classes_).index(1)]
        result['DefaultScore'] = score
        result['PredictedDefault'] = np.where(score >= threshold, 'Yes', 'No')
    else:
        result[TRACKS[track]['output']] = pipe.predict(clean)
    return result
