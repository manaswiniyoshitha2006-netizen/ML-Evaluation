"""Shared raw-input contract and serializable feature engineering."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parent
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtectionPlan', 'PremiumTechSupport',
            'StreamingTV', 'StreamingMovies', 'StreamingMusic', 'UnlimitedData']
NUMERIC = ['Age', 'NumberofDependents', 'Number_of_Referrals', 'TenureinMonths',
           'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload', 'MonthlyCharge',
           'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges', 'TotalLongDistanceCharges']
CATEGORICAL = ['Gender', 'SeniorCitizen', 'Married', 'Offer', 'PhoneService',
               'MultipleLines', 'InternetService', 'InternetType', *SERVICES,
               'Contract', 'PaperlessBilling', 'PaymentMethod']
FEATURES = NUMERIC + CATEGORICAL
SEGMENT_FEATURES = ['TenureinMonths', 'MonthlyCharge', 'AvgMonthlyGBDownload',
                    'Number_of_Referrals', 'ServiceCount']

def validate(frame):
    if frame.empty:
        raise ValueError('Provide at least one customer row.')
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    result = frame[FEATURES].copy()
    for col in NUMERIC:
        values = pd.to_numeric(result[col], errors='coerce')
        if (result[col].notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f'{col} must contain finite numbers or blank values.')
        if (values < 0).any():
            raise ValueError(f'{col} cannot be negative.')
        result[col] = values
    for col in CATEGORICAL:
        result[col] = result[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    return result

class CustomerFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate(X)
        return self

    def transform(self, X):
        frame = validate(X)
        frame['Offer'] = frame['Offer'].fillna('No Offer')
        frame['InternetType'] = frame['InternetType'].fillna('No Internet')
        frame['ServiceCount'] = frame[SERVICES].eq('Yes').sum(axis=1)
        frame['AvgChargePerTenureMonth'] = frame.TotalCharges / frame.TenureinMonths.replace(0, np.nan)
        return frame

def predict(bundle, frame, threshold=0.5):
    clean = validate(frame)
    pipe = bundle['pipeline']
    result = frame.copy()
    if bundle['track'] == 'churn':
        probability = pipe.predict_proba(clean)[:, list(pipe.classes_).index(1)]
        result['ChurnProbability'] = probability
        result['PredictedChurn'] = np.where(probability >= threshold, 'Yes', 'No')
    else:
        result['PredictedCLTV' if bundle['track'] == 'cltv' else 'Segment'] = pipe.predict(clean)
    return result
