import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from ml_core import ROOT, FEATURES, CustomerFeatures, predict, validate

@pytest.fixture
def customers():
    return pd.read_csv(ROOT / 'data' / 'TelcoCustomerChurn.csv').head(5)

@pytest.mark.parametrize('track', ['cltv', 'churn', 'segments'])
def test_saved_pipeline_batch_single_and_missing(track, customers):
    bundle = joblib.load(ROOT / 'models' / 'app' / f'{track}.joblib')
    batch = predict(bundle, customers)
    col = {'cltv': 'PredictedCLTV', 'churn': 'ChurnProbability', 'segments': 'Segment'}[track]
    assert np.isfinite(batch[col]).all()
    assert batch[col].iloc[0] == pytest.approx(predict(bundle, customers.head(1))[col].iloc[0])
    changed = customers.copy()
    changed['ChurnLabel'] = 'arbitrary outcome'
    changed['CLTV'] = -999
    assert np.allclose(predict(bundle, changed)[col], batch[col])
    changed.loc[0, 'Age'] = np.nan
    changed.loc[0, 'Contract'] = 'Unknown category'
    assert np.isfinite(predict(bundle, changed)[col]).all()
    if track == 'churn':
        assert batch[col].between(0, 1).all()
        assert (predict(bundle, customers, threshold=0)['PredictedChurn'] == 'Yes').all()

def test_validation_and_engineering(customers):
    with pytest.raises(ValueError, match='Missing columns'):
        validate(customers.drop(columns='Age'))
    with pytest.raises(ValueError, match='at least one'):
        validate(customers.head(0))
    bad = customers.copy()
    bad['Age'] = 'invalid'
    with pytest.raises(ValueError, match='finite numbers'):
        validate(bad)
    bad['Age'] = -1
    with pytest.raises(ValueError, match='negative'):
        validate(bad)
    customers['TenureinMonths'] = 0
    transformed = CustomerFeatures().fit_transform(customers)
    assert transformed.AvgChargePerTenureMonth.isna().all()
    assert transformed.ServiceCount.between(0, 8).all()

def test_regression_partition_integrity():
    train = pd.read_csv(ROOT / 'data' / 'regression_train.csv')
    test = pd.read_csv(ROOT / 'data' / 'regression_test.csv')
    source = pd.read_csv(ROOT / 'data' / 'regression_dataset.csv')
    assert len(train) + len(test) == len(source)
    assert not set(pd.util.hash_pandas_object(train, index=False)) & set(pd.util.hash_pandas_object(test, index=False))

@pytest.mark.parametrize('track', ['Regression · CLTV', 'Classification · Churn', 'Clustering · Segments'])
def test_streamlit_tracks_and_form(track):
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value(track).run()
    assert not app.exception
    assert len(app.dataframe) >= 1
    app.main.radio[0].set_value('Single customer').run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.success) == 1
    app.main.radio[0].set_value('Upload CSV').run()
    assert not app.exception

def test_churn_result_persists_when_threshold_changes():
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Classification · Churn').run()
    app.main.radio[0].set_value('Single customer').run()
    app.button[0].click().run()
    assert len(app.success) == 1
    app.slider[0].set_value(0.9).run()
    assert not app.exception
    assert len(app.success) == 1
    assert 'submitted_churn' in app.session_state
