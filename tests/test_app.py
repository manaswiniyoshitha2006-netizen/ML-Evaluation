import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from ml_core import ROOT
from track_data import (TRACKS, examples, infer, validate_credit, validate_meteorites,
                        CreditFeatures, MeteoriteFeatures, clean_credit, clean_meteorites)

@pytest.mark.parametrize('track', list(TRACKS))
def test_saved_pipeline_matches_its_dataset(track):
    data = examples(track).head(5)
    bundle = joblib.load(ROOT / 'models' / 'app' / f'{track}.joblib')
    assert bundle['track'] == track
    result = infer(bundle, data)
    field = TRACKS[track]['output']
    assert np.isfinite(result[field]).all()
    assert infer(bundle, data.head(1))[field].iloc[0] == pytest.approx(result[field].iloc[0])
    changed = data.assign(Y=999, CLTV=999, ChurnLabel='irrelevant', id=999)
    assert np.allclose(infer(bundle, changed)[field], result[field])
    other = examples('credit' if track != 'credit' else 'meteorites').head(2)
    with pytest.raises(ValueError, match='Missing columns'):
        infer(bundle, other)

@pytest.mark.parametrize('track', list(TRACKS))
def test_csv_template_roundtrip(track):
    from io import StringIO
    data = examples(track).head(3)
    loaded = pd.read_csv(StringIO(data.to_csv(index=False)))
    bundle = joblib.load(ROOT / 'models' / 'app' / f'{track}.joblib')
    assert len(infer(bundle, loaded)) == 3


def test_credit_cleaning_features_and_validation():
    data = clean_credit()
    assert len(data) == 29965
    assert not data.duplicated().any()
    assert data.Y.isin([0, 1]).all()
    sample = data.head(3).copy()
    sample['X6'] = -2
    sample['X12'] = -100
    validate_credit(sample)
    features = CreditFeatures().fit(sample)
    assert 'Avg_Bill_Amount' in features.transform(sample)
    sample['X1'] = np.nan
    validate_credit(sample)
    sample['X2'] = 12
    with pytest.raises(ValueError, match='category'):
        validate_credit(sample)


def test_meteorite_cleaning_and_limits():
    data = clean_meteorites()
    assert len(data) == 31929
    assert np.isfinite(MeteoriteFeatures().fit_transform(data)).all().all()
    sample = data.head(1).copy()
    sample['reclat'] = -40
    sample['reclong'] = -120
    validate_meteorites(sample)
    sample[['reclat', 'reclong']] = 0
    with pytest.raises(ValueError, match='0, 0'):
        validate_meteorites(sample)
    sample['reclat'] = 100
    with pytest.raises(ValueError, match='Latitude'):
        validate_meteorites(sample)
    sample['reclat'] = 40
    sample['year'] = 2027
    with pytest.raises(ValueError, match='Year'):
        validate_meteorites(sample)
    sample['year'] = np.nan
    with pytest.raises(ValueError, match='required'):
        validate_meteorites(sample)

@pytest.mark.parametrize('track', list(TRACKS))
def test_app_track_examples_form_upload_and_dataset_labels(track):
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value(TRACKS[track]['nav']).run()
    assert not app.exception
    assert len(app.success) == 1
    assert TRACKS[track]['file'] in app.success[0].value
    app.main.radio[0].set_value('Single record').run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.success) == 1
    app.main.radio[0].set_value('Upload CSV').run()
    assert not app.exception


def test_credit_threshold_preserves_submitted_record():
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value(TRACKS['credit']['nav']).run()
    app.main.radio[0].set_value('Single record').run()
    app.button[0].click().run()
    app.slider[0].set_value(.9).run()
    assert not app.exception
    assert len(app.success) == 1
    assert 'submitted_credit' in app.session_state


def test_provenance_datasets_are_distinct():
    import json
    reports = json.loads((ROOT / 'models' / 'app' / 'metrics.json').read_text(encoding='utf-8'))
    assert len({reports[t]['data_sha256'] for t in TRACKS}) == 3
    for track in TRACKS:
        assert reports[track]['dataset'] == TRACKS[track]['file']
