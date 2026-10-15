import json
import joblib
import pandas as pd
import streamlit as st
from ml_core import ROOT, FEATURES, NUMERIC, predict

st.set_page_config(page_title='Telco Customer Insights', page_icon='📊', layout='wide')
st.title('Telco Customer Insights')
st.caption('Explore customer value, churn risk, and customer segments.')
MODELS = ROOT / 'models' / 'app'

@st.cache_resource
def load_bundle(track):
    return joblib.load(MODELS / f'{track}.joblib')

@st.cache_data
def sample_data():
    return pd.read_csv(ROOT / 'data' / 'TelcoCustomerChurn.csv')[FEATURES]

track_name = st.sidebar.radio('Track', ['Regression · CLTV', 'Classification · Churn', 'Clustering · Segments'])
track = {'Regression · CLTV': 'cltv', 'Classification · Churn': 'churn', 'Clustering · Segments': 'segments'}[track_name]
try:
    bundle = load_bundle(track)
    metrics = json.loads((MODELS / 'metrics.json').read_text())
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(f'Model files are unavailable. Run python train_models.py with the pinned requirements. {exc}')
    st.stop()

if track == 'cltv':
    st.subheader('Estimate customer lifetime value')
    st.info('CLTV is the dataset’s supplied value; its forecast horizon and currency are unspecified. Estimates have substantial error; review the held-out metrics below.')
elif track == 'churn':
    st.subheader('Estimate churn risk')
    st.caption('Historical churn classification. Scores are not calibrated future churn probabilities.')
else:
    st.subheader('Assign customer segments')
    st.caption('Segments describe similarities in service use and spending. Segment numbers are identifiers, not rankings.')

threshold = st.slider('Churn decision threshold', 0.05, 0.95, 0.5, 0.05) if track == 'churn' else 0.5
mode = st.radio('Input method', ['Example customers', 'Single customer', 'Upload CSV'], horizontal=True)
samples = sample_data()
frame = None
if mode == 'Example customers':
    frame = samples.head(10).copy()
elif mode == 'Upload CSV':
    st.download_button('Download input template', samples.head(3).to_csv(index=False), 'customer_template.csv', 'text/csv')
    upload = st.file_uploader('Customer CSV (maximum 10,000 rows)', type=['csv'])
    if upload is not None:
        try:
            frame = pd.read_csv(upload, nrows=10001)
            if len(frame) > 10000:
                st.error('Please upload at most 10,000 rows.')
                frame = None
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            st.error(f'Cannot read this CSV: {exc}')
else:
    with st.form('customer'):
        values = {}
        columns = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with columns[i % 3]:
                if feature in NUMERIC:
                    values[feature] = st.number_input(feature, min_value=0.0, value=float(samples[feature].median()))
                else:
                    choices = sorted(samples[feature].dropna().unique().tolist())
                    if feature in ['Offer', 'InternetType']:
                        choices = sorted(set(choices + ['No Offer' if feature == 'Offer' else 'No Internet']))
                    values[feature] = st.selectbox(feature, choices)
        if st.form_submit_button('Predict'):
            frame = pd.DataFrame([values])

if frame is not None:
    try:
        result = predict(bundle, frame, threshold)
        st.success(f'Processed {len(result):,} customer(s).')
        st.dataframe(result, hide_index=True)
        prediction_col = {'cltv': 'PredictedCLTV', 'churn': 'ChurnProbability', 'segments': 'Segment'}[track]
        st.bar_chart(result[prediction_col].value_counts().sort_index()) if track == 'segments' else st.line_chart(result[prediction_col])
        st.download_button('Download predictions', result.to_csv(index=False), f'{track}_predictions.csv', 'text/csv')
    except (ValueError, TypeError) as exc:
        st.error(str(exc))

if track == 'segments':
    st.subheader('Segment profiles')
    st.dataframe(pd.read_csv(MODELS / 'segment_profiles.csv'), hide_index=True)
with st.expander('Model evaluation and input notes'):
    st.json(metrics[track])
    st.write('Numeric blanks use training medians. Missing Offer and InternetType mean No Offer and No Internet. Other category blanks use the training mode; unseen categories are accepted. Extra columns are retained in downloads but excluded from predictions.')
    st.write('Outcome fields, ChurnScore, CLTV, identifiers, and SatisfactionScore are excluded from model inputs. Supervised preprocessing is fitted only on training rows. Segmentation uses the complete customer population.')
