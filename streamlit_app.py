import json
import altair as alt
import joblib
import pandas as pd
import streamlit as st
from ml_core import ROOT, FEATURES, NUMERIC, predict

st.set_page_config(page_title='Telco Customer Insights', page_icon='📊', layout='wide')
st.markdown("""
<style>
.stApp {background: #f4f7fb; color: #182b44;}
[data-testid="stSidebar"] {background: #101f35;}
[data-testid="stSidebar"] * {color: #e7eef8;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: #a7b9d2;}
.block-container {max-width: 1360px; padding-top: 2rem; padding-bottom: 3rem;}
h1,h2,h3 {letter-spacing: -.035em;}
[data-testid="stMetric"] {background: white; border: 1px solid #e2e9f2; border-radius: 16px; padding: 20px; box-shadow: 0 4px 20px #172e4910;}
[data-testid="stMetricLabel"] {color: #60718a;}
[data-testid="stMetricValue"] {color: #172e49; font-weight: 700;}
[data-testid="stForm"], [data-testid="stExpander"] {background: white; border: 1px solid #e2e9f2; border-radius: 16px;}
.stButton button, .stDownloadButton button {border-radius: 10px; font-weight: 600;}
.hero {padding: 32px 36px; border-radius: 22px; background: linear-gradient(115deg,#11243e,#174c65); color: white; margin-bottom: 24px;}
.hero h1 {color: white; font-size: 2.5rem; margin: 8px 0; padding: 0;}
.hero p {color: #c6d7e9; max-width: 750px; margin: 0; font-size: 1.02rem;}
.eyebrow {color: #70e0cb; font-size: .72rem; font-weight: 700; letter-spacing: .17em; text-transform: uppercase;}
.section-note {color: #697c94; font-size: .9rem; margin-top: -8px; margin-bottom: 20px;}
.footer {border-top: 1px solid #dce5ef; color: #708099; padding-top: 20px; margin-top: 32px; font-size: .8rem;}
@media (max-width: 640px) {.hero {padding: 24px;} .hero h1 {font-size: 1.9rem;} .block-container {padding-top: 1rem;}}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown('### ◈ TELCO INSIGHTS')
st.sidebar.caption('CUSTOMER INTELLIGENCE WORKSPACE')
st.sidebar.divider()
MODELS = ROOT / 'models' / 'app'

@st.cache_resource
def load_bundle(track):
    return joblib.load(MODELS / f'{track}.joblib')

@st.cache_data
def sample_data():
    return pd.read_csv(ROOT / 'data' / 'TelcoCustomerChurn.csv')[FEATURES]

track_name = st.sidebar.radio('Track', ['Regression · CLTV', 'Classification · Churn', 'Clustering · Segments'])
track = {'Regression · CLTV': 'cltv', 'Classification · Churn': 'churn', 'Clustering · Segments': 'segments'}[track_name]
titles = {
    'cltv': ('Customer value, made clearer.', 'Estimate lifetime value and discover where your customer portfolio stands.'),
    'churn': ('Understand risk. Plan retention.', 'Explore churn signals and identify customers who may need a closer look.'),
    'segments': ('Find the patterns in your customers.', 'Turn service usage and spending into meaningful customer groups.')
}
heading, description = titles[track]
st.markdown(f'<div class="hero"><div class="eyebrow">Customer intelligence / {track}</div><h1>{heading}</h1><p>{description}</p></div>', unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown('**Your workflow**')
st.sidebar.caption('01  Choose an analysis\n\n02  Add customer details\n\n03  Explore and export results')
st.sidebar.divider()
st.sidebar.caption('Built on 7,043 Telco customers.\nModels run locally on the server.')
try:
    bundle = load_bundle(track)
    metrics = json.loads((MODELS / 'metrics.json').read_text())
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(f'Model files are unavailable. Run python train_models.py with the pinned requirements. {exc}')
    st.stop()

evaluation = metrics[track]
if track == 'cltv':
    stats = [('Holdout R²', f"{evaluation['r2']:.3f}"), ('Typical absolute error', f"{evaluation['mae']:,.0f}"), ('Training customers', f"{evaluation['train_rows']:,}")]
elif track == 'churn':
    stats = [('Holdout ROC-AUC', f"{evaluation['roc_auc']:.3f}"), ('Holdout precision', f"{evaluation['precision']:.1%}"), ('Holdout recall', f"{evaluation['recall']:.1%}")]
else:
    stats = [('Customer groups', str(evaluation['selected_k'])), ('Customers analyzed', f"{evaluation['rows']:,}"), ('Sample silhouette', f"{max(c['silhouette'] for c in evaluation['candidates']):.3f}")]
for column, (label, value) in zip(st.columns(3), stats):
    column.metric(label, value)
st.write('')
st.subheader('Start your analysis')
st.caption('Try a sample, explore one customer, or bring your own customer list.')

threshold = st.slider('Churn decision threshold', 0.05, 0.95, 0.5, 0.05) if track == 'churn' else 0.5
mode = st.radio('Input method', ['Example customers', 'Single customer', 'Upload CSV'], horizontal=True)
samples = sample_data()
frame = None
if mode == 'Example customers':
    sample_size = st.select_slider('Customers to explore', options=[10, 25, 50, 100], value=25)
    frame = samples.sample(sample_size, random_state=42).reset_index(drop=True)
    st.caption('A reproducible sample from the included dataset. Adjust the size to explore a broader portfolio.')
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
        groups = {
            'Customer profile': ['Age', 'Gender', 'SeniorCitizen', 'Married', 'NumberofDependents', 'Number_of_Referrals'],
            'Account & billing': ['TenureinMonths', 'Offer', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharge', 'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges', 'TotalLongDistanceCharges', 'AvgMonthlyLongDistanceCharges'],
            'Services & usage': [f for f in FEATURES if f not in ['Age', 'Gender', 'SeniorCitizen', 'Married', 'NumberofDependents', 'Number_of_Referrals', 'TenureinMonths', 'Offer', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharge', 'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges', 'TotalLongDistanceCharges', 'AvgMonthlyLongDistanceCharges']]
        }
        for group, fields in groups.items():
            st.markdown(f'#### {group}')
            columns = st.columns(3)
            for i, feature in enumerate(fields):
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
        st.subheader('Your customer snapshot')
        summary = st.columns(3)
        summary[0].metric('Customers scored', f'{len(result):,}')
        if track == 'cltv':
            summary[1].metric('Average predicted CLTV', f"{result.PredictedCLTV.mean():,.0f}")
            summary[2].metric('Highest predicted CLTV', f"{result.PredictedCLTV.max():,.0f}")
            st.caption('CLTV currency and forecast horizon are unspecified in the source. Consider the holdout error when interpreting estimates.')
        elif track == 'churn':
            summary[1].metric('Average churn score', f"{result.ChurnProbability.mean():.1%}")
            summary[2].metric('Above chosen threshold', str(int((result.PredictedChurn == 'Yes').sum())))
            st.caption('Historical classification scores are not calibrated future probabilities. The threshold controls the Yes/No decision.')
        else:
            summary[1].metric('Groups represented', str(result.Segment.nunique()))
            summary[2].metric('Largest group', f"Segment {result.Segment.mode().iloc[0]}")
            st.caption('Segment IDs describe groups; they are not rankings.')
        st.write('')
        chart_area, details = st.columns([2, 1])
        with chart_area:
            if track == 'segments':
                st.markdown('#### Customers by segment')
                counts = result.groupby('Segment').size().reset_index(name='Customers')
                chart = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X('Segment:O', title='Customer segment'), y='Customers:Q',
                    color=alt.Color('Segment:O', scale=alt.Scale(scheme='tealblues'), legend=None), tooltip=['Segment', 'Customers'])
            else:
                field = 'PredictedCLTV' if track == 'cltv' else 'ChurnProbability'
                label = 'Predicted customer value' if track == 'cltv' else 'Churn score'
                st.markdown('#### ' + ('Customer value distribution' if track == 'cltv' else 'Churn score distribution'))
                chart = alt.Chart(result[[field]]).mark_bar(color='#238b91', cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                    x=alt.X(f'{field}:Q', bin=alt.Bin(maxbins=15), title=label),
                    y=alt.Y('count():Q', title='Customers'), tooltip=[alt.Tooltip('count():Q', title='Customers')])
                if track == 'churn':
                    rule = alt.Chart(pd.DataFrame({'Threshold': [threshold]})).mark_rule(color='#e18a46', strokeDash=[5, 4], size=2).encode(x='Threshold:Q', tooltip=['Threshold'])
                    chart = chart + rule
            st.altair_chart(chart.properties(height=290).configure_view(strokeWidth=0), use_container_width=True)
        with details:
            st.markdown('#### Reading this view')
            st.write({'cltv': 'The distribution shows how predicted value varies across the customers you selected. Use it to compare the portfolio, alongside the model’s measured error.',
                      'churn': 'The orange line marks your decision threshold. Customers at or above it receive a Yes label. Change the threshold to explore how the flagged group changes.',
                      'segments': 'Each bar counts customers assigned to a group. Compare the profiles below to understand the spending and service patterns behind each segment.'}[track])
            st.download_button('↓ Export predictions', result.to_csv(index=False), f'{track}_predictions.csv', 'text/csv', use_container_width=True)
        st.markdown('#### Customer results')
        predictions = {'cltv': ['PredictedCLTV'], 'churn': ['ChurnProbability', 'PredictedChurn'], 'segments': ['Segment']}[track]
        ordered = predictions + [c for c in result.columns if c not in predictions]
        st.dataframe(result[ordered], hide_index=True, use_container_width=True,
                     column_config={'ChurnProbability': st.column_config.ProgressColumn('Churn score', min_value=0, max_value=1, format='%.2f'),
                                    'PredictedCLTV': st.column_config.NumberColumn('Predicted CLTV', format='%.0f')})
    except (ValueError, TypeError) as exc:
        st.error(str(exc))

if track == 'segments':
    st.subheader('Segment profiles')
    profiles = pd.read_csv(MODELS / 'segment_profiles.csv')
    profile_chart = alt.Chart(profiles).mark_circle(opacity=0.85).encode(
        x=alt.X('TenureinMonths:Q', title='Average tenure (months)'),
        y=alt.Y('MonthlyCharge:Q', title='Average monthly charge'),
        size=alt.Size('Customers:Q', scale=alt.Scale(range=[250, 1800])),
        color=alt.Color('Segment:N', scale=alt.Scale(scheme='tealblues')),
        tooltip=['Segment', 'Customers', 'TenureinMonths', 'MonthlyCharge', 'ServiceCount'])
    st.altair_chart(profile_chart.properties(height=300).configure_view(strokeWidth=0), use_container_width=True)
    st.caption('Each bubble is a segment. Its size represents the number of customers in the full training population.')
    st.dataframe(profiles, hide_index=True, use_container_width=True)
with st.expander('Model evaluation and input notes'):
    st.json(metrics[track])
    st.write('Numeric blanks use training medians. Missing Offer and InternetType mean No Offer and No Internet. Other category blanks use the training mode; unseen categories are accepted. Extra columns are retained in downloads but excluded from predictions.')
    st.write('Outcome fields, ChurnScore, CLTV, identifiers, and SatisfactionScore are excluded from model inputs. Supervised preprocessing is fitted only on training rows. Segmentation uses the complete customer population.')

st.markdown('<div class="footer">TELCO INSIGHTS &nbsp; / &nbsp; ML capstone workspace · Regression, classification & segmentation</div>', unsafe_allow_html=True)
