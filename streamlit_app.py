import json
import altair as alt
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from ml_core import ROOT, FEATURES, NUMERIC, predict
from ui_helpers import CSS, HELP, INTEGERS, SEGMENTS, label as display_label

st.set_page_config(page_title='Telco Customer Insights', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
st.markdown(CSS, unsafe_allow_html=True)
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

track_name = st.sidebar.radio('Choose your analysis', ['Regression · CLTV', 'Classification · Churn', 'Clustering · Segments'])
track = {'Regression · CLTV': 'cltv', 'Classification · Churn': 'churn', 'Clustering · Segments': 'segments'}[track_name]
titles = {
    'cltv': ('Customer lifetime value', 'Estimate customer value from account history, billing, and service usage. Compare customers and export your results.'),
    'churn': ('Customer churn analysis', 'Estimate historical churn risk, explore which customers cross your chosen threshold, and prepare a review list.'),
    'segments': ('Customer segmentation', 'Group customers by spending, tenure, and service usage. Understand each group before exploring individual assignments.')
}
heading, description = titles[track]
st.markdown(f'<div class="hero"><div class="eyebrow">TELCO INSIGHTS / CUSTOMER INTELLIGENCE</div><h1>{heading}</h1><p>{description}</p></div>', unsafe_allow_html=True)
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

analysis_tab, model_tab, guide_tab = st.tabs(['Analyze customers', 'Model performance', 'User guide'])
with analysis_tab:
    st.markdown('<div class="section-title"><span class="step">1</span>Add your customers</div>', unsafe_allow_html=True)
    st.caption('Start with a sample, enter one customer, or upload a CSV. Results appear below your inputs.')
    threshold = 0.5
    if track == 'churn':
        with st.expander('Adjust churn decision threshold', expanded=False):
            st.caption('A score at or above this threshold is flagged Yes. Lowering it flags more customers; it does not change the model score.')
            threshold = st.slider('Flag customers at or above', 0.05, 0.95, 0.5, 0.05, format='%.2f')
    mode = st.radio('Input method', ['Example customers', 'Single customer', 'Upload CSV'], horizontal=True)
    samples = sample_data()
    frame = None
    values = {}
    if mode == 'Example customers':
        sample_size = st.select_slider('Customers to explore', options=[10, 25, 50, 100], value=25)
        frame = samples.sample(sample_size, random_state=42).reset_index(drop=True)
        st.caption('A reproducible sample from the included dataset. Adjust the size to explore a broader portfolio.')
    elif mode == 'Upload CSV':
        st.caption('1. Download the template.  2. Keep the column headers and replace the example rows.  3. Upload your completed file.')
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
        st.caption('Fields start with one example customer. Replace them with the actual account details before submitting.')
        default_customer = samples.iloc[0].fillna({'Offer': 'No Offer', 'InternetType': 'No Internet'})
        with st.form('customer', clear_on_submit=False):
            values = {}
            groups = {
                'Customer profile': ['Age', 'Gender', 'SeniorCitizen', 'Married', 'NumberofDependents', 'Number_of_Referrals'],
                'Account & billing': ['TenureinMonths', 'Offer', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharge', 'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges', 'TotalLongDistanceCharges', 'AvgMonthlyLongDistanceCharges'],
                'Services & usage': [f for f in FEATURES if f not in ['Age', 'Gender', 'SeniorCitizen', 'Married', 'NumberofDependents', 'Number_of_Referrals', 'TenureinMonths', 'Offer', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharge', 'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges', 'TotalLongDistanceCharges', 'AvgMonthlyLongDistanceCharges']]
            }
            st.caption('Review the three sections below, then run the analysis. All sections start with example values.')
            form_tabs = st.tabs(list(groups))
            for section, (group, fields) in zip(form_tabs, groups.items()):
                with section:
                    columns = st.columns(3)
                    for i, feature in enumerate(fields):
                        with columns[i % 3]:
                            if feature in NUMERIC:
                                values[feature] = st.number_input(display_label(feature), min_value=0 if feature in INTEGERS else 0.0,
                                                                value=int(default_customer[feature]) if feature in INTEGERS else float(default_customer[feature]),
                                                                step=1 if feature in INTEGERS else 0.01,
                                                                help=HELP.get(feature), key=f'{track}_{feature}')
                            else:
                                choices = sorted(samples[feature].dropna().unique().tolist())
                                if feature in ['Offer', 'InternetType']:
                                    choices = sorted(set(choices + ['No Offer' if feature == 'Offer' else 'No Internet']))
                                values[feature] = st.selectbox(display_label(feature), choices, index=choices.index(default_customer[feature]),
                                                             help=HELP.get(feature), key=f'{track}_{feature}')
            if st.form_submit_button('Run customer analysis', type='primary', use_container_width=True):
                st.session_state[f'submitted_{track}'] = pd.DataFrame([values])
        frame = st.session_state.get(f'submitted_{track}')
        if frame is not None:
            st.caption('Showing the last submitted customer. After editing fields, select Run customer analysis to refresh the result.')

    if frame is not None:
        try:
            result = predict(bundle, frame, threshold)
            st.success(f'Processed {len(result):,} customer(s).')
            st.markdown('<div class="section-title"><span class="step">2</span>Understand your results</div>', unsafe_allow_html=True)
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
            if len(result) == 1:
                if track == 'cltv':
                    message = f"Estimated lifetime value: <b>{result.PredictedCLTV.iloc[0]:,.0f}</b> in the dataset's value units. The model's average absolute holdout error is {metrics['cltv']['mae']:,.0f}; this is a broad estimate, not a guaranteed value."
                elif track == 'churn':
                    flag = result.PredictedChurn.iloc[0] == 'Yes'
                    message = f"This customer's churn score is <b>{result.ChurnProbability.iloc[0]:.1%}</b>. At your {threshold:.0%} threshold, the customer is <b>{'flagged for review' if flag else 'below the review threshold'}</b>. This is a model score, not a certainty that the customer will leave."
                else:
                    segment = int(result.Segment.iloc[0])
                    message = f"This customer belongs to <b>Segment {segment}: {SEGMENTS[segment]}</b>. Compare the group profile below to understand its typical characteristics. Individual customers can differ from the group average."
                st.markdown(f'<div class="result-note">{message}</div>', unsafe_allow_html=True)
            overview_tab, records_tab = st.tabs(['Visual summary', 'Customer records & export'])
            with overview_tab:
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
                        st.markdown('#### ' + (label if len(result) == 1 else label + ' distribution'))
                        if len(result) == 1:
                            chart = alt.Chart(result[[field]]).mark_bar(color='#238b91', size=48, cornerRadiusEnd=6).encode(
                                x=alt.X(f'{field}:Q', title=label, scale=alt.Scale(domain=[0, 1]) if track == 'churn' else alt.Scale(zero=True)),
                                tooltip=[alt.Tooltip(f'{field}:Q', title=label)])
                        else:
                            # Aggregate before plotting so full 10,000-row uploads stay below Altair's row limit.
                            counts, edges = np.histogram(result[field], bins=15)
                            histogram = pd.DataFrame({'From': edges[:-1], 'To': edges[1:], 'Customers': counts})
                            chart = alt.Chart(histogram).mark_bar(color='#238b91', cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                                x=alt.X('From:Q', title=label, bin='binned'), x2='To:Q',
                                y=alt.Y('Customers:Q', title='Customers'), tooltip=['From', 'To', 'Customers'])
                        if track == 'churn':
                            rule = alt.Chart(pd.DataFrame({'Threshold': [threshold]})).mark_rule(color='#e18a46', strokeDash=[5, 4], size=2).encode(x='Threshold:Q', tooltip=['Threshold'])
                            chart = chart + rule
                    st.altair_chart(chart.properties(height=290).configure_view(strokeWidth=0), use_container_width=True)
                with details:
                    st.markdown('#### Reading this view')
                    st.write({'cltv': 'The chart shows estimated value for your selected customers. For multiple customers, bars count predictions within each value range. Interpret estimates alongside the model’s measured error.',
                              'churn': 'The orange line marks your decision threshold. Customers at or above it receive a Yes label. Change the threshold to explore how the flagged group changes.',
                              'segments': 'Each bar counts customers assigned to a group. Compare the profiles below to understand the spending and service patterns behind each segment.'}[track])
                    st.download_button('↓ Export predictions', result.to_csv(index=False), f'{track}_predictions.csv', 'text/csv', use_container_width=True)
            with records_tab:
                st.markdown('#### Customer records')
                st.caption('Predictions appear first. Download the full CSV to retain the original column names for further analysis.')
                predictions = {'cltv': ['PredictedCLTV'], 'churn': ['ChurnProbability', 'PredictedChurn'], 'segments': ['Segment']}[track]
                ordered = predictions + [c for c in result.columns if c not in predictions]
                st.dataframe(result[ordered], hide_index=True, use_container_width=True,
                             column_config={**{c: st.column_config.Column(display_label(c)) for c in ordered},
                                            'ChurnProbability': st.column_config.ProgressColumn('Churn score (0–1)', min_value=0, max_value=1, format='%.2f'),
                                            'PredictedCLTV': st.column_config.NumberColumn('Predicted CLTV', format='%.0f')})
                st.download_button('Download complete results', result.to_csv(index=False), f'{track}_results.csv', 'text/csv', type='primary')
        except (ValueError, TypeError) as exc:
            st.error(str(exc))

    if track == 'segments':
        st.markdown('<div class="section-title"><span class="step">3</span>Understand the customer groups</div>', unsafe_allow_html=True)
        st.caption('These profiles describe the full source dataset, not just the customers scored above.')
        profiles = pd.read_csv(MODELS / 'segment_profiles.csv')
        profiles['Group description'] = profiles.Segment.map(SEGMENTS)
        profile_chart = alt.Chart(profiles).mark_circle(opacity=0.85).encode(
            x=alt.X('TenureinMonths:Q', title='Average tenure (months)'),
            y=alt.Y('MonthlyCharge:Q', title='Average monthly charge'),
            size=alt.Size('Customers:Q', scale=alt.Scale(range=[250, 1800])),
            color=alt.Color('Segment:N', scale=alt.Scale(scheme='tealblues')),
            tooltip=['Segment', 'Group description', 'Customers', 'TenureinMonths', 'MonthlyCharge', 'ServiceCount'])
        st.altair_chart(profile_chart.properties(height=300).configure_view(strokeWidth=0), use_container_width=True)
        st.caption('Each bubble is a segment. Its size represents the number of customers in the full training population.')
        st.dataframe(profiles.rename(columns={c: display_label(c) for c in profiles.columns}), hide_index=True, use_container_width=True)
with model_tab:
    st.subheader('How reliable is this model?')
    st.caption('Evaluation describes the fitted model. These numbers do not measure confidence for an individual customer.')
    st.markdown('**Method:** ' + {'cltv': 'Gradient Boosting Regressor', 'churn': 'Gradient Boosting Classifier', 'segments': 'K-Means with scaled customer features'}[track])
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
    if track == 'cltv':
        st.markdown('**What the numbers mean**')
        st.write('R-squared measures how much variation in customer value the model explains on held-out customers. A value of 0.205 means about 20.5% of the variation is explained. The model has limited predictive strength.')
        st.write('Typical absolute error is the average gap between a prediction and the recorded value. It is not a prediction interval. Currency and the value forecast horizon are unspecified in the source data.')
        st.info('Use this track to demonstrate and compare estimates. Avoid treating a single prediction as an exact customer valuation.')
    elif track == 'churn':
        st.markdown('**What the numbers mean**')
        st.write('ROC-AUC measures how well the model ranks historical churners above other customers. Precision measures how many flagged customers actually churned. Recall measures how many actual churners were flagged.')
        st.caption('Precision and recall shown here use the default 0.50 threshold on the held-out test set. They do not update when you change the scoring threshold.')
        matrix = pd.DataFrame(metrics['churn']['confusion_matrix'], index=['Actually stayed', 'Actually churned'], columns=['Predicted stay', 'Predicted churn'])
        st.markdown('#### Test-set outcomes at threshold 0.50')
        st.dataframe(matrix, use_container_width=True)
    else:
        st.markdown('**What the numbers mean**')
        st.write('Silhouette compares how similar customers are within their group versus other groups. Higher is better, with a maximum of 1. The observed score suggests overlapping groups, rather than completely separate customer types.')
        st.write('Two to six groups were compared using the same sampled customers. Six had the highest sampled silhouette score. Clustering is descriptive; it does not predict a verified customer label.')
        st.dataframe(pd.DataFrame(metrics['segments']['candidates']).rename(columns={'k': 'Number of groups', 'silhouette': 'Sample silhouette score'}), hide_index=True)
    with st.expander('Technical details and training report'):
        st.json(metrics[track])
        st.write('Supervised preprocessing is fitted only on training rows. Outcome fields, identifiers, CLTV, ChurnScore, and SatisfactionScore are excluded from model inputs. Segmentation is fitted on the complete customer population.')

with guide_tab:
    st.subheader('From customer details to a useful result')
    cards = [
        ('01 / Choose the question', 'Customer value estimates CLTV. Churn analysis flags customers for review. Segmentation assigns customers to groups with similar usage and spending.'),
        ('02 / Provide customer data', 'Use a reproducible sample to explore, enter a single account, or download the CSV template and replace its example rows with your data.'),
        ('03 / Interpret and export', 'Read the summary, inspect the charts, and open Customer records & export for individual predictions. Review Model performance before presenting conclusions.')
    ]
    for col, (title, body) in zip(st.columns(3), cards):
        col.markdown(f'<div class="guide-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)
    st.write('')
    with st.expander('What does each analysis tell me?', expanded=True):
        st.markdown('**Customer lifetime value (CLTV)**: a numerical estimate of the value recorded for a customer in this dataset.\n\n**Churn score**: a score from 0 to 1 used to classify historical churn risk. It is not a calibrated future probability.\n\n**Customer segment**: a group based on service usage, spending, tenure, and referrals. Group numbers indicate identity, not quality or rank.')
    with st.expander('How should I prepare a CSV?'):
        st.write('Download the template from Upload CSV. Keep the headers exactly as supplied. Each row represents one customer. The limit is 10,000 rows and 10 MB. Extra columns are retained in the exported result but ignored by the model.')
        st.write('Numeric blanks use the training median. Missing Offer and InternetType mean No Offer and No Internet. Other missing categories use the training mode. Unseen categories are accepted. Missing columns, invalid numeric text, infinities, and negative numbers are rejected.')
    with st.expander('Why do the graphs and model performance use different customer counts?'):
        st.write('Analysis charts summarize only the customers you selected or uploaded. Model performance comes from the saved evaluation. Customer group profiles describe all 7,043 customers in the source dataset.')
    st.markdown('#### Input reference')
    reference = pd.DataFrame([{'CSV column': f, 'Display label': display_label(f), 'Type': 'Number' if f in NUMERIC else 'Category', 'Meaning': HELP.get(f, 'Use the customer account value for this field.')} for f in FEATURES])
    st.dataframe(reference, hide_index=True, use_container_width=True)

st.markdown('<div class="footer">TELCO INSIGHTS / Customer intelligence workspace</div>', unsafe_allow_html=True)
