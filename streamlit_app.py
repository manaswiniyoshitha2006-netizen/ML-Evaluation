"""Structured application for three independent ML capstone datasets."""
import json
from html import escape
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from ml_core import ROOT
from track_data import TRACKS, examples, infer
from track_ui import label, help_text, groups, CHOICES, CODES, STATUS, INTEGERS
from ui_helpers import CSS, CHART_PRIMARY, CHART_THRESHOLD, CHART_PALETTE

MODELS = ROOT / 'models' / 'app'
st.set_page_config(page_title='ML Capstone Explorer', page_icon=':bar_chart:', layout='wide', initial_sidebar_state='expanded')
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def load_model(track, modified):
    return joblib.load(MODELS / f'{track}.joblib')

@st.cache_data
def load_examples(track):
    return examples(track)

st.sidebar.markdown('### ML CAPSTONE EXPLORER')
st.sidebar.caption('THREE TASKS / THREE DATASETS')
st.sidebar.divider()
nav = st.sidebar.radio('Choose your analysis', [v['nav'] for v in TRACKS.values()])
track = next(k for k, v in TRACKS.items() if v['nav'] == nav)
config = TRACKS[track]
st.sidebar.divider()
st.sidebar.markdown('**Active dataset**')
st.sidebar.caption(config['file'])
st.sidebar.markdown('**Your workflow**')
st.sidebar.caption('01  Choose a task\n\n02  Provide the matching input\n\n03  Interpret and export')

st.markdown(f'<div class="hero"><div class="eyebrow">ML CAPSTONE / {escape(track.upper())}</div><h1>{config["title"]}</h1><p>{config["description"]}</p></div>', unsafe_allow_html=True)
st.caption(f"Dataset: {config['file']}  |  Method: {config['method']}")
try:
    bundle = load_model(track, (MODELS / f'{track}.joblib').stat().st_mtime_ns)
    report = json.loads((MODELS / 'metrics.json').read_text(encoding='utf-8-sig'))[track]
    samples = load_examples(track)
except (FileNotFoundError, ValueError, ImportError, KeyError) as exc:
    st.error(f'Matching model artifacts are unavailable. Run python train_models.py with the pinned requirements. {exc}')
    st.stop()

analysis, performance, guide = st.tabs(['Run analysis', 'Model performance', 'Dataset & user guide'])
with analysis:
    st.markdown('<div class="section-title"><span class="step">1</span>Provide your input</div>', unsafe_allow_html=True)
    st.caption(f"Every row must describe one {config['entity']}. This track accepts only the {config['file']} input schema.")
    threshold = .5
    if track == 'credit':
        with st.expander('Default classification threshold'):
            st.caption('A default score at or above this threshold receives a Yes label. Performance metrics use 0.50.')
            threshold = st.slider('Flag accounts at or above', .05, .95, .5, .05)
    mode = st.radio('Input method', ['Example records', 'Single record', 'Upload CSV'], horizontal=True)
    frame = None
    if mode == 'Example records':
        count = st.select_slider('Records to explore', [10, 25, 50, 100], value=25)
        frame = samples.sample(count, random_state=42).reset_index(drop=True)
        st.caption(f'A reproducible sample from {len(samples):,} usable source records. Example records are exploratory and may have been used during training.')
    elif mode == 'Upload CSV':
        st.caption('Download this track\'s template, replace the example rows, keep the headers, then upload. Other tracks use different templates.')
        st.download_button('Download input template', samples.head(3).to_csv(index=False), f'{track}_template.csv', 'text/csv')
        upload = st.file_uploader('CSV file / up to 10,000 rows and 10 MB', type=['csv'], key=f'upload_{track}')
        if upload is not None:
            try:
                frame = pd.read_csv(upload, nrows=10001)
                if len(frame) > 10000:
                    st.error('Please upload at most 10,000 rows.')
                    frame = None
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                st.error(f'Cannot read this CSV: {exc}')
    else:
        st.caption('The form starts with an example record. Review each section and submit to update the result.')
        default = samples.iloc[0].fillna({'Offer': 'No Offer', 'InternetType': 'No Internet'})
        with st.form(f'form_{track}'):
            values = {}
            field_groups = groups(track)
            for section, fields in zip(st.tabs(list(field_groups)), field_groups.values()):
                with section:
                    columns = st.columns(2 if track == 'meteorites' else 3)
                    for i, field in enumerate(fields):
                        with columns[i % len(columns)]:
                            key = f'{track}_{field}'
                            if track == 'credit' and field in CHOICES:
                                choices = CHOICES[field]
                                descriptions = CODES.get(field, STATUS)
                                values[field] = st.selectbox(label(track, field), choices, index=choices.index(int(default[field])),
                                    format_func=lambda v, desc=descriptions: f'{v}: {desc[v]}', help=help_text(track, field), key=key)
                            elif field not in config['numeric']:
                                choices = sorted(samples[field].dropna().unique().tolist())
                                if field in ['Offer', 'InternetType']:
                                    choices = sorted(set(choices + ['No Offer' if field == 'Offer' else 'No Internet']))
                                values[field] = st.selectbox(label(track, field), choices, index=choices.index(default[field]), help=help_text(track, field), key=key)
                            else:
                                integer = field in INTEGERS if track == 'cltv' else field in ['X5', 'year']
                                lower, upper = (0, None) if integer else (0., None)
                                if track == 'credit' and field.startswith('X') and 12 <= int(field[1:]) <= 17:
                                    lower = None
                                if track == 'meteorites':
                                    lower, upper = {'mass (g)': (0., None), 'year': (0, 2026), 'reclat': (-90., 90.), 'reclong': (-180., 180.)}[field]
                                values[field] = st.number_input(label(track, field), min_value=lower, max_value=upper,
                                    value=int(default[field]) if integer else float(default[field]),
                                    step=1 if integer else .01, help=help_text(track, field), key=key)
            if st.form_submit_button('Run analysis', type='primary', use_container_width=True):
                st.session_state[f'submitted_{track}'] = pd.DataFrame([values])
        frame = st.session_state.get(f'submitted_{track}')
        if frame is not None:
            st.caption('Showing the last submitted record. Submit again after editing fields.')

    if frame is not None:
        try:
            result = infer(bundle, frame, threshold)
            st.success(f'Analyzed {len(result):,} record(s) using {config["file"]}.')
            st.markdown('<div class="section-title"><span class="step">2</span>Understand the result</div>', unsafe_allow_html=True)
            columns = st.columns(3)
            columns[0].metric('Records analyzed', f'{len(result):,}')
            if track == 'cltv':
                columns[1].metric('Average predicted CLTV', f'{result.PredictedCLTV.mean():,.0f}')
                columns[2].metric('Highest predicted CLTV', f'{result.PredictedCLTV.max():,.0f}')
                interpretation = f"Estimated values use the source dataset's CLTV units. Currency and forecast horizon are unspecified. Average absolute holdout error: {report['mae']:,.0f}."
            elif track == 'credit':
                columns[1].metric('Average default score', f'{result.DefaultScore.mean():.1%}')
                columns[2].metric('Predicted default: Yes', str(int(result.PredictedDefault.eq('Yes').sum())))
                interpretation = f"At threshold {threshold:.0%}, Yes means predicted default payment next month. Scores are not calibrated probabilities. This academic model is not a lending or credit-eligibility decision system."
            else:
                columns[1].metric('Clusters represented', str(result.Cluster.nunique()))
                columns[2].metric('Largest cluster', f'Cluster {result.Cluster.mode().iloc[0]}')
                interpretation = 'Clusters summarize similarity in log mass, year, and standardized coordinates. IDs are not rankings or meteorite types. Latitude/longitude use Euclidean distance here, not great-circle distance.'
            if len(result) == 1:
                interpretation = ('Predicted CLTV: ' + f'{result.PredictedCLTV.iloc[0]:,.0f}. ' if track == 'cltv' else
                                  f'Default score: {result.DefaultScore.iloc[0]:.1%}; predicted default: {result.PredictedDefault.iloc[0]}. ' if track == 'credit' else
                                  f'This landing is assigned to Cluster {int(result.Cluster.iloc[0])}. ') + interpretation
            st.markdown(f'<div class="result-note">{interpretation}</div>', unsafe_allow_html=True)
            visual, records = st.tabs(['Visual summary', 'Records & export'])
            with visual:
                if track == 'meteorites':
                    counts = result.groupby('Cluster').size().reset_index(name='Records')
                    chart = alt.Chart(counts).mark_bar(color=CHART_PRIMARY, cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(x='Cluster:O', y='Records:Q', tooltip=['Cluster', 'Records'])
                    st.altair_chart(chart.properties(height=240), use_container_width=True)
                    st.markdown('#### Landing coordinates')
                    st.caption('Coordinate overview of the submitted records. Color represents assigned cluster.')
                    chart = alt.Chart(result[['reclong', 'reclat', 'Cluster']]).mark_circle(size=70, opacity=.7).encode(
                        x=alt.X('reclong:Q', title='Longitude', scale=alt.Scale(domain=[-180, 180])),
                        y=alt.Y('reclat:Q', title='Latitude', scale=alt.Scale(domain=[-90, 90])), color=alt.Color('Cluster:N', scale=alt.Scale(range=CHART_PALETTE)), tooltip=['reclat', 'reclong', 'Cluster'])
                    # Aggregate duplicate points to keep full uploads below the chart row limit.
                    points = result.groupby(['reclong', 'reclat', 'Cluster']).size().reset_index(name='Records')
                    if len(points) > 5000:
                        st.caption('Coordinate chart shows a fixed sample of 5,000 points; all rows remain in summaries and exports.')
                        points = points.sample(5000, random_state=42)
                    st.altair_chart(chart.properties(data=points, height=280), use_container_width=True)
                else:
                    field = config['output']
                    title = 'Predicted CLTV' if track == 'cltv' else 'Default score'
                    if len(result) == 1:
                        chart = alt.Chart(result[[field]]).mark_bar(color=CHART_PRIMARY, size=45).encode(
                            x=alt.X(f'{field}:Q', title=title, scale=alt.Scale(domain=[0, 1]) if track == 'credit' else alt.Scale(zero=True)), tooltip=[field])
                    else:
                        counts, edges = np.histogram(result[field], bins=15)
                        data = pd.DataFrame({'From': edges[:-1], 'To': edges[1:], 'Records': counts})
                        chart = alt.Chart(data).mark_bar(color=CHART_PRIMARY).encode(x=alt.X('From:Q', bin='binned', title=title), x2='To:Q', y='Records:Q', tooltip=['From', 'To', 'Records'])
                    if track == 'credit':
                        rule = alt.Chart(pd.DataFrame({'Threshold': [threshold]})).mark_rule(color=CHART_THRESHOLD, strokeDash=[5, 4]).encode(x='Threshold:Q')
                        chart = chart + rule
                    st.altair_chart(chart.properties(height=260), use_container_width=True)
                    st.caption('For multiple records, bars count predictions within each range. The rose line, when shown, marks the classification threshold.')
            with records:
                output = [config['output']] + (['PredictedDefault'] if track == 'credit' else [])
                ordered = output + [c for c in result if c not in output]
                st.dataframe(result[ordered], hide_index=True, use_container_width=True,
                    column_config={c: st.column_config.Column(label(track, c)) for c in ordered})
                st.download_button('Download complete results', result.to_csv(index=False), f'{track}_results.csv', 'text/csv', type='primary')
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    if track == 'meteorites':
        st.markdown('### Full-dataset cluster profiles')
        st.caption('Feature medians and record counts describe the cleaned source dataset, rather than only your submitted records.')
        st.dataframe(pd.read_csv(MODELS / 'meteorite_profiles.csv').rename(columns={c: label(track, c) for c in config['features']}), hide_index=True, use_container_width=True)

with performance:
    st.subheader('How was this model evaluated?')
    st.markdown(f"**Dataset:** {config['file']}  \n**Method:** {config['method']}")
    if track == 'cltv':
        stats = [('Holdout R-squared', f"{report['r2']:.3f}"), ('Mean absolute error', f"{report['mae']:,.0f}"), ('Test records', f"{report['test_rows']:,}")]
    elif track == 'credit':
        stats = [('Holdout ROC-AUC', f"{report['roc_auc']:.3f}"), ('Default precision', f"{report['precision']:.1%}"), ('Default recall', f"{report['recall']:.1%}")]
    else:
        stats = [('Selected clusters', str(report['selected_k'])), ('Retained landings', f"{report['rows']:,}"), ('Sample silhouette', f"{report['silhouette']:.3f}")]
    for col, (name, value) in zip(st.columns(3), stats):
        col.metric(name, value)
    if track == 'cltv':
        st.write('R-squared measures variation explained on held-out customers. The model explains about 20.5% here. Mean absolute error measures the average prediction gap, not an uncertainty interval. This model has limited predictive strength.')
        st.caption('Uses the existing regression_train.csv and regression_test.csv partition. SatisfactionScore and churn-related outcomes are excluded.')
    elif track == 'credit':
        st.write('ROC-AUC measures the ability to rank defaults above non-defaults. Precision is the share of flagged accounts that actually defaulted; recall is the share of actual defaults flagged. Metrics use threshold 0.50, independently of your current threshold.')
        matrix = pd.DataFrame(report['confusion_matrix'], index=['Actually no default', 'Actually default'], columns=['Predicted no default', 'Predicted default'])
        st.dataframe(matrix, use_container_width=True)
        st.caption('Stratified 80/20 split, seed 42. Five-fold weighted-F1 depth selection. Clipping, imputation, encoding, and scaling are fitted within each training fold.')
    else:
        st.write('Silhouette compares within-cluster similarity to separation between clusters. Higher is better. K=2 through K=10 were compared on the same 1,500 sampled records. Clustering is descriptive, with no verified target labels.')
        st.dataframe(pd.DataFrame(report['candidates']), hide_index=True)
        st.caption(f"Excluded {report['excluded_rows']:,} of {report['raw_rows']:,} records using the notebook's missing-data, mass, coordinate, year, and (0, 0) location rules.")
    with st.expander('Complete evaluation and provenance'):
        st.json(report)

with guide:
    st.subheader('One project, three independent datasets')
    st.dataframe(pd.DataFrame([{'Track': c['nav'], 'Dataset': c['file'], 'Input columns': len(c['features'])} for c in TRACKS.values()]), hide_index=True, use_container_width=True)
    st.markdown('#### How to use this track')
    st.write('1. Select the task in the sidebar.\n2. Try example records or review all sections of a single-record form.\n3. For batches, download the selected track\'s template and keep its exact headers.\n4. Read the interpretation, inspect Model performance, then export your results.')
    if track == 'credit':
        st.info('Target Y = 1 means default payment next month; Y = 0 means no default. The embedded text header and duplicate records are removed before training. Y and account IDs are never model inputs.')
        st.write('X2: 1 male, 2 female. X3: 1 graduate school, 2 university, 3 high school, 4 other; 0, 5 and 6 are unspecified codes. X4: 1 married, 2 single, 3 other, 0 unspecified. Repayment status: -2 no consumption, -1 paid duly, 0 revolving credit, 1-9 delay in months. All bill/payment months refer to 2005. Amounts are in NT dollars.')
        st.write('Negative bill balances and repayment-status codes are valid. Blank numeric values use training medians; blank category codes use training modes. Unsupported category and repayment codes are rejected. Demographic fields reproduce the academic dataset; outputs should not be used to decide lending eligibility.')
    elif track == 'meteorites':
        st.info('No target is required. Cluster IDs describe fitted groups, not meteorite composition, safety, or scientific class.')
        st.write('All four input values are required. Mass must be nonnegative; year must be 0-2026. Latitude must be -90 to 90 and longitude -180 to 180. The pair (0, 0) is excluded. Invalid uploaded rows cause an explicit validation message; they are not silently dropped. Mass uses log1p before standard scaling. The fitted K-Means model assigns new records without refitting.')
    else:
        st.info('CLTV is the supplied Telco target. Its horizon and currency are not specified.')
        st.write('Numeric blanks use training medians. Missing Offer and InternetType become No Offer and No Internet. Other categorical blanks use the training mode; unseen categories are accepted. Invalid numbers and negative values are rejected.')
    st.caption('Uploads: maximum 10,000 records / 10 MB. Extra columns are preserved in exports but ignored by the pipeline. Files must match the selected track. Example scores are not a fresh held-out evaluation.')
    reference = pd.DataFrame([{'CSV column': f, 'Label': label(track, f), 'Meaning': help_text(track, f)} for f in config['features']])
    st.dataframe(reference, hide_index=True, use_container_width=True)

st.markdown('<div class="footer">ML CAPSTONE EXPLORER / Regression, classification and clustering</div>', unsafe_allow_html=True)
