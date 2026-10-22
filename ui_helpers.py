"""Presentation labels and styles; model input names remain unchanged."""
import re

LABELS = {
    'Age': 'Age (years)', 'SeniorCitizen': 'Senior citizen',
    'NumberofDependents': 'Number of dependents', 'Number_of_Referrals': 'Customers referred',
    'TenureinMonths': 'Time with company (months)', 'AvgMonthlyLongDistanceCharges': 'Average monthly long-distance charges',
    'AvgMonthlyGBDownload': 'Average monthly downloads (GB)', 'MonthlyCharge': 'Monthly charge',
    'TotalCharges': 'Total charges to date', 'TotalRefunds': 'Total refunds to date',
    'TotalExtraDataCharges': 'Total extra-data charges', 'TotalLongDistanceCharges': 'Total long-distance charges',
    'InternetType': 'Internet connection type', 'DeviceProtectionPlan': 'Device protection',
    'PremiumTechSupport': 'Premium technical support', 'Offer': 'Current offer',
    'Contract': 'Contract term', 'PredictedCLTV': 'Predicted customer lifetime value',
    'ChurnProbability': 'Churn score', 'PredictedChurn': 'Flagged for churn review',
    'ServiceCount': 'Number of add-on services', 'Customers': 'Customers in group',
}
HELP = {
    'TenureinMonths': 'The number of months this customer has been with the company.',
    'Number_of_Referrals': 'How many customers this person has referred.',
    'TotalCharges': 'Charges accumulated to date, rather than the current monthly bill.',
    'AvgMonthlyGBDownload': 'Average monthly internet data usage in gigabytes.',
    'Offer': 'Choose No Offer if the customer has no promotional offer.',
    'InternetType': 'Choose No Internet when Internet service is No.',
}
INTEGERS = {'Age', 'NumberofDependents', 'Number_of_Referrals', 'TenureinMonths', 'AvgMonthlyGBDownload'}
SEGMENTS = {
    0: 'Low spend / strong referrals', 1: 'Established / service-rich',
    2: 'Newer / moderate spend', 3: 'Low spend / essential services',
    4: 'Established / strong referrals', 5: 'High data usage',
}

def label(name):
    return LABELS.get(name, re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).replace('_', ' '))

CHART_PRIMARY = '#7961CE'
CHART_THRESHOLD = '#CB4E78'
CHART_PALETTE = ['#7961CE', '#38A6A3', '#DD9F43', '#CB4E78', '#5E82C4', '#93735A', '#9B66B1', '#729845', '#D87850', '#717B97']

CSS = """
<style>
.stApp {background:#f6f5fb;color:#292540;}
[data-testid="stSidebar"] {background:#211c3b; border-right:1px solid #3a315b;}
[data-testid="stSidebar"] * {color:#f1edff;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#c7bedf;}
.block-container {max-width:1400px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;}
h3 {font-size:1.35rem;}
[data-testid="stMetric"] {background:white;border:1px solid #e6e1f1;border-radius:16px;padding:20px;border-top:3px solid #ad96eb;box-shadow:0 3px 12px #4434690a;}
[data-testid="stMetricLabel"] {color:#756c88;}
[data-testid="stMetricValue"] {color:#292540;font-weight:700;}
[data-testid="stForm"], [data-testid="stExpander"] {background:white;border:1px solid #e6e1f1;border-radius:14px;}
[data-testid="stForm"] {padding:24px;}
.stButton button,.stDownloadButton button {border-radius:9px;font-weight:600;min-height:42px;}
[data-testid="stBaseButton-primary"] {background:#7157c9;border-color:#7157c9;color:white;}
[data-testid="stBaseButton-primary"]:hover {background:#5d45b1;border-color:#5d45b1;color:white;}
[data-testid="stBaseButton-secondary"]:hover {border-color:#9178dc;color:#6149ac;background:#f1edfb;}
button[data-baseweb="tab"] {font-weight:600;font-size:.95rem;padding:12px 20px;}
.hero {padding:30px 36px;border-radius:20px;background:radial-gradient(ellipse at 92% 0%,#a881cf55,transparent 55%),linear-gradient(115deg,#322455,#6751a8);color:white;margin-bottom:20px;}
.hero h1 {color:white;font-size:2.2rem;line-height:1.2;margin:10px 0;padding:0;}
.hero p {color:#e2d9f7;max-width:740px;margin:0;line-height:1.65;}
.eyebrow {color:#d6c5ff;font-size:.7rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;}
.step {display:inline-flex;align-items:center;justify-content:center;background:#ece7fb;color:#6550b5;width:30px;height:30px;border-radius:9px;margin-right:10px;font-size:.9rem;vertical-align:middle;}
.section-title {font-size:1.35rem;font-weight:700;margin:22px 0 8px;letter-spacing:-.025em;}
.guide-card {background:white;border:1px solid #e6e1f1;border-top:3px solid #7961ce;border-radius:14px;padding:22px;min-height:180px;}
.guide-card h4 {color:#292540;margin:0 0 10px;font-size:1.1rem;}
.guide-card p {color:#756c88;line-height:1.6;font-size:.92rem;}
.result-note {border-left:4px solid #7961ce;background:#eee9fb;border-radius:0 12px 12px 0;padding:18px 22px;margin:18px 0;color:#4b3c78;line-height:1.65;}
.footer {border-top:1px solid #e4deee;color:#80758f;padding-top:20px;margin-top:32px;font-size:.8rem;}
@media(max-width:700px){.hero{padding:24px;}.hero h1{font-size:1.7rem;}.block-container{padding-top:1rem;}button[data-baseweb="tab"]{padding:10px 8px;font-size:.85rem;}[data-testid="stForm"]{padding:12px;}}
</style>
"""
