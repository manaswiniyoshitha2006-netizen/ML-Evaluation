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

CSS = """
<style>
.stApp {background:#f5f7fb;color:#172b45;}
[data-testid="stSidebar"] {background:#10243c; border-right:1px solid #193852;}
[data-testid="stSidebar"] * {color:#e7eff8;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#b4c6db;}
.block-container {max-width:1400px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;}
h3 {font-size:1.35rem;}
[data-testid="stMetric"] {background:white;border:1px solid #dfe7f0;border-radius:14px;padding:20px;box-shadow:0 3px 12px #132e4505;}
[data-testid="stMetricLabel"] {color:#5e7088;}
[data-testid="stMetricValue"] {color:#172b45;font-weight:700;}
[data-testid="stForm"], [data-testid="stExpander"] {background:white;border:1px solid #dfe7f0;border-radius:14px;}
[data-testid="stForm"] {padding:24px;}
.stButton button,.stDownloadButton button {border-radius:9px;font-weight:600;min-height:42px;}
button[data-baseweb="tab"] {font-weight:600;font-size:.95rem;padding:12px 20px;}
.hero {padding:30px 36px;border-radius:20px;background:linear-gradient(110deg,#102840,#1b5266);color:white;margin-bottom:20px;}
.hero h1 {color:white;font-size:2.2rem;line-height:1.2;margin:10px 0;padding:0;}
.hero p {color:#d2e0eb;max-width:740px;margin:0;line-height:1.65;}
.eyebrow {color:#88e0cc;font-size:.7rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;}
.step {display:inline-flex;align-items:center;justify-content:center;background:#e5f2f2;color:#176b73;width:30px;height:30px;border-radius:9px;margin-right:10px;font-size:.9rem;vertical-align:middle;}
.section-title {font-size:1.35rem;font-weight:700;margin:22px 0 8px;letter-spacing:-.025em;}
.guide-card {background:white;border:1px solid #dfe7f0;border-top:3px solid #298a8e;border-radius:14px;padding:22px;min-height:180px;}
.guide-card h4 {color:#172b45;margin:0 0 10px;font-size:1.1rem;}
.guide-card p {color:#5b6c82;line-height:1.6;font-size:.92rem;}
.result-note {border-left:4px solid #298a8e;background:#eaf5f3;border-radius:0 12px 12px 0;padding:18px 22px;margin:18px 0;color:#214d53;line-height:1.65;}
.footer {border-top:1px solid #dce5ef;color:#708099;padding-top:20px;margin-top:32px;font-size:.8rem;}
@media(max-width:700px){.hero{padding:24px;}.hero h1{font-size:1.7rem;}.block-container{padding-top:1rem;}button[data-baseweb="tab"]{padding:10px 8px;font-size:.85rem;}[data-testid="stForm"]{padding:12px;}}
</style>
"""
