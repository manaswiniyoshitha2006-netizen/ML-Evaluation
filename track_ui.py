"""Display metadata for each independent capstone track."""
from ml_core import FEATURES, NUMERIC
from ui_helpers import label as telco_label, HELP as TELCO_HELP, INTEGERS
from track_data import CREDIT, METEORITE

CREDIT_LABELS = {'X1': 'Credit limit (NT dollars)', 'X2': 'Sex code', 'X3': 'Education code',
                 'X4': 'Marital status code', 'X5': 'Age (years)'}
MONTHS = ['September', 'August', 'July', 'June', 'May', 'April']
for i, month in enumerate(MONTHS):
    CREDIT_LABELS[f'X{6+i}'] = f'{month} repayment status'
    CREDIT_LABELS[f'X{12+i}'] = f'{month} bill amount (NT dollars)'
    CREDIT_LABELS[f'X{18+i}'] = f'{month} payment (NT dollars)'
METEORITE_LABELS = {'mass (g)': 'Mass (grams)', 'year': 'Recorded year',
                     'reclat': 'Latitude (degrees)', 'reclong': 'Longitude (degrees)'}
CHOICES = {'X2': [1, 2], 'X3': list(range(7)), 'X4': list(range(4)),
           **{f'X{i}': list(range(-2, 10)) for i in range(6, 12)}}
CODES = {
    'X2': {1: 'Male', 2: 'Female'},
    'X3': {0: 'Unspecified', 1: 'Graduate school', 2: 'University', 3: 'High school', 4: 'Other', 5: 'Unspecified (5)', 6: 'Unspecified (6)'},
    'X4': {0: 'Unspecified', 1: 'Married', 2: 'Single', 3: 'Other'},
}
STATUS = {-2: 'No consumption', -1: 'Paid duly', 0: 'Revolving credit',
          **{i: f'Payment delay: {i} month(s)' for i in range(1, 10)}}

def label(track, field):
    return (CREDIT_LABELS if track == 'credit' else METEORITE_LABELS if track == 'meteorites' else {}).get(field, telco_label(field))

def help_text(track, field):
    if track == 'cltv':
        return TELCO_HELP.get(field, 'Use the value recorded for the customer account.')
    if track == 'credit':
        if field in [f'X{i}' for i in range(6, 12)]:
            return 'Repayment status for the corresponding month in 2005. Negative codes and zero are valid.'
        if field in [f'X{i}' for i in range(12, 18)]:
            return 'Statement balance in NT dollars. A negative value can represent a credit balance.'
        return 'Original dataset column ' + field + '. See the user guide for the code definitions.'
    return {'mass (g)': 'Nonnegative mass in grams; transformed with log(1 + mass) inside the fitted pipeline.',
            'year': 'Year recorded in the source. Must not exceed the notebook analysis year 2026.',
            'reclat': 'Latitude between -90 and 90. The pair (0, 0) is excluded.',
            'reclong': 'Longitude between -180 and 180. The pair (0, 0) is excluded.'}[field]

def groups(track):
    if track == 'credit':
        return {'Account profile': CREDIT[:5], 'Repayment history': CREDIT[5:11],
                'Bill statements': CREDIT[11:17], 'Payments': CREDIT[17:]}
    if track == 'meteorites':
        return {'Landing information': METEORITE}
    first = ['Age', 'Gender', 'SeniorCitizen', 'Married', 'NumberofDependents', 'Number_of_Referrals']
    billing = ['TenureinMonths', 'Offer', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharge', 'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges', 'TotalLongDistanceCharges', 'AvgMonthlyLongDistanceCharges']
    return {'Customer profile': first, 'Account & billing': billing, 'Services & usage': [f for f in FEATURES if f not in first + billing]}
