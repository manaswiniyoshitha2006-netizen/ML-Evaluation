# Validation report

## Three-dataset correction — 21 October 2026

- Replaced the Telco-only classification/clustering app tracks with credit-card default classification and meteorite clustering, matching the actual project notebooks.
- Trained complete pipelines for each independent dataset. Classification retained 29,965 accounts; meteorite cleaning retained 31,929 landings.
- `python -m pytest -q`: **13 passed** in 9.66 seconds, including different-track schema rejection and independent data provenance.
- Classification: decision tree depth 5, holdout accuracy 0.8173, weighted F1 0.7978, ROC-AUC 0.7446.
- Meteorites: K=2, sampled silhouette 0.4881. Regression results unchanged.
- Prior Telco-only model artifacts and metrics preserved in `models/legacy_telco/`. Original notebooks and data unchanged.
- App branding, templates, forms, charts, and documentation now distinguish all three datasets.

## Structured interface update — 21 October 2026

- `python -m pytest -q`: **9 passed** in 4.09 seconds.
- Added regression coverage for keeping a submitted churn result visible when changing its threshold.
- Browser review checked the reorganized analysis page, readable labels, and section navigation.
- Fitted models, source datasets, and notebook work are unchanged by this interface update.

## Initial implementation

- Source repository commit: 14c45b6229c5ff26908d4b8ed15ce17e8feac943.
- Python 3.11.9; scikit-learn 1.5.0; Streamlit 1.50.0; NumPy 1.26.4; pandas 2.2.2; PyArrow 17.0.0.
- Training completed on the repository datasets; three fitted pipelines saved in models/app/.
- `python -m pytest -q`: 8 passed (14.09 seconds).
- Tests cover all three model artifacts, outcome exclusion, batch/single consistency, missing and unseen inputs, input validation, partition integrity, all app views and single-customer form submissions.
- `python -m pip check`: no broken requirements.
- `git diff --cached --check`: passed.
- Real localhost server: HTTP 200, health endpoint ok, application HTML served. Temporary server stopped after verification.
- CSV inference is tested through the shared prediction layer. Browser file-picker automation and a live Community Cloud deployment were not performed.
- Original notebooks/data unchanged; prior README retained as README.original.md.
- GitHub remote was not modified. Deliverables are ready for an account with write access.

The newly released PyArrow dependency required NumPy 2; pinning PyArrow 17.0.0 resolved the interface failure while preserving the original NumPy version. Cloud deployment uses Python 3.11 and the pinned runtime requirements. Notebook plotting/Jupyter dependencies are in requirements-notebooks.txt.
