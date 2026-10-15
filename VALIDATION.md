# Validation report

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
