# MarketMind Intelligence V2 — Corrected deployment package

The ZIP is ready to extract as a GitHub repository folder. The layout has been fixed:
`app.py`, `requirements.txt`, `models/` and `data/`.

## Run
Python 3.11 recommended. Install with `pip install -r requirements.txt` and run `python app.py`.

## Deploy separately on Render
Create a **new** GitHub repository and **new** Render web service (do not replace V1).
Build: `pip install -r requirements.txt`; start: `python app.py`.
The sentence-transformers embedding model downloads at startup; ensure enough RAM and startup time.

## Validation
The archive integrity, Python syntax, model/vectorizer inference and one demand-model prediction were checked.
The saved models were trained with scikit-learn 1.9.1; requirements pin that version.
The full Gradio web service and SentenceTransformer runtime have **not** been executed here.
Model probabilities are uncalibrated. Demand/inventory data are simulated.
Only load trusted joblib/pickle model files.
