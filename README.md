# World Cup AI Predictor

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A machine learning analytics platform that predicts international association football match outcomes (Win/Draw/Loss probabilities and expected goals) using historical match results, rolling team statistics, and ELO ratings. Explains model decisions using SHAP value feature contributions.

---

## Key Features

- **Match Prediction**: Generates home win, draw, and away win probabilities alongside Expected Goals (xG) estimates for both teams.
- **Explainability**: Integrated SHAP (SHapley Additive exPlanations) values to detail exactly why the model made a prediction (positive/negative contributors).
- **ELO Rating Engine**: Calculates dynamic team Elo ratings updated match-by-match chronologically.
- **Rolling Form & Stats**: Engineers team-level metrics (rolling attack/defense rating, weighted form score, average goals scored/conceded).
- **Interactive UI**: A dark-themed Streamlit interface featuring glassmorphic designs, Gauge indicators, radar charts, and interactive Plotly visualizations.
- **Robust Schema**: A structured 3-layer database architecture (Raw -> Feature -> ML) managed via SQLAlchemy.

---

## Tech Stack

- **Data Engineering**: `pandas`, `numpy`, `SQLAlchemy`, PostgreSQL (Supabase)
- **Machine Learning**: `scikit-learn`, `xgboost`, `shap`, `joblib`
- **Frontend / Visualizations**: `streamlit`, `plotly`

---

## Repository Structure

```
world-cup-ai-predictor/
├── .streamlit/
│   └── config.toml            # Dark theme configuration
├── .env.example               # Template for database credentials
├── run_pipeline.py            # Full ETL + training pipeline (main entry point)
├── data/
│   └── raw/                   # Local pipeline output (git-ignored)
├── sql/
│   └── schema.sql             # Database schema SQL statements
├── src/
│   ├── app/
│   │   ├── streamlit_app.py   # Streamlit UI implementation
│   │   ├── data/              # Deployment copy of the dataset + database
│   │   └── models/            # Deployment copy of trained .joblib artifacts
│   ├── ingestion/
│   │   ├── download_data.py   # CSV downloader + database loader
│   │   └── statsbomb_loader.py # Optional StatsBomb xG enrichment
│   ├── database/
│   │   ├── models.py          # SQLAlchemy database models
│   │   └── db_manager.py      # CRUD and SQL database operations
│   ├── features/
│   │   ├── elo_calculator.py  # Elo calculation logic
│   │   ├── team_features.py   # Team-level rolling features
│   │   └── match_features.py  # Match-level difference features
│   └── models/
│       ├── trainer.py         # Model training pipeline
│       ├── predictor.py       # Inference interface
│       └── explainer.py       # SHAP explainer wrapper
├── tests/
│   ├── test_elo.py            # Unit tests for Elo calculator
│   ├── test_features.py       # Unit tests for feature engine
│   └── test_predictor.py      # Unit tests for prediction and SHAP
├── requirements.txt           # Package dependencies
└── README.md                  # Documentation
```

> **Why data and model files are committed under `src/app/`:** the deployed
> Streamlit app cannot run the training pipeline, so it loads the pre-trained
> artifacts straight from the repository. The equivalent paths at the project
> root are git-ignored and used only for local pipeline runs.

---

## Quick Start Guide

### 1. Clone & Set Up Environment
First, ensure you have Python 3.11+ installed. Navigate to the repository and install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Database Credentials
The app stores everything in a Supabase PostgreSQL database. Copy the template and
fill in your own project details:
```bash
cp .env.example .env
```
All five keys (`SUPABASE_HOST`, `SUPABASE_PORT`, `SUPABASE_USER`, `SUPABASE_PASSWORD`,
`SUPABASE_DB`) are required - the app fails fast with a clear message if any are missing.
When deploying to Streamlit Cloud, set the same keys in the app's **Secrets** panel
rather than committing a `.env` file.

### 3. Run the Pipeline
Download historical results, calculate Elo ratings, build the feature tables, and train
the XGBoost models in one step:
```bash
python run_pipeline.py
```
This writes trained artifacts to `models/` and is safe to re-run - each stage skips
itself if the data already exists. You can also trigger it from the Streamlit sidebar.

### 4. Launch Streamlit Application
Start the interactive UI:
```bash
streamlit run src/app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Running Tests
Verify database, features, and model prediction logic using `unittest`:
```bash
python -m unittest discover tests
```

---

## License
This project is licensed under the MIT License.
