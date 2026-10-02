# FootyForecast

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Predicts international football match outcomes (win/draw/loss probabilities and expected goals) from historical results, rolling team stats, and Elo ratings, and explains each prediction with SHAP feature contributions.

## Features

- **Match prediction** — home win, draw, and away win probabilities, plus expected goals (xG) for both teams
- **Explainability** — SHAP (SHapley Additive exPlanations) values show which factors pushed a prediction up or down
- **Elo ratings** — team Elo is recalculated match by match, in chronological order
- **Rolling form and stats** — attack/defense ratings, a weighted form score, and average goals scored/conceded, tracked per team over a rolling window
- **Streamlit UI** — dark-themed interface with gauge indicators, radar charts, and Plotly visualizations
- **3-layer database schema** — raw, feature, and ML layers, managed with SQLAlchemy

## Tech stack
Data engineering: pandas, numpy, SQLAlchemy, PostgreSQL (Supabase)
Machine learning: scikit-learn, xgboost, shap, joblib
Frontend / visualizations: streamlit, plotly

## Repository structure

footyforecast/
├── .streamlit/
│   └── config.toml            # Dark theme configuration
├── .env.example               # Template for database credentials
├── run_pipeline.py            # Full ETL + training pipeline (main entry point)
├── data/
│   └── raw/                   # Local pipeline output (git-ignored)
├── src/
│   ├── app/
│   │   ├── streamlit_app.py   # Streamlit UI implementation
│   │   └── models/            # Deployment copy of trained .joblib artifacts
│   ├── ingestion/
│   │   └── download_data.py   # CSV downloader + database loader
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
│   ├── test_db_manager.py     # Unit tests for model-version bookkeeping
│   ├── test_elo.py            # Unit tests for Elo calculator
│   ├── test_features.py       # Unit tests for feature engine
│   └── test_predictor.py      # Unit tests for prediction and SHAP
├── requirements.txt           # Package dependencies
└── README.md                  # Documentation

The deployed Streamlit app can't run the training pipeline, so it loads pre-trained artifacts straight from the repo — that's why data and model files are committed under `src/app/`. The equivalent paths at the project root are git-ignored and used only for local pipeline runs.

## Quick start

### 1. Clone and set up the environment
You'll need Python 3.11+. From the repo root:

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 2. Configure database credentials
The app stores everything in a Supabase PostgreSQL database. Copy the template and fill in your own project details:

cp .env.example .env

All five keys (SUPABASE_HOST, SUPABASE_PORT, SUPABASE_USER, SUPABASE_PASSWORD, SUPABASE_DB) are required — the app fails fast with a clear message if any are missing. When deploying to Streamlit Cloud, set the same keys in the app's Secrets panel instead of committing a .env file.

### 3. Run the pipeline
Download historical results, calculate Elo ratings, build the feature tables, and train the XGBoost models in one step:

python run_pipeline.py

This writes trained artifacts to `models/` and is safe to re-run — each stage skips itself if the data already exists. You can also trigger it from the Streamlit sidebar.

### 4. Launch the Streamlit app

streamlit run src/app/streamlit_app.py

Open http://localhost:8501 in your browser.

## Running tests
Verify database, feature, and prediction logic with unittest:

python -m unittest discover tests

## License
MIT
