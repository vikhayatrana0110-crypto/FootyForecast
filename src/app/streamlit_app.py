import streamlit as st
import os
if hasattr(st, "secrets"):
    for key in ["SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_USER", "SUPABASE_PASSWORD", "SUPABASE_DB"]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

st.set_page_config(
    page_title="FootyForecast",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.db_manager import DatabaseManager
from src.database.models import TeamFeature, RawEloRating
from src.features.elo_calculator import EloCalculator
from src.features.team_features import TeamFeatureEngine
from src.features.match_features import MatchFeatureEngine
from src.models.predictor import MatchPredictor
from src.models.explainer import MatchExplainer
from src.ingestion.download_data import run_ingestion
from src.models.trainer import ModelTrainer

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.gradient-title {
    background: linear-gradient(135deg, #00d4ff, #ffd700, #00e676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem; font-weight: 800;
    text-align: center; padding-bottom: 0.5rem;
}
.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 24px; margin: 10px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(0,212,255,0.15); }
.prob-card-home { border-left: 4px solid #00d4ff; }
.prob-card-draw { border-left: 4px solid #ffd700; }
.prob-card-away { border-left: 4px solid #ff5252; }
.factor-positive { border-left: 4px solid #00e676; background: rgba(0,230,118,0.05); }
.factor-negative { border-left: 4px solid #ff5252; background: rgba(255,82,82,0.05); }
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
    color: white !important; font-weight: 700 !important;
    padding: 0.75rem 3rem !important; border-radius: 12px !important;
    border: none !important; font-size: 1.1rem !important;
    transition: all 0.3s ease !important; width: 100%; margin-top: 1rem;
}
.stButton > button:hover { transform: scale(1.02) !important; box-shadow: 0 6px 20px rgba(0,212,255,0.4) !important; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db() -> DatabaseManager:
    db = DatabaseManager()
    db.init_db()
    return db


def get_predictor() -> Optional[MatchPredictor]:
    try:
        return MatchPredictor(db_manager=DatabaseManager())
    except Exception as e:
        st.error(f"Predictor error: {e}")
        return None


def db_stats(db: DatabaseManager):
    session = db.SessionLocal()
    try:
        from src.database.models import RawMatch
        match_count = session.query(RawMatch).count()
        teams = db.get_all_teams()
        model = db.get_active_model()
        return match_count, len(teams), (model.version if model else "N/A")
    finally:
        session.close()


def load_context(db: DatabaseManager):
    session = db.SessionLocal()
    try:
        matches = db.get_all_matches()
        team_feats = pd.read_sql(session.query(TeamFeature).statement, db.engine)
        elo_hist   = pd.read_sql(session.query(RawEloRating).statement, db.engine)
        return matches, team_feats, elo_hist
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="gradient-title">FootyForecast</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#888;font-size:1.2rem;margin-top:-10px;'>"
    "Predict outcomes &amp; expected goals with XGBoost + SHAP explanations</p>",
    unsafe_allow_html=True
)

db  = get_db()
match_cnt, team_cnt, model_ver = db_stats(db)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Database & Model Status")
st.sidebar.write(f"**Historical Matches:** {match_cnt:,}")
st.sidebar.write(f"**Unique Teams:** {team_cnt}")
st.sidebar.write(f"**Active Model:** `{model_ver}`")
st.sidebar.markdown("---")
st.sidebar.markdown("### Operations")

if match_cnt == 0 or model_ver == "N/A":
    st.sidebar.warning("Setup required - database is empty or model not trained.")
    if st.sidebar.button("Run Setup Pipeline"):
        with st.spinner("Ingesting historical match data..."):
            run_ingestion()
        with st.spinner("Computing Elo ratings..."):
            matches = db.get_all_matches()
            elo_calc = EloCalculator()
            elo_history = elo_calc.compute_all_ratings(matches)
            db.save_elo_ratings(elo_history)
        with st.spinner("Engineering features..."):
            team_feats = TeamFeatureEngine().compute_team_features(matches, elo_history)
            db.save_team_features(team_feats)
            match_feats = MatchFeatureEngine().compute_match_features(matches, team_feats, elo_history)
            db.save_match_features(match_feats)
        with st.spinner("Training XGBoost models..."):
            ModelTrainer().run_training_pipeline(match_feats, db_manager=db)
        st.sidebar.success("Pipeline complete! Refresh the page.")
        st.rerun()
else:
    if st.sidebar.button("Retrain Models"):
        with st.spinner("Retraining XGBoost models..."):
            match_feats = db.get_all_match_features()
            ModelTrainer().run_training_pipeline(match_feats, db_manager=db)
        st.sidebar.success("Model retrained successfully!")
        st.cache_resource.clear()
        st.rerun()

st.sidebar.markdown("""
**About the Model**
- Algorithm: XGBoost Multi-Class Classifier
- Target: Win / Draw / Loss
- Features: Elo diff, rolling form, rolling goals, H2H
- Explainability: SHAP values
""")

# ---------------------------------------------------------------------------
# Prediction workspace
# ---------------------------------------------------------------------------
if match_cnt == 0 or model_ver == "N/A":
    st.info("Predictor is disabled until data is ingested and a model is trained.")
    st.stop()

predictor = get_predictor()
if predictor is None:
    st.error("Could not load models. Run the pipeline first.")
    st.stop()

teams = db.get_all_teams()
default_home = teams.index("Brazil")    if "Brazil"    in teams else 0
default_away = teams.index("Argentina") if "Argentina" in teams else min(1, len(teams) - 1)

st.markdown("### Match Details")
col1, col2, col3 = st.columns(3)
with col1:
    home_team  = st.selectbox("Home Team",  teams, index=default_home, key="home")
with col2:
    away_team  = st.selectbox("Away Team",  teams, index=default_away, key="away")
with col3:
    tournament = st.selectbox("Tournament", [
        "FIFA World Cup", "UEFA Euro", "Copa America",
        "FIFA World Cup Qualification", "Friendly", "Other Match"
    ])

if home_team == away_team:
    st.error("Home and away teams must be different.")
    st.stop()

if st.button("Predict Match Outcome"):
    with st.spinner("Generating predictions..."):
        matches, team_feats_df, elo_hist_df = load_context(db)

        pred = predictor.predict_match(
            home_team, away_team, tournament,
            team_feats_df, elo_hist_df, matches,
            save_to_db=True
        )

        feats_dict = predictor.feature_engine.create_prediction_features(
            home_team, away_team, tournament, team_feats_df, elo_hist_df, matches
        )
        feats_df  = pd.DataFrame([feats_dict])
        feat_cols = predictor.feature_engine.get_feature_columns()

        explainer   = MatchExplainer(predictor.classifier)
        class_idx   = {"Home Win": 2, "Draw": 1, "Away Win": 0}.get(pred.get("outcome", "Home Win"), 2)
        explanation    = explainer.explain_prediction(feats_df, feat_cols, class_idx)
        shap_plot_data = explainer.get_shap_plot_data(feats_df, feat_cols, class_idx)

        # Persist the SHAP contributions alongside the prediction. predict_match()
        # returns prediction_id only when it saved the prediction, and
        # explanation_records() returns [] when the values would be heuristic
        # rather than real SHAP - in either case there is nothing worth storing.
        prediction_id = pred.get("prediction_id")
        if prediction_id:
            records = explainer.explanation_records(feats_df, feat_cols)
            if records:
                try:
                    db.save_prediction_explanations(prediction_id, records)
                except Exception as e:
                    # A logging failure must not take the prediction down with it.
                    st.warning(f"Could not save explanation details: {e}")

        st.session_state.update({
            "pred":       pred,
            "explanation": explanation,
            "shap_plot":  shap_plot_data,
            "home_feats": db.get_latest_team_features(home_team),
            "away_feats": db.get_latest_team_features(away_team),
        })

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if "pred" not in st.session_state:
    st.stop()

pred   = st.session_state["pred"]
p_home = pred["home_win_prob"]
p_draw = pred["draw_prob"]
p_away = pred["away_win_prob"]

st.markdown("## Prediction Dashboard")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="glass-card prob-card-home">', unsafe_allow_html=True)
    st.metric(f"{home_team} Win", f"{p_home*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with m2:
    st.markdown('<div class="glass-card prob-card-draw">', unsafe_allow_html=True)
    st.metric("Draw", f"{p_draw*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with m3:
    st.markdown('<div class="glass-card prob-card-away">', unsafe_allow_html=True)
    st.metric(f"{away_team} Win", f"{p_away*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

prob_fig = px.bar(
    pd.DataFrame({"Outcome": [home_team, "Draw", away_team], "Probability": [p_home, p_draw, p_away]}),
    x="Probability", y="Outcome", orientation="h", color="Outcome",
    color_discrete_map={home_team: "#00d4ff", "Draw": "#ffd700", away_team: "#ff5252"},
    title="Outcome Probability Distribution", template="plotly_dark"
)
prob_fig.update_layout(xaxis=dict(tickformat=".0%"), showlegend=False, height=220)
st.plotly_chart(prob_fig, use_container_width=True)

g1, g2 = st.columns(2)
for col, label, val, color in [
    (g1, home_team, pred["expected_home_goals"], "#00d4ff"),
    (g2, away_team, pred["expected_away_goals"], "#ff5252")
]:
    with col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            gauge={
                "axis": {"range": [0, 4.0]}, "bar": {"color": color},
                "steps": [
                    {"range": [0, 1.0],   "color": "rgba(255,255,255,0.05)"},
                    {"range": [1.0, 2.5], "color": "rgba(255,255,255,0.10)"},
                    {"range": [2.5, 4.0], "color": "rgba(255,255,255,0.15)"},
                ]
            }
        ))
        fig.update_layout(template="plotly_dark", height=220,
                          title_text=f"xG: {label}", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# Verdict. Stated with its probability rather than bare: a 37/29/34 split is a
# near coin-flip, and showing "Away Win" alone reads as far more certain than the
# model actually is.
verdict      = pred.get("outcome", "")
most_likely  = pred.get("most_likely_outcome", verdict)
verdict_prob = {"Home Win": p_home, "Draw": p_draw, "Away Win": p_away}.get(verdict, 0.0)

st.markdown(f"### Prediction: **{verdict}** ({verdict_prob:.1%})")
if most_likely != verdict:
    # The verdict came from the draw threshold rather than the top probability.
    st.caption(
        f"Called a draw because the draw probability ({p_draw:.1%}) is high for this "
        f"fixture, though {most_likely.lower()} is the single most likely result."
    )

conf = pred["confidence"]
conf_level = ("High Confidence" if conf > 0.35 else
              "Medium Confidence" if conf > 0.15 else
              "Low Confidence / Highly Competitive")
st.markdown(f"**Separation between the top two outcomes: {conf:.1%}** - {conf_level}")
if conf < 0.05:
    st.caption("The top two outcomes are within 5 percentage points: treat this as closely matched rather than a firm call.")
st.progress(min(1.0, conf / 0.6))

# SHAP chart
st.markdown("## ML Model Explanation (SHAP)")
exp       = st.session_state["explanation"]
shap_data = st.session_state["shap_plot"]

shap_df = pd.DataFrame({
    "Feature":      [f.replace("_", " ").title() for f in shap_data["features"]],
    "Contribution": shap_data["shap_values"]
})
shap_df["Direction"] = np.where(shap_df["Contribution"] > 0, "Positive", "Negative")

shap_fig = px.bar(
    shap_df, x="Contribution", y="Feature", orientation="h",
    color="Direction",
    color_discrete_map={"Positive": "#00e676", "Negative": "#ff5252"},
    title=f"Feature Contributions: {pred.get('outcome', '')}",
    template="plotly_dark"
)
shap_fig.update_layout(showlegend=False, height=320)
st.plotly_chart(shap_fig, use_container_width=True)

st.markdown(f"<div class='glass-card'>{exp.get('explanation_text', '')}</div>", unsafe_allow_html=True)

cp, cn = st.columns(2)
with cp:
    st.markdown("#### Positive Factors")
    for f in exp.get("positive_factors", []):
        st.markdown(f"""
        <div class='glass-card factor-positive'>
          <strong>{f['feature_name'].replace('_',' ').title()}</strong>: {f['feature_value']:.2f}<br/>
          <span style='color:#00e676'>Contribution: +{f['shap_value']:.3f}</span>
        </div>""", unsafe_allow_html=True)
with cn:
    st.markdown("#### Negative Factors")
    for f in exp.get("negative_factors", []):
        st.markdown(f"""
        <div class='glass-card factor-negative'>
          <strong>{f['feature_name'].replace('_',' ').title()}</strong>: {f['feature_value']:.2f}<br/>
          <span style='color:#ff5252'>Contribution: {f['shap_value']:.3f}</span>
        </div>""", unsafe_allow_html=True)

# Radar
st.markdown("## Team Comparison")
h_feats = st.session_state.get("home_feats")
a_feats = st.session_state.get("away_feats")

if h_feats and a_feats:
    metrics = ["ELO Rating", "Win Rate", "Goals Scored", "Attack Rating", "Defense Rating", "Form Score"]
    keys    = ["elo_rating", "recent_win_rate", "goals_scored_avg", "attack_rating", "defense_rating", "form_score"]
    scales  = [1200, 1.0, 4.0, 4.0, 1.0, 3.0]
    offsets = [1000, 0,   0,   0,   0,   0  ]

    h_vals = [min(1.0, (h_feats[k] - o) / s) for k, s, o in zip(keys, scales, offsets)]
    a_vals = [min(1.0, (a_feats[k] - o) / s) for k, s, o in zip(keys, scales, offsets)]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=h_vals, theta=metrics, fill="toself",
                                    name=home_team, line_color="#00d4ff"))
    radar.add_trace(go.Scatterpolar(r=a_vals, theta=metrics, fill="toself",
                                    name=away_team, line_color="#ff5252"))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=True, template="plotly_dark",
        title="Relative Strength Radar"
    )
    st.plotly_chart(radar, use_container_width=True)

    comp_rows = [
        {"Metric": m, home_team: f"{h_feats[k]:.2f}" if isinstance(h_feats[k], float) else str(h_feats[k]),
         away_team: f"{a_feats[k]:.2f}" if isinstance(a_feats[k], float) else str(a_feats[k])}
        for m, k in zip(metrics, keys)
    ]
    st.table(pd.DataFrame(comp_rows).set_index("Metric"))
