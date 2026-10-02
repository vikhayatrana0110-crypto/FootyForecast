from src.database.models import RawEloRating, TeamFeature, MatchFeature
from src.features.elo_calculator import EloCalculator
from src.features.team_features import TeamFeatureEngine
from src.features.match_features import MatchFeatureEngine
from src.ingestion.download_data import run_ingestion
from src.models.trainer import ModelTrainer

def save_once(db_manager, model, df, save):
    """Insert df only if the table is empty, so re-runs don't duplicate rows."""
    if db_manager.is_empty(model):
        print(f"Saving {len(df)} {model.__tablename__} rows...")
        save(df)
    else:
        print(f"{model.__tablename__} already populated. Skipping insertion.")

def main():
    print("FootyForecast -- Setup & Training Pipeline")

    print("\n[Step 1/4] Ingesting historical match data...")
    db_manager = run_ingestion(min_date='2000-01-01')

    print("\n[Step 2/4] Calculating ELO ratings...")
    matches = db_manager.get_all_matches()
    elo_history = EloCalculator().compute_all_ratings(matches)
    save_once(db_manager, RawEloRating, elo_history, db_manager.save_elo_ratings)

    print("\n[Step 3/4] Engineering team & match features...")
    team_feats = TeamFeatureEngine().compute_team_features(matches, elo_history)
    match_feats = MatchFeatureEngine().compute_match_features(matches, team_feats)
    save_once(db_manager, TeamFeature, team_feats, db_manager.save_team_features)
    save_once(db_manager, MatchFeature, match_feats, db_manager.save_match_features)

    print("\n[Step 4/4] Training XGBoost classifier & goals regressors...")
    ModelTrainer().run_training_pipeline(match_feats, db_manager=db_manager)

    print("\nPipeline completed successfully! Run: streamlit run src/app/streamlit_app.py")


if __name__ == '__main__':
    main()
