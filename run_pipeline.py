import os
import sys
from src.database.db_manager import DatabaseManager
from src.features.elo_calculator import EloCalculator
from src.features.team_features import TeamFeatureEngine
from src.features.match_features import MatchFeatureEngine
from src.ingestion.download_data import run_ingestion
from src.models.trainer import ModelTrainer

def main():
    print("FootyForecast -- Setup & Training Pipeline")
    
    # 1. Ingest historical matches (post-2000)
    print("\n[Step 1/4] Ingesting historical match data...")

    db_manager = run_ingestion(min_date='2000-01-01')
    
    # 2. Calculate ELO ratings
    print("\n[Step 2/4] Calculating ELO ratings...")
    matches = db_manager.get_all_matches()
    elo_calc = EloCalculator()
    elo_history = elo_calc.compute_all_ratings(matches)
    
    with db_manager:
        # Check if ELO ratings already exist to avoid duplicate inserts
        session = db_manager.get_session()
        from src.database.models import RawEloRating
        count = session.query(RawEloRating).count()
        if count > 0:
            print(f"ELO history already exists ({count} entries). Skipping insertion.")
        else:
            print(f"Saving {len(elo_history)} ELO rating history entries...")
            db_manager.save_elo_ratings(elo_history)
            
    # 3. Engineer features
    print("\n[Step 3/4] Engineering team & match features...")
    t_engine = TeamFeatureEngine()
    team_feats = t_engine.compute_team_features(matches, elo_history)
    
    m_engine = MatchFeatureEngine()
    match_feats = m_engine.compute_match_features(matches, team_feats, elo_history)
    
    with db_manager:
        session = db_manager.get_session()
        from src.database.models import TeamFeature, MatchFeature
        
        # Save team features
        tf_count = session.query(TeamFeature).count()
        if tf_count > 0:
            print(f"Team features already exist ({tf_count} entries). Skipping insertion.")
        else:
            print(f"Saving {len(team_feats)} team rolling feature entries...")
            db_manager.save_team_features(team_feats)
            
        # Save match features
        mf_count = session.query(MatchFeature).count()
        if mf_count > 0:
            print(f"Match features already exist ({mf_count} entries). Skipping insertion.")
        else:
            print(f"Saving {len(match_feats)} match feature entries...")
            db_manager.save_match_features(match_feats)
            
    # 4. Train models
    print("\n[Step 4/4] Training XGBoost classifier & goals regressors...")
    trainer = ModelTrainer()
    trainer.run_training_pipeline(match_feats, db_manager=db_manager)
    
    print("\n==================================================")
    print("Pipeline completed successfully! Ready to run.")
    print("Run command: streamlit run src/app/streamlit_app.py")
    print("==================================================")


if __name__ == '__main__':
    main()
