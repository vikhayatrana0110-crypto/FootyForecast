import os
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import select, and_, or_, desc, text
from dotenv import load_dotenv
load_dotenv()

from src.database.models import (
    create_all_tables, RawMatch, RawEloRating,
    RawTeamStatistic, TeamFeature, MatchFeature, ModelVersion,
    Prediction, PredictionExplanation
)

class DatabaseManager:
    def __init__(self):
        """
        Connect to the Supabase PostgreSQL database.

        Connection details come from the environment (loaded from .env locally,
        or injected from st.secrets by the Streamlit app). Nothing is hardcoded:
        the repository is public, so the host must not be committed.
        """
        # Host and password identify and unlock the database, so they have no
        # defaults - a missing value must fail loudly rather than silently
        # connecting somewhere unintended.
        DB_HOST = os.getenv("SUPABASE_HOST")
        DB_PASSWORD = os.getenv("SUPABASE_PASSWORD")

        # These are Supabase conventions, not secrets, so defaults are safe.
        DB_PORT = os.getenv("SUPABASE_PORT", "5432")
        DB_USER = os.getenv("SUPABASE_USER", "postgres")
        DB_NAME = os.getenv("SUPABASE_DB", "postgres")

        missing = [
            name for name, value in (
                ("SUPABASE_HOST", DB_HOST),
                ("SUPABASE_PASSWORD", DB_PASSWORD),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required database configuration: " + ", ".join(missing) + ". "
                "Set these in a local .env file, or in the app's Streamlit secrets "
                "when deploying."
            )

        # Credentials are percent-encoded: an unescaped '@', ':' or '/' in the
        # password would otherwise be parsed as part of the URL structure.
        DATABASE_URL = (
            f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.session: Optional[Session] = None

    def __enter__(self):
        self.session = self.SessionLocal()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()
            self.session = None

    def get_session(self) -> Session:
        if self.session is None:
            return self.SessionLocal()
        return self.session

    def init_db(self):
        create_all_tables(self.engine)
        self._apply_column_migrations()

    def _apply_column_migrations(self):
        """
        Add columns introduced after a table was first created.

        create_all_tables() only creates missing *tables*, so a new column on an
        existing table would otherwise never appear on a deployed database. Each
        statement is guarded by IF NOT EXISTS, so this is safe to run on every
        startup and on a database that is already up to date.
        """
        statements = [
            "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS mae DOUBLE PRECISION",
            "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS rmse DOUBLE PRECISION",
        ]
        with self.engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))

    def bulk_insert_matches(self, df: pd.DataFrame):
        """Bulk insert raw matches from a DataFrame."""
        session = self.get_session()
        try:
            records = []
            for _, row in df.iterrows():
                # Parse date if string
                d = row['date']
                if isinstance(d, str):
                    d = datetime.strptime(d, '%Y-%m-%d').date()
                
                records.append(RawMatch(
                    date=d,
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    home_score=int(row['home_score']),
                    away_score=int(row['away_score']),
                    tournament=row.get('tournament'),
                    city=row.get('city'),
                    country=row.get('country'),
                    neutral=bool(row.get('neutral', False))
                ))
            session.bulk_save_objects(records)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if self.session is None:
                session.close()

    def get_all_matches(self, start_date: Optional[Any] = None) -> pd.DataFrame:
        """Fetch all raw matches as a DataFrame."""
        session = self.get_session()
        try:
            query = session.query(RawMatch)
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                query = query.filter(RawMatch.date >= start_date)
            query = query.order_by(RawMatch.date)
            
            # Convert query to DataFrame
            df = pd.read_sql(query.statement, self.engine)
            return df
        finally:
            if self.session is None:
                session.close()

    def get_team_matches(self, team: str, before_date: Optional[Any] = None) -> pd.DataFrame:
        """Fetch matches played by a specific team, optionally before a date."""
        session = self.get_session()
        try:
            query = session.query(RawMatch).filter(
                or_(RawMatch.home_team == team, RawMatch.away_team == team)
            )
            if before_date:
                if isinstance(before_date, str):
                    before_date = datetime.strptime(before_date, '%Y-%m-%d').date()
                elif isinstance(before_date, datetime):
                    before_date = before_date.date()
                query = query.filter(RawMatch.date < before_date)
            query = query.order_by(RawMatch.date)
            df = pd.read_sql(query.statement, self.engine)
            return df
        finally:
            if self.session is None:
                session.close()

    def get_all_teams(self) -> List[str]:
        """Get list of unique team names."""
        session = self.get_session()
        try:
            home_teams = session.query(RawMatch.home_team).distinct().all()
            away_teams = session.query(RawMatch.away_team).distinct().all()
            teams = set([t[0] for t in home_teams] + [t[0] for t in away_teams])
            return sorted(list(teams))
        finally:
            if self.session is None:
                session.close()

    def save_elo_ratings(self, df: pd.DataFrame):
        """Bulk save ELO ratings."""
        session = self.get_session()
        try:
            records = []
            for _, row in df.iterrows():
                d = row['date']
                if isinstance(d, str):
                    d = datetime.strptime(d, '%Y-%m-%d').date()
                elif isinstance(d, datetime):
                    d = d.date()
                records.append(RawEloRating(
                    team=row['team'],
                    date=d,
                    elo_rating=float(row['elo_rating'])
                ))
            session.bulk_save_objects(records)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if self.session is None:
                session.close()

    def get_latest_elo(self, team: str, before_date: Optional[Any] = None) -> float:
        """Get the latest ELO rating for a team before a specific date (default 1500)."""
        session = self.get_session()
        try:
            query = session.query(RawEloRating).filter(RawEloRating.team == team)
            if before_date:
                if isinstance(before_date, str):
                    before_date = datetime.strptime(before_date, '%Y-%m-%d').date()
                elif isinstance(before_date, datetime):
                    before_date = before_date.date()
                query = query.filter(RawEloRating.date < before_date)
            
            latest = query.order_by(desc(RawEloRating.date)).first()
            return latest.elo_rating if latest else 1500.0
        finally:
            if self.session is None:
                session.close()

    def save_team_features(self, df: pd.DataFrame):
        """Bulk save team features."""
        session = self.get_session()
        try:
            records = []
            for _, row in df.iterrows():
                d = row['date']
                if isinstance(d, str):
                    d = datetime.strptime(d, '%Y-%m-%d').date()
                elif isinstance(d, datetime):
                    d = d.date()
                records.append(TeamFeature(
                    team=row['team'],
                    date=d,
                    elo_rating=float(row['elo_rating']),
                    recent_win_rate=float(row['recent_win_rate']),
                    goals_scored_avg=float(row['goals_scored_avg']),
                    goals_conceded_avg=float(row['goals_conceded_avg']),
                    attack_rating=float(row['attack_rating']),
                    defense_rating=float(row['defense_rating']),
                    form_score=float(row['form_score']),
                    matches_played=int(row['matches_played'])
                ))
            session.bulk_save_objects(records)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if self.session is None:
                session.close()

    def save_match_features(self, df: pd.DataFrame):
        """Bulk save match features."""
        session = self.get_session()
        try:
            records = []
            for _, row in df.iterrows():
                d = row['date']
                if isinstance(d, str):
                    d = datetime.strptime(d, '%Y-%m-%d').date()
                elif isinstance(d, datetime):
                    d = d.date()
                
                # Check for nan/nullable values
                result_val = row.get('result')
                if pd.isna(result_val):
                    result_val = None
                else:
                    result_val = int(result_val)
                    
                records.append(MatchFeature(
                    match_id=int(row['match_id']) if not pd.isna(row.get('match_id')) else None,
                    date=d,
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    elo_difference=float(row['elo_difference']) if not pd.isna(row.get('elo_difference')) else None,
                    attack_difference=float(row['attack_difference']) if not pd.isna(row.get('attack_difference')) else None,
                    defense_difference=float(row['defense_difference']) if not pd.isna(row.get('defense_difference')) else None,
                    form_difference=float(row['form_difference']) if not pd.isna(row.get('form_difference')) else None,
                    goals_scored_diff=float(row['goals_scored_diff']) if not pd.isna(row.get('goals_scored_diff')) else None,
                    goals_conceded_diff=float(row['goals_conceded_diff']) if not pd.isna(row.get('goals_conceded_diff')) else None,
                    h2h_advantage=float(row['h2h_advantage']) if not pd.isna(row.get('h2h_advantage')) else None,
                    is_neutral_venue=int(row['is_neutral_venue']) if not pd.isna(row.get('is_neutral_venue')) else 0,
                    tournament_importance=float(row['tournament_importance']) if not pd.isna(row.get('tournament_importance')) else 1.0,
                    result=result_val,
                    home_goals=int(row['home_goals']) if not pd.isna(row.get('home_goals')) else None,
                    away_goals=int(row['away_goals']) if not pd.isna(row.get('away_goals')) else None
                ))
            session.bulk_save_objects(records)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if self.session is None:
                session.close()

    def get_latest_team_features(self, team: str) -> Optional[Dict[str, Any]]:
        """Get the latest feature record for a team as a dictionary."""
        session = self.get_session()
        try:
            latest = session.query(TeamFeature).filter(
                TeamFeature.team == team
            ).order_by(desc(TeamFeature.date)).first()
            
            if not latest:
                return None
                
            return {
                'team': latest.team,
                'date': latest.date,
                'elo_rating': latest.elo_rating,
                'recent_win_rate': latest.recent_win_rate,
                'goals_scored_avg': latest.goals_scored_avg,
                'goals_conceded_avg': latest.goals_conceded_avg,
                'attack_rating': latest.attack_rating,
                'defense_rating': latest.defense_rating,
                'form_score': latest.form_score,
                'matches_played': latest.matches_played
            }
        finally:
            if self.session is None:
                session.close()

    def save_model_version(self, name: str, version: str, metrics: Dict[str, float], model_path: str) -> int:
        """
        Save training metrics and metadata for a model version.

        Metrics are routed to the column that matches them: classifiers populate
        accuracy/log_loss/f1_score, regressors populate mae/rmse, and whichever
        set does not apply stays NULL. Anything absent from `metrics` is simply
        not written, so callers pass only what they actually measured.

        A given model name and version occupies one row, updated in place. This
        previously inserted unconditionally, so every pipeline run added three
        rows describing the same three models - 51 rows for 3 models after 17
        runs - which made the training history unreadable and hid the fact that
        the recorded metrics no longer matched the committed artifacts.

        Rows for earlier runs are left untouched rather than cleaned up here:
        `predictions.model_id` points at them, so deleting them would orphan
        prediction records.
        """
        session = self.get_session()
        try:
            # Only one row per model name may be active.
            session.query(ModelVersion).filter(
                ModelVersion.model_name == name
            ).update({ModelVersion.is_active: False})

            # Reuse the row for this name and version if one exists. Where older
            # duplicates are present, the most recent is the one carried forward.
            record = session.query(ModelVersion).filter(
                ModelVersion.model_name == name,
                ModelVersion.version == version
            ).order_by(desc(ModelVersion.model_id)).first()

            if record is None:
                record = ModelVersion(model_name=name, version=version)
                session.add(record)

            record.training_date = date.today()
            record.accuracy = metrics.get('accuracy')
            record.log_loss = metrics.get('log_loss')
            record.f1_score = metrics.get('f1_macro')
            record.mae = metrics.get('mae')
            record.rmse = metrics.get('rmse')
            record.model_path = model_path
            record.is_active = True

            session.commit()
            return record.model_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if self.session is None:
                session.close()

    def save_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Save prediction outcome probabilities."""
        session = self.get_session()
        try:
            pred = Prediction(
                model_id=prediction_data.get('model_id'),
                home_team=prediction_data['home_team'],
                away_team=prediction_data['away_team'],
                home_win_prob=float(prediction_data['home_win_prob']),
                draw_prob=float(prediction_data['draw_prob']),
                away_win_prob=float(prediction_data['away_win_prob']),
                expected_home_goals=float(prediction_data['expected_home_goals']) if prediction_data.get('expected_home_goals') is not None else None,
                expected_away_goals=float(prediction_data['expected_away_goals']) if prediction_data.get('expected_away_goals') is not None else None
            )
            session.add(pred)
            session.commit()
            return pred.prediction_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if self.session is None:
                session.close()

    def save_prediction_explanations(self, prediction_id: int, explanations: List[Dict[str, Any]]):
        """Save SHAP explanations for a prediction."""
        session = self.get_session()
        try:
            records = []
            for exp in explanations:
                records.append(PredictionExplanation(
                    prediction_id=prediction_id,
                    feature_name=exp['feature_name'],
                    feature_value=float(exp['feature_value']) if exp.get('feature_value') is not None else None,
                    shap_value_home=float(exp.get('shap_value_home', 0.0)),
                    shap_value_draw=float(exp.get('shap_value_draw', 0.0)),
                    shap_value_away=float(exp.get('shap_value_away', 0.0))
                ))
            session.bulk_save_objects(records)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if self.session is None:
                session.close()

    def get_active_model(self) -> Optional[ModelVersion]:
        """Fetch the active model metadata."""
        session = self.get_session()
        try:
            model = session.query(ModelVersion).filter(
                ModelVersion.is_active == True
            ).order_by(desc(ModelVersion.training_date)).first()
            return model
        finally:
            if self.session is None:
                session.close()

    def get_all_match_features(self) -> pd.DataFrame:
        """Fetch all match features as a DataFrame for training."""
        session = self.get_session()
        try:
            query = session.query(MatchFeature).order_by(MatchFeature.date)
            df = pd.read_sql(query.statement, self.engine)
            return df
        finally:
            if self.session is None:
                session.close()
