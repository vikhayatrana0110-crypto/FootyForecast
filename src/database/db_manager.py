import os
import pandas as pd
from datetime import date
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from sqlalchemy import create_engine, select, desc, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
load_dotenv()

from src.database.models import (
    create_all_tables, RawMatch, RawEloRating,
    TeamFeature, MatchFeature, ModelVersion,
    Prediction, PredictionExplanation
)

class DatabaseManager:
    def __init__(self):
        """Connect to Supabase Postgres using credentials from the environment (.env or Streamlit secrets)."""
        DB_HOST = os.getenv("SUPABASE_HOST")
        DB_PASSWORD = os.getenv("SUPABASE_PASSWORD")

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

        DATABASE_URL = (
            f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def init_db(self):
        create_all_tables(self.engine)
        self._apply_column_migrations()

    def _apply_column_migrations(self):
        """create_all() won't add columns to existing tables, so newer columns are added here."""
        statements = [
            "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS mae DOUBLE PRECISION",
            "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS rmse DOUBLE PRECISION",
        ]
        with self.engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))

    def is_empty(self, model) -> bool:
        with self.SessionLocal() as session:
            return session.query(model).first() is None

    def _append(self, df: pd.DataFrame, model):
        """Append the DataFrame columns that exist on the model's table. NaN is stored as NULL."""
        cols = [c.name for c in model.__table__.columns if not c.primary_key and c.name in df.columns]
        out = df[cols].copy()
        out['date'] = pd.to_datetime(out['date']).dt.date
        out.to_sql(model.__tablename__, self.engine, if_exists='append', index=False)

    def bulk_insert_matches(self, df: pd.DataFrame):
        self._append(df, RawMatch)

    def save_elo_ratings(self, df: pd.DataFrame):
        self._append(df, RawEloRating)

    def save_team_features(self, df: pd.DataFrame):
        self._append(df, TeamFeature)

    def save_match_features(self, df: pd.DataFrame):
        self._append(df.fillna({'is_neutral_venue': 0, 'tournament_importance': 1.0}), MatchFeature)

    def get_all_matches(self) -> pd.DataFrame:
        return pd.read_sql(select(RawMatch).order_by(RawMatch.date), self.engine)

    def get_all_match_features(self) -> pd.DataFrame:
        return pd.read_sql(select(MatchFeature).order_by(MatchFeature.date), self.engine)

    def get_all_teams(self) -> List[str]:
        with self.SessionLocal() as session:
            home = session.scalars(select(RawMatch.home_team).distinct()).all()
            away = session.scalars(select(RawMatch.away_team).distinct()).all()
        return sorted(set(home) | set(away))

    def get_latest_team_features(self, team: str) -> Optional[Dict[str, Any]]:
        """Get the latest feature record for a team as a dictionary."""
        with self.SessionLocal() as session:
            latest = session.query(TeamFeature).filter(
                TeamFeature.team == team
            ).order_by(desc(TeamFeature.date)).first()
        if not latest:
            return None
        return {c.name: getattr(latest, c.name) for c in TeamFeature.__table__.columns if c.name != 'feature_id'}

    def save_model_version(self, name: str, version: str, metrics: Dict[str, float], model_path: str) -> int:
        """One row per (name, version), updated in place, and the only active row for that name.

        Old versions are kept since predictions reference them.
        """
        with self.SessionLocal.begin() as session:
            session.query(ModelVersion).filter(
                ModelVersion.model_name == name
            ).update({ModelVersion.is_active: False})

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

            session.flush()
            return record.model_id

    def save_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Save prediction outcome probabilities."""
        cols = [c.name for c in Prediction.__table__.columns if c.name not in ('prediction_id', 'date')]
        with self.SessionLocal.begin() as session:
            pred = Prediction(**{c: prediction_data.get(c) for c in cols})
            session.add(pred)
            session.flush()
            return pred.prediction_id

    def save_prediction_explanations(self, prediction_id: int, explanations: List[Dict[str, Any]]):
        """Save SHAP explanations for a prediction."""
        with self.SessionLocal.begin() as session:
            session.add_all(PredictionExplanation(prediction_id=prediction_id, **exp) for exp in explanations)

    def get_active_model(self) -> Optional[ModelVersion]:
        """Fetch the active model metadata."""
        with self.SessionLocal() as session:
            return session.query(ModelVersion).filter(
                ModelVersion.is_active == True
            ).order_by(desc(ModelVersion.training_date)).first()
