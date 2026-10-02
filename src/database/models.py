from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class RawMatch(Base):
    __tablename__ = 'raw_matches'

    match_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_score = Column(Integer, nullable=False)
    away_score = Column(Integer, nullable=False)
    tournament = Column(String)
    city = Column(String)
    country = Column(String)
    neutral = Column(Boolean, default=False)

class RawEloRating(Base):
    __tablename__ = 'raw_elo_ratings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    team = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    elo_rating = Column(Float, nullable=False)

class RawTeamStatistic(Base):
    __tablename__ = 'raw_team_statistics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('raw_matches.match_id'))
    team = Column(String, nullable=False)
    shots = Column(Integer)
    possession = Column(Float)
    xg = Column(Float)

class TeamFeature(Base):
    __tablename__ = 'team_features'

    feature_id = Column(Integer, primary_key=True, autoincrement=True)
    team = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    elo_rating = Column(Float)
    recent_win_rate = Column(Float)
    goals_scored_avg = Column(Float)
    goals_conceded_avg = Column(Float)
    attack_rating = Column(Float)
    defense_rating = Column(Float)
    form_score = Column(Float)
    matches_played = Column(Integer)

class MatchFeature(Base):
    __tablename__ = 'match_features'

    match_feature_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('raw_matches.match_id'))
    date = Column(Date, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    elo_difference = Column(Float)
    attack_difference = Column(Float)
    defense_difference = Column(Float)
    form_difference = Column(Float)
    goals_scored_diff = Column(Float)
    goals_conceded_diff = Column(Float)
    h2h_advantage = Column(Float)
    is_neutral_venue = Column(Integer)
    tournament_importance = Column(Float)
    result = Column(Integer)  # 0=away win, 1=draw, 2=home win
    home_goals = Column(Integer)
    away_goals = Column(Integer)

class ModelVersion(Base):
    __tablename__ = 'model_versions'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    training_date = Column(Date, nullable=False)
    # Classifier metrics; left NULL for regressors.
    accuracy = Column(Float)
    log_loss = Column(Float)
    f1_score = Column(Float)
    # Regressor metrics; left NULL for the classifier. Before these existed,
    # MAE and RMSE were written into accuracy and log_loss, which made a
    # regressor look like it had an accuracy of 1.05.
    mae = Column(Float)
    rmse = Column(Float)
    model_path = Column(String)
    is_active = Column(Boolean, default=True)

class Prediction(Base):
    __tablename__ = 'predictions'

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('model_versions.model_id'))
    date = Column(DateTime, default=datetime.utcnow)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    expected_home_goals = Column(Float)
    expected_away_goals = Column(Float)

class PredictionExplanation(Base):
    __tablename__ = 'prediction_explanations'

    explanation_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.prediction_id'))
    feature_name = Column(String, nullable=False)
    feature_value = Column(Float)
    shap_value_home = Column(Float)
    shap_value_draw = Column(Float)
    shap_value_away = Column(Float)

def create_all_tables(engine):
    Base.metadata.create_all(engine)
