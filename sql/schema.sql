-- FootyForecast - Database Schema
-- Three-layer architecture: Raw -> Feature -> ML
-- RAW LAYER - Original data from external sources

CREATE TABLE IF NOT EXISTS raw_matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    tournament TEXT,
    city TEXT,
    country TEXT,
    neutral BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS raw_elo_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    date DATE NOT NULL,
    elo_rating REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_team_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    team TEXT NOT NULL,
    shots INTEGER,
    possession REAL,
    xg REAL,
    FOREIGN KEY (match_id) REFERENCES raw_matches(match_id)
);

-- FEATURE LAYER - Engineered ML features


CREATE TABLE IF NOT EXISTS team_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    date DATE NOT NULL,
    elo_rating REAL,
    recent_win_rate REAL,
    goals_scored_avg REAL,
    goals_conceded_avg REAL,
    attack_rating REAL,
    defense_rating REAL,
    form_score REAL,
    matches_played INTEGER
);

CREATE TABLE IF NOT EXISTS match_features (
    match_feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    elo_difference REAL,
    attack_difference REAL,
    defense_difference REAL,
    form_difference REAL,
    goals_scored_diff REAL,
    goals_conceded_diff REAL,
    h2h_advantage REAL,
    is_neutral_venue INTEGER,
    tournament_importance INTEGER,
    result INTEGER,  -- 0=away win, 1=draw, 2=home win
    home_goals INTEGER,
    away_goals INTEGER,
    FOREIGN KEY (match_id) REFERENCES raw_matches(match_id)
);

-- ============================================================
-- ML LAYER - Models, predictions, explanations
-- ============================================================

CREATE TABLE IF NOT EXISTS model_versions (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    training_date DATE NOT NULL,
    accuracy REAL,
    log_loss REAL,
    f1_score REAL,
    model_path TEXT,
    is_active BOOLEAN DEFAULT 1
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_win_prob REAL,
    draw_prob REAL,
    away_win_prob REAL,
    expected_home_goals REAL,
    expected_away_goals REAL,
    FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
);

CREATE TABLE IF NOT EXISTS prediction_explanations (
    explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER,
    feature_name TEXT NOT NULL,
    feature_value REAL,
    shap_value_home REAL,
    shap_value_draw REAL,
    shap_value_away REAL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_raw_matches_date ON raw_matches(date);
CREATE INDEX IF NOT EXISTS idx_raw_matches_teams ON raw_matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_raw_elo_team_date ON raw_elo_ratings(team, date);
CREATE INDEX IF NOT EXISTS idx_team_features_team_date ON team_features(team, date);
CREATE INDEX IF NOT EXISTS idx_match_features_date ON match_features(date);
