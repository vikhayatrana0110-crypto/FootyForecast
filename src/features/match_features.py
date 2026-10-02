import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.features.elo_calculator import tournament_tier

TOURNAMENT_IMPORTANCE = {'world_cup': 3.0, 'continental': 2.5, 'qualifier': 1.8, 'friendly': 1.0, 'other': 1.5, 'unknown': 1.0}

def get_tournament_importance(tournament: str) -> float:
    """Get the importance score of a tournament (1.0 to 3.0)."""
    return TOURNAMENT_IMPORTANCE[tournament_tier(tournament)]

def compute_h2h(matches_df: pd.DataFrame, team_a: str, team_b: str, before_date: Optional[Any] = None, max_matches: int = 10) -> float:
    """Mean result (+1/0/-1) of team_a's last max_matches games against team_b."""
    h2h = matches_df[
        ((matches_df['home_team'] == team_a) & (matches_df['away_team'] == team_b)) |
        ((matches_df['home_team'] == team_b) & (matches_df['away_team'] == team_a))
    ].copy()

    if before_date:
        before_date = pd.to_datetime(before_date)
        h2h['date'] = pd.to_datetime(h2h['date'])
        h2h = h2h[h2h['date'] < before_date]

    if h2h.empty:
        return 0.0

    h2h = h2h.sort_values('date', ascending=False).head(max_matches)

    pts = np.sign(h2h['home_score'] - h2h['away_score'])
    return float(np.where(h2h['home_team'] == team_a, pts, -pts).mean())

def compute_h2h_series(matches_df: pd.DataFrame, max_matches: int = 10) -> pd.Series:
    """compute_h2h() for every row in a single pass instead of rescanning per row.

    Meetings on the same day don't see each other's results.
    """
    dates = pd.to_datetime(matches_df['date']).values
    home = matches_df['home_team'].values
    away = matches_df['away_team'].values
    h_score = matches_df['home_score'].values
    away_score = matches_df['away_score'].values

    groups: Dict[tuple, List[int]] = {}
    for pos in range(len(matches_df)):
        key = (home[pos], away[pos]) if home[pos] <= away[pos] else (away[pos], home[pos])
        groups.setdefault(key, []).append(pos)

    out = np.zeros(len(matches_df), dtype=float)

    for (team_a, _team_b), positions in groups.items():
        positions.sort(key=lambda p: dates[p])
        history: List[float] = []

        i = 0
        n = len(positions)
        while i < n:
            j = i
            while j < n and dates[positions[j]] == dates[positions[i]]:
                j += 1

            window = history[-max_matches:]
            if window:
                mean_pts = sum(window) / len(window)
                for k in range(i, j):
                    pos = positions[k]
                    out[pos] = mean_pts if home[pos] == team_a else -mean_pts

            for k in range(i, j):
                pos = positions[k]
                if h_score[pos] > away_score[pos]:
                    pts = 1.0
                elif h_score[pos] < away_score[pos]:
                    pts = -1.0
                else:
                    pts = 0.0
                history.append(pts if home[pos] == team_a else -pts)

            i = j

    return pd.Series(out, index=matches_df.index)


class MatchFeatureEngine:
    def get_feature_columns(self) -> List[str]:
        """Return the list of features used in the ML model."""
        return [
            'elo_difference',
            'attack_difference',
            'defense_difference',
            'form_difference',
            'goals_scored_diff',
            'goals_conceded_diff',
            'h2h_advantage',
            'is_neutral_venue',
            'tournament_importance'
        ]

    def compute_match_features(self, matches_df: pd.DataFrame, team_features_df: pd.DataFrame) -> pd.DataFrame:
        """Join home and away team features onto each match and build the difference features."""
        matches = matches_df.copy()
        matches['date'] = pd.to_datetime(matches['date'])

        team_feats = team_features_df.copy()
        team_feats['date'] = pd.to_datetime(team_feats['date'])

        home_feats = team_feats.rename(columns={col: f'home_{col}' for col in team_feats.columns if col != 'date'})
        matches = matches.merge(home_feats, on=['date', 'home_team'])
        away_feats = team_feats.rename(columns={col: f'away_{col}' for col in team_feats.columns if col != 'date'})
        matches = matches.merge(away_feats, on=['date', 'away_team'])

        matches['elo_difference'] = matches['home_elo_rating'] - matches['away_elo_rating']
        matches['attack_difference'] = matches['home_attack_rating'] - matches['away_attack_rating']
        matches['defense_difference'] = matches['home_defense_rating'] - matches['away_defense_rating']
        matches['form_difference'] = matches['home_form_score'] - matches['away_form_score']
        matches['goals_scored_diff'] = matches['home_goals_scored_avg'] - matches['away_goals_scored_avg']
        matches['goals_conceded_diff'] = matches['home_goals_conceded_avg'] - matches['away_goals_conceded_avg']

        print("Calculating H2H advantage for all matches...")
        matches['h2h_advantage'] = compute_h2h_series(matches, max_matches=10)
        matches['is_neutral_venue'] = matches['neutral'].astype(int)
        matches['tournament_importance'] = matches['tournament'].apply(get_tournament_importance)

        matches['result'] = np.select(
            [
                matches['home_score'] > matches['away_score'],
                matches['home_score'] == matches['away_score']
            ],
            [2, 1],
            default=0
        )

        return matches.rename(columns={'home_score': 'home_goals', 'away_score': 'away_goals'})

    def create_prediction_features(
        self,
        home_team: str,
        away_team: str,
        tournament: str,
        team_features_df: pd.DataFrame,
        matches_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Feature vector for an upcoming match, built from each team's latest features."""
        defaults = {'elo_rating': 1500.0, 'attack_rating': 1.33, 'defense_rating': 0.5,
                    'form_score': 1.0, 'goals_scored_avg': 1.0, 'goals_conceded_avg': 1.0}

        def latest(team):
            rows = team_features_df[team_features_df['team'] == team].sort_values('date', ascending=False).head(1)
            return {k: d if rows.empty else rows[k].values[0] for k, d in defaults.items()}

        home, away = latest(home_team), latest(away_team)
        diff = {k: home[k] - away[k] for k in defaults}

        h2h_adv = compute_h2h(matches_df, home_team, away_team)

        is_neutral = 1 if 'world cup' in tournament.lower() or 'neutral' in tournament.lower() else 0
        tourn_imp = get_tournament_importance(tournament)

        return {
            'elo_difference': diff['elo_rating'],
            'attack_difference': diff['attack_rating'],
            'defense_difference': diff['defense_rating'],
            'form_difference': diff['form_score'],
            'goals_scored_diff': diff['goals_scored_avg'],
            'goals_conceded_diff': diff['goals_conceded_avg'],
            'h2h_advantage': h2h_adv,
            'is_neutral_venue': is_neutral,
            'tournament_importance': tourn_imp
        }
