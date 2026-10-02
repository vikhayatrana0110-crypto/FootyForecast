import pandas as pd
import numpy as np
from typing import Dict, Any

class TeamFeatureEngine:
    def __init__(self, window: int = 10):
        self.window = window

    def compute_team_features(self, matches_df: pd.DataFrame, elo_history_df: pd.DataFrame) -> pd.DataFrame:
        """Rolling features per team and match date, using only matches played before that date."""
        matches = matches_df.copy()
        matches['date'] = pd.to_datetime(matches['date'])
        matches = matches.sort_values('date').reset_index(drop=True)

        elo_hist = elo_history_df.copy()
        elo_hist['date'] = pd.to_datetime(elo_hist['date'])
        elo_hist = elo_hist.sort_values('date').reset_index(drop=True)

        all_team_features = []
        teams = set(matches['home_team']).union(set(matches['away_team']))

        for team in teams:
            team_matches = matches[(matches['home_team'] == team) | (matches['away_team'] == team)].copy()

            is_home = team_matches['home_team'] == team
            team_matches['goals_scored'] = np.where(is_home, team_matches['home_score'], team_matches['away_score'])
            team_matches['goals_conceded'] = np.where(is_home, team_matches['away_score'], team_matches['home_score'])
            team_matches['points'] = np.select(
                [
                    team_matches['goals_scored'] > team_matches['goals_conceded'],
                    team_matches['goals_scored'] == team_matches['goals_conceded']
                ],
                [3, 1],
                default=0
            )
            team_matches['is_win'] = np.where(team_matches['goals_scored'] > team_matches['goals_conceded'], 1, 0)

            # shift(1) so each row only sees matches before it
            rolling = team_matches[['goals_scored', 'goals_conceded', 'is_win']].rolling(self.window, min_periods=1).mean().shift(1)
            team_matches['goals_scored_avg'] = rolling['goals_scored'].fillna(1.0)
            team_matches['goals_conceded_avg'] = rolling['goals_conceded'].fillna(1.0)
            team_matches['recent_win_rate'] = rolling['is_win'].fillna(0.33)
            team_matches['form_score'] = team_matches['points'].ewm(span=self.window, adjust=False).mean().shift(1).fillna(1.0)
            team_matches['matches_played'] = np.arange(len(team_matches))

            team_matches['attack_rating'] = team_matches['goals_scored_avg'] * (1.0 + team_matches['recent_win_rate'])
            team_matches['defense_rating'] = 1.0 / (1.0 + team_matches['goals_conceded_avg'])

            team_elo = elo_hist[elo_hist['team'] == team].copy()
            team_elo['elo_rating_before'] = team_elo['elo_rating'].shift(1).fillna(1500.0)

            team_features = pd.merge_asof(
                team_matches[['date', 'goals_scored_avg', 'goals_conceded_avg', 'recent_win_rate', 'matches_played', 'form_score', 'attack_rating', 'defense_rating']],
                team_elo[['date', 'elo_rating_before']],
                on='date',
                direction='backward'
            )

            team_features['team'] = team
            all_team_features.append(team_features.rename(columns={'elo_rating_before': 'elo_rating'}))

        return pd.concat(all_team_features, ignore_index=True)
