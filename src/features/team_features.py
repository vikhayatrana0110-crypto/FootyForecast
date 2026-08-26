import pandas as pd
import numpy as np
from typing import Dict, Any

class TeamFeatureEngine:
    def __init__(self, window: int = 10):
        self.window = window

    def compute_team_features(self, matches_df: pd.DataFrame, elo_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute team-level features for each team and match date.
        CRITICAL: All features for a team at match date T must only use information BEFORE date T.
        """
        # Sort matches chronologically
        matches = matches_df.copy()
        matches['date'] = pd.to_datetime(matches['date'])
        matches = matches.sort_values('date').reset_index(drop=True)
        
        # Sort ELO history
        elo_hist = elo_history_df.copy()
        elo_hist['date'] = pd.to_datetime(elo_hist['date'])
        elo_hist = elo_hist.sort_values('date').reset_index(drop=True)
        
        all_team_features = []
        teams = set(matches['home_team']).union(set(matches['away_team']))
        
        for team in teams:
            # Filter matches involving this team
            team_matches = matches[(matches['home_team'] == team) | (matches['away_team'] == team)].copy()
            
            # Determine goals scored/conceded and results
            team_matches['is_home'] = team_matches['home_team'] == team
            team_matches['goals_scored'] = np.where(team_matches['is_home'], team_matches['home_score'], team_matches['away_score'])
            team_matches['goals_conceded'] = np.where(team_matches['is_home'], team_matches['away_score'], team_matches['home_score'])
            
            # 3 for win, 1 for draw, 0 for loss
            team_matches['points'] = np.select(
                [
                    team_matches['goals_scored'] > team_matches['goals_conceded'],
                    team_matches['goals_scored'] == team_matches['goals_conceded']
                ],
                [3, 1],
                default=0
            )
            team_matches['is_win'] = np.where(team_matches['goals_scored'] > team_matches['goals_conceded'], 1, 0)
            
            # Compute rolling features shifted by 1 (state BEFORE the match)
            # Goals avg
            team_matches['goals_scored_avg'] = team_matches['goals_scored'].rolling(window=self.window, min_periods=1).mean().shift(1)
            team_matches['goals_conceded_avg'] = team_matches['goals_conceded'].rolling(window=self.window, min_periods=1).mean().shift(1)
            
            # Win rate
            team_matches['recent_win_rate'] = team_matches['is_win'].rolling(window=self.window, min_periods=1).mean().shift(1)
            
            # Total matches played before this match
            team_matches['matches_played'] = np.arange(len(team_matches))

            
            # Form score: Exponentially decaying weighted average of points in recent matches
            # Let's compute it with ewm (exponential moving average) shifted by 1
            team_matches['form_score'] = team_matches['points'].ewm(span=self.window, adjust=False).mean().shift(1)
            
            # Fill NaNs with defaults for the very first matches
            team_matches['goals_scored_avg'] = team_matches['goals_scored_avg'].fillna(1.0)
            team_matches['goals_conceded_avg'] = team_matches['goals_conceded_avg'].fillna(1.0)
            team_matches['recent_win_rate'] = team_matches['recent_win_rate'].fillna(0.33)
            team_matches['form_score'] = team_matches['form_score'].fillna(1.0)  # average of 1.0 points per game (draw)
            
            # Composite ratings
            team_matches['attack_rating'] = team_matches['goals_scored_avg'] * (1.0 + team_matches['recent_win_rate'])
            team_matches['defense_rating'] = 1.0 / (1.0 + team_matches['goals_conceded_avg'])
            
            # Get ELO rating BEFORE each match date
            # We look at the ELO history for this team
            team_elo = elo_hist[elo_hist['team'] == team].copy()
            
            # Merging Elo rating
            # We want the Elo rating after the team's PREVIOUS match. 
            # We can shift the Elo ratings by 1 within the team Elo series.
            team_elo['elo_rating_before'] = team_elo['elo_rating'].shift(1).fillna(1500.0)
            
            # Merge team_matches and team_elo on date
            team_features = pd.merge_asof(
                team_matches[['date', 'goals_scored_avg', 'goals_conceded_avg', 'recent_win_rate', 'matches_played', 'form_score', 'attack_rating', 'defense_rating']],
                team_elo[['date', 'elo_rating_before']],
                on='date',
                direction='backward'
            )
            
            team_features['team'] = team
            team_features = team_features.rename(columns={'elo_rating_before': 'elo_rating'})
            all_team_features.append(team_features)
            
        # Combine all teams
        features_df = pd.concat(all_team_features, ignore_index=True)
        return features_df
