import pandas as pd
from typing import Dict, Tuple

def tournament_tier(tournament: str) -> str:
    """Classify a tournament name; shared by the Elo K-factor and the importance feature."""
    if not isinstance(tournament, str):
        return 'unknown'
    t = tournament.lower()
    if 'world cup' in t and 'qualifying' not in t:
        return 'world_cup'
    if any(k in t for k in ('euro', 'copa', 'nations cup', 'afcon', 'asian cup', 'gold cup')):
        return 'continental'
    if any(k in t for k in ('qualify', 'qualification', 'nations league')):
        return 'qualifier'
    if 'friendly' in t:
        return 'friendly'
    return 'other'

K_FACTORS = {'world_cup': 60.0, 'continental': 50.0, 'qualifier': 40.0, 'friendly': 20.0, 'other': 30.0, 'unknown': 30.0}

class EloCalculator:
    def __init__(self, k_factor: int = 30, home_advantage: float = 100.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Get current Elo rating of a team, default to initial_rating."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score/probability for team A."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def get_k_factor(self, tournament: str) -> float:
        """Return K-factor based on tournament importance."""
        return K_FACTORS[tournament_tier(tournament)]

    def update_ratings(self, home_team: str, away_team: str, home_score: int, away_score: int, tournament: str, neutral: bool = False) -> Tuple[float, float]:
        """Calculate new Elo ratings for home and away teams and update self.ratings."""
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        
        # Apply home advantage to expectation calculation if not a neutral venue
        eff_r_home = r_home + (0.0 if neutral else self.home_advantage)
        eff_r_away = r_away
        
        exp_home = self.expected_score(eff_r_home, eff_r_away)
        exp_away = 1.0 - exp_home
        
        # Actual result from home team perspective: 1.0 win, 0.5 draw, 0.0 loss
        act_home = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
        act_away = 1.0 - act_home
        
        # Goal difference multiplier (G)
        goal_diff = abs(home_score - away_score)
        if goal_diff <= 1:
            g = 1.0
        elif goal_diff == 2:
            g = 1.5
        else:
            g = (11.0 + goal_diff) / 8.0
            
        k = self.get_k_factor(tournament)
        
        # Update ratings
        new_r_home = r_home + k * g * (act_home - exp_home)
        new_r_away = r_away + k * g * (act_away - exp_away)
        
        # Save updated ratings
        self.ratings[home_team] = round(new_r_home, 1)
        self.ratings[away_team] = round(new_r_away, 1)
        
        return self.ratings[home_team], self.ratings[away_team]

    def compute_all_ratings(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process matches chronologically to calculate historical Elo rating sequences.
        Returns a DataFrame with [date, team, elo_rating] suitable for the raw_elo_ratings table.
        """
        # Ensure chronological order
        df = matches_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        elo_history = []
        
        for _, row in df.iterrows():
            d = row['date']
            home = row['home_team']
            away = row['away_team']
            home_s = int(row['home_score'])
            away_s = int(row['away_score'])
            tournament = row['tournament']
            neutral = bool(row['neutral'])
            
            # Update ratings based on the match outcome
            home_elo_after, away_elo_after = self.update_ratings(
                home, away, home_s, away_s, tournament, neutral
            )
            
            # Record Elo rating AFTER the match
            elo_history.append({'date': d, 'team': home, 'elo_rating': home_elo_after})
            elo_history.append({'date': d, 'team': away, 'elo_rating': away_elo_after})
            
        return pd.DataFrame(elo_history)
