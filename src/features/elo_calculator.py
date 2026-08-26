import os
import math
import pandas as pd
from typing import Dict, Tuple

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
        if not isinstance(tournament, str):
            return 30.0
            
        t_lower = tournament.lower()
        if 'world cup' in t_lower and 'qualifying' not in t_lower:
            return 60.0
        elif 'euro' in t_lower or 'copa' in t_lower or 'nations cup' in t_lower or 'afcon' in t_lower or 'asian cup' in t_lower or 'gold cup' in t_lower:
            # Major continental tournaments
            return 50.0
        elif 'qualify' in t_lower or 'qualification' in t_lower or 'nations league' in t_lower:
            return 40.0
        elif 'friendly' in t_lower:
            return 20.0
        else:
            return 30.0

    def update_ratings(self, home_team: str, away_team: str, home_score: int, away_score: int, tournament: str, neutral: bool = False) -> Tuple[float, float]:
        """Calculate new Elo ratings for home and away teams and update self.ratings."""
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        
        # Apply home advantage to expectation calculation if not a neutral venue
        eff_r_home = r_home + (0.0 if neutral else self.home_advantage)
        eff_r_away = r_away
        
        exp_home = self.expected_score(eff_r_home, eff_r_away)
        exp_away = 1.0 - exp_home
        
        # Determine actual result (from home team perspective)
        # 1.0 = Home Win, 0.5 = Draw, 0.0 = Away Win
        if home_score > away_score:
            act_home = 1.0
        elif home_score < away_score:
            act_home = 0.0
        else:
            act_home = 0.5
            
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
            
            # Record Elo rating BEFORE the match for features
            home_elo_before = self.get_rating(home)
            away_elo_before = self.get_rating(away)
            
            # Update ratings based on the match outcome
            home_elo_after, away_elo_after = self.update_ratings(
                home, away, home_s, away_s, tournament, neutral
            )
            
            # Record Elo rating AFTER the match
            elo_history.append({'date': d, 'team': home, 'elo_rating': home_elo_after})
            elo_history.append({'date': d, 'team': away, 'elo_rating': away_elo_after})
            
        return pd.DataFrame(elo_history)

if __name__ == '__main__':
    # Simple demo/test
    calc = EloCalculator()
    print("Expected prob (1500 vs 1500):", calc.expected_score(1500, 1500))
    print("Expected prob (1600 vs 1500):", calc.expected_score(1600, 1500))
    print("K-factor for WC:", calc.get_k_factor("FIFA World Cup"))
    print("Update ratings (Brazil 2 - 0 Argentina, Friendly):", calc.update_ratings("Brazil", "Argentina", 2, 0, "Friendly"))
