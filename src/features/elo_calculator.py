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
        home_edge = 0.0 if neutral else self.home_advantage
        exp_home = self.expected_score(r_home + home_edge, r_away)
        act_home = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5

        goal_diff = abs(home_score - away_score)
        g = 1.0 if goal_diff <= 1 else 1.5 if goal_diff == 2 else (11.0 + goal_diff) / 8.0
        delta = self.get_k_factor(tournament) * g * (act_home - exp_home)

        self.ratings[home_team] = round(r_home + delta, 1)
        self.ratings[away_team] = round(r_away - delta, 1)
        return self.ratings[home_team], self.ratings[away_team]

    def compute_all_ratings(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Walk matches in date order and record each team's rating after every match."""
        df = matches_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        elo_history = []
        for m in df.itertuples():
            home_elo, away_elo = self.update_ratings(
                m.home_team, m.away_team, int(m.home_score), int(m.away_score), m.tournament, bool(m.neutral)
            )
            elo_history.append({'date': m.date, 'team': m.home_team, 'elo_rating': home_elo})
            elo_history.append({'date': m.date, 'team': m.away_team, 'elo_rating': away_elo})
        return pd.DataFrame(elo_history)
