import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def get_tournament_importance(tournament: str) -> float:
    """Get the importance score of a tournament (1.0 to 3.0)."""
    if not isinstance(tournament, str):
        return 1.0
    t_lower = tournament.lower()
    if 'world cup' in t_lower and 'qualifying' not in t_lower:
        return 3.0
    elif 'euro' in t_lower or 'copa' in t_lower or 'nations cup' in t_lower or 'afcon' in t_lower or 'asian cup' in t_lower or 'gold cup' in t_lower:
        return 2.5
    elif 'qualify' in t_lower or 'qualification' in t_lower or 'nations league' in t_lower:
        return 1.8
    elif 'friendly' in t_lower:
        return 1.0
    else:
        return 1.5

def compute_h2h(matches_df: pd.DataFrame, team_a: str, team_b: str, before_date: Optional[Any] = None, max_matches: int = 10) -> float:
    """
    Calculate head-to-head advantage score for team_a against team_b.
    Returns value between -1.0 and 1.0. Positive means team_a advantage.
    """
    # Filter matches between A and B
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
        
    # Take last N meetings
    h2h = h2h.sort_values('date', ascending=False).head(max_matches)
    
    points = []
    for _, match in h2h.iterrows():
        home = match['home_team']
        home_s = int(match['home_score'])
        away_s = int(match['away_score'])
        
        # Calculate points from team_a perspective
        if home == team_a:
            if home_s > away_s:
                points.append(1.0) # Win
            elif home_s < away_s:
                points.append(-1.0) # Loss
            else:
                points.append(0.0) # Draw
        else:
            if away_s > home_s:
                points.append(1.0) # Win
            elif away_s < home_s:
                points.append(-1.0) # Loss
            else:
                points.append(0.0) # Draw
                
    return float(np.mean(points))

def compute_h2h_series(matches_df: pd.DataFrame, max_matches: int = 10) -> pd.Series:
    """
    Head-to-head advantage for every match in one pass.

    Equivalent to calling compute_h2h() per row, but linear instead of quadratic:
    the per-row version rescans the whole match table for each of ~25k matches.
    Here each fixture pair keeps a running history, so every match is touched once.

    Matches are grouped by unordered team pair and walked in date order. Results
    are stored from the perspective of the alphabetically-first team of the pair
    and negated when the current home team is the other one - valid because the
    scoring is symmetric (+1 win / 0 draw / -1 loss).

    One deliberate difference from compute_h2h(): when the same pair met more than
    once on the date that falls exactly on the max_matches cutoff, compute_h2h()
    picks which of those meetings to keep via an unstable sort, so its answer is
    arbitrary. This walks them in row order instead, which is deterministic. The
    two agree on every real fixture, where a pair does not meet twice in one day.

    Returns a Series aligned to matches_df.index.
    """
    dates = pd.to_datetime(matches_df['date']).values
    home = matches_df['home_team'].values
    away = matches_df['away_team'].values
    h_score = matches_df['home_score'].values
    away_score = matches_df['away_score'].values

    # Group row positions by unordered pair.
    groups: Dict[tuple, List[int]] = {}
    for pos in range(len(matches_df)):
        key = (home[pos], away[pos]) if home[pos] <= away[pos] else (away[pos], home[pos])
        groups.setdefault(key, []).append(pos)

    out = np.zeros(len(matches_df), dtype=float)

    for (team_a, _team_b), positions in groups.items():
        # Chronological order within the pair.
        positions.sort(key=lambda p: dates[p])
        history: List[float] = []

        i = 0
        n = len(positions)
        while i < n:
            # Take every meeting on the same date together, so that same-day
            # fixtures cannot see each other's results.
            j = i
            while j < n and dates[positions[j]] == dates[positions[i]]:
                j += 1

            window = history[-max_matches:]
            if window:
                mean_pts = sum(window) / len(window)
                for k in range(i, j):
                    pos = positions[k]
                    out[pos] = mean_pts if home[pos] == team_a else -mean_pts
            # else: leave 0.0, matching compute_h2h() on an empty history

            # Only now fold this date's results into the history.
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
    def __init__(self, team_feature_engine: Any = None):
        self.team_feature_engine = team_feature_engine

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

    def compute_match_features(self, matches_df: pd.DataFrame, team_features_df: pd.DataFrame, elo_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge team rolling features into match records and construct match-level feature vectors.
        """
        matches = matches_df.copy()
        matches['date'] = pd.to_datetime(matches['date'])
        
        # Convert date to datetime in team features for joining
        team_feats = team_features_df.copy()
        team_feats['date'] = pd.to_datetime(team_feats['date'])
        
        # Merge home team features
        home_feats = team_feats.rename(columns={col: f'home_{col}' for col in team_feats.columns if col != 'date'})
        matches = pd.merge(
            matches,
            home_feats,
            on=['date', 'home_team'],
            how='inner'
        )
        
        # Merge away team features
        away_feats = team_feats.rename(columns={col: f'away_{col}' for col in team_feats.columns if col != 'date'})
        matches = pd.merge(
            matches,
            away_feats,
            on=['date', 'away_team'],
            how='inner'
        )

        
        # Calculate difference features
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
        
        # Result: 2 = Home Win, 1 = Draw, 0 = Away Win
        matches['result'] = np.select(
            [
                matches['home_score'] > matches['away_score'],
                matches['home_score'] == matches['away_score']
            ],
            [2, 1],
            default=0
        )
        
        # Store home_goals and away_goals for regression/xG modelling
        matches = matches.rename(columns={'home_score': 'home_goals', 'away_score': 'away_goals'})
        
        return matches

    def create_prediction_features(
        self,
        home_team: str,
        away_team: str,
        tournament: str,
        team_features_df: pd.DataFrame,
        elo_history_df: pd.DataFrame,
        matches_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate feature vector for a NEW prediction using the latest available features.
        """
        # Get latest features for Home Team
        home_latest = team_features_df[team_features_df['team'] == home_team].sort_values('date', ascending=False).head(1)
        # Get latest features for Away Team
        away_latest = team_features_df[team_features_df['team'] == away_team].sort_values('date', ascending=False).head(1)
        
        # Defaults if no prior matches
        home_elo = home_latest['elo_rating'].values[0] if not home_latest.empty else 1500.0
        away_elo = away_latest['elo_rating'].values[0] if not away_latest.empty else 1500.0
        
        home_attack = home_latest['attack_rating'].values[0] if not home_latest.empty else 1.33
        away_attack = away_latest['attack_rating'].values[0] if not away_latest.empty else 1.33
        
        home_defense = home_latest['defense_rating'].values[0] if not home_latest.empty else 0.5
        away_defense = away_latest['defense_rating'].values[0] if not away_latest.empty else 0.5
        
        home_form = home_latest['form_score'].values[0] if not home_latest.empty else 1.0
        away_form = away_latest['form_score'].values[0] if not away_latest.empty else 1.0
        
        home_scored_avg = home_latest['goals_scored_avg'].values[0] if not home_latest.empty else 1.0
        away_scored_avg = away_latest['goals_scored_avg'].values[0] if not away_latest.empty else 1.0
        
        home_conceded_avg = home_latest['goals_conceded_avg'].values[0] if not home_latest.empty else 1.0
        away_conceded_avg = away_latest['goals_conceded_avg'].values[0] if not away_latest.empty else 1.0
        
        # Calculate difference features
        elo_diff = home_elo - away_elo
        attack_diff = home_attack - away_attack
        defense_diff = home_defense - away_defense
        form_diff = home_form - away_form
        goals_scored_diff = home_scored_avg - away_scored_avg
        goals_conceded_diff = home_conceded_avg - away_conceded_avg
        
        # H2H advantage
        h2h_adv = compute_h2h(matches_df, home_team, away_team)
        
        # Assume World Cup or other neutral venue logic depending on inputs, or set neutral venue = 0 for default
        # For prediction, we can assume non-neutral unless we override
        is_neutral = 1 if 'world cup' in tournament.lower() or 'neutral' in tournament.lower() else 0
        tourn_imp = get_tournament_importance(tournament)
        
        return {
            'elo_difference': elo_diff,
            'attack_difference': attack_diff,
            'defense_difference': defense_diff,
            'form_difference': form_diff,
            'goals_scored_diff': goals_scored_diff,
            'goals_conceded_diff': goals_conceded_diff,
            'h2h_advantage': h2h_adv,
            'is_neutral_venue': is_neutral,
            'tournament_importance': tourn_imp
        }
