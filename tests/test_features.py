import unittest
import pandas as pd
from src.features.team_features import TeamFeatureEngine
from src.features.match_features import MatchFeatureEngine, compute_h2h, compute_h2h_series

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.matches = pd.DataFrame([
            {'match_id': 1, 'date': '2026-06-01', 'home_team': 'France', 'away_team': 'England', 'home_score': 2, 'away_score': 1, 'tournament': 'Friendly', 'neutral': False},
            {'match_id': 2, 'date': '2026-06-05', 'home_team': 'England', 'away_team': 'Germany', 'home_score': 3, 'away_score': 2, 'tournament': 'Friendly', 'neutral': False},
            {'match_id': 3, 'date': '2026-06-10', 'home_team': 'France', 'away_team': 'Germany', 'home_score': 1, 'away_score': 1, 'tournament': 'Friendly', 'neutral': False}
        ])
        self.elo_history = pd.DataFrame([
            {'date': '2026-06-01', 'team': 'France', 'elo_rating': 1515.0},
            {'date': '2026-06-01', 'team': 'England', 'elo_rating': 1485.0},
            {'date': '2026-06-05', 'team': 'England', 'elo_rating': 1500.0},
            {'date': '2026-06-05', 'team': 'Germany', 'elo_rating': 1485.0},
            {'date': '2026-06-10', 'team': 'France', 'elo_rating': 1515.0},
            {'date': '2026-06-10', 'team': 'Germany', 'elo_rating': 1485.0}
        ])

    def test_team_features(self):
        engine = TeamFeatureEngine(window=5)
        team_features = engine.compute_team_features(self.matches, self.elo_history)
        
        # Test columns
        expected_cols = ['team', 'date', 'elo_rating', 'recent_win_rate', 'goals_scored_avg', 'goals_conceded_avg', 'attack_rating', 'defense_rating', 'form_score', 'matches_played']
        for col in expected_cols:
            self.assertIn(col, team_features.columns)

    def test_h2h_calculation(self):
        # 1 match between France and England: France won 2-1
        # H2H advantage for France vs England should be 1.0 (win)
        val = compute_h2h(self.matches, 'France', 'England')
        self.assertEqual(val, 1.0)
        
        # H2H advantage for England vs France should be -1.0 (loss)
        val2 = compute_h2h(self.matches, 'England', 'France')
        self.assertEqual(val2, -1.0)

    def test_h2h_series_matches_per_row_computation(self):
        """The vectorised H2H must agree exactly with the per-row version.

        compute_h2h_series() replaced an O(n^2) per-row loop. A divergence here
        would silently corrupt a training feature rather than raise, so the two
        implementations are compared directly.
        """
        import numpy as np

        rng = np.random.default_rng(7)
        teams = ['France', 'England', 'Germany', 'Spain', 'Italy']
        rows = []
        seen = set()
        for i in range(200):
            h, a = rng.choice(len(teams), size=2, replace=False)
            date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=int(i // 2))
            # A given pair meets at most once per date, as real fixtures do. Two
            # meetings on the cutoff date would land on the documented tie-break
            # where compute_h2h()'s unstable sort makes its own answer arbitrary.
            key = (date, tuple(sorted((teams[h], teams[a]))))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                'date': date,
                'home_team': teams[h], 'away_team': teams[a],
                'home_score': int(rng.integers(0, 4)), 'away_score': int(rng.integers(0, 4)),
            })
        df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

        fast = compute_h2h_series(df, max_matches=10)
        for pos in range(len(df)):
            slow = compute_h2h(df, df.home_team.iloc[pos], df.away_team.iloc[pos],
                               before_date=df.date.iloc[pos], max_matches=10)
            self.assertAlmostEqual(slow, fast.iloc[pos], places=12,
                                   msg=f"H2H mismatch at row {pos}")

    def test_h2h_series_excludes_same_day_meetings(self):
        """A match must not see the result of another match played the same day."""
        df = pd.DataFrame([
            {'date': pd.Timestamp('2024-01-01'), 'home_team': 'A', 'away_team': 'B',
             'home_score': 3, 'away_score': 0},
            {'date': pd.Timestamp('2024-01-01'), 'home_team': 'A', 'away_team': 'B',
             'home_score': 0, 'away_score': 1},
            {'date': pd.Timestamp('2024-02-01'), 'home_team': 'A', 'away_team': 'B',
             'home_score': 1, 'away_score': 1},
        ])
        s = compute_h2h_series(df, max_matches=10)
        self.assertEqual(s.iloc[0], 0.0)
        self.assertEqual(s.iloc[1], 0.0)
        # Third match sees one win and one loss -> mean of +1 and -1
        self.assertAlmostEqual(s.iloc[2], 0.0, places=12)

    def test_match_features(self):
        t_engine = TeamFeatureEngine(window=5)
        team_features = t_engine.compute_team_features(self.matches, self.elo_history)
        
        m_engine = MatchFeatureEngine()
        match_features = m_engine.compute_match_features(self.matches, team_features, self.elo_history)
        
        # Check expected columns
        feature_cols = m_engine.get_feature_columns()
        for col in feature_cols:
            self.assertIn(col, match_features.columns)
            
        # Target column result should exist
        self.assertIn('result', match_features.columns)
        self.assertIn('home_goals', match_features.columns)
        self.assertIn('away_goals', match_features.columns)

if __name__ == '__main__':
    unittest.main()
