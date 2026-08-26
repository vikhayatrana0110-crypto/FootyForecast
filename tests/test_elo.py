import unittest
import pandas as pd
from src.features.elo_calculator import EloCalculator

class TestEloCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = EloCalculator(k_factor=30, home_advantage=100.0, initial_rating=1500.0)

    def test_initial_rating(self):
        self.assertEqual(self.calc.get_rating("Team A"), 1500.0)

    def test_expected_score(self):
        # 1500 vs 1500 expectation should be 0.5
        self.assertAlmostEqual(self.calc.expected_score(1500.0, 1500.0), 0.5)
        # Higher rating team should have expectation > 0.5
        self.assertGreater(self.calc.expected_score(1600.0, 1500.0), 0.5)

    def test_k_factors(self):
        self.assertEqual(self.calc.get_k_factor("FIFA World Cup"), 60.0)
        self.assertEqual(self.calc.get_k_factor("UEFA Euro"), 50.0)
        self.assertEqual(self.calc.get_k_factor("Friendly"), 20.0)

    def test_update_ratings(self):
        # Neutral venue: 1500 vs 1500, friendly (K=20), home team wins 1-0
        # Expected: exp_home = 0.5, act_home = 1.0, G=1.0
        # new_rating = 1500 + 20 * 1.0 * (1.0 - 0.5) = 1510
        new_home, new_away = self.calc.update_ratings("Home", "Away", 1, 0, "Friendly", neutral=True)
        self.assertEqual(new_home, 1510.0)
        self.assertEqual(new_away, 1490.0)

    def test_compute_all_ratings(self):
        matches = pd.DataFrame([
            {'date': '2026-06-01', 'home_team': 'Brazil', 'away_team': 'Argentina', 'home_score': 2, 'away_score': 0, 'tournament': 'Friendly', 'neutral': True},
            {'date': '2026-06-02', 'home_team': 'Brazil', 'away_team': 'Germany', 'home_score': 1, 'away_score': 1, 'tournament': 'Friendly', 'neutral': True}
        ])
        elo_history = self.calc.compute_all_ratings(matches)
        
        # Should contain 4 records (2 matches * 2 teams)
        self.assertEqual(len(elo_history), 4)
        self.assertTrue('elo_rating' in elo_history.columns)
        self.assertTrue('team' in elo_history.columns)

if __name__ == '__main__':
    unittest.main()
