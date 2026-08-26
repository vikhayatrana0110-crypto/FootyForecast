import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.models.predictor import MatchPredictor
from src.models.explainer import MatchExplainer

class TestPredictorAndExplainer(unittest.TestCase):
    def setUp(self):
        # Create mock models
        self.mock_classifier = MagicMock()
        # predict_proba returns [Away Prob, Draw Prob, Home Prob]
        # Let's say Brazil (home) vs Argentina (away): 50% Win, 30% Draw, 20% Loss
        self.mock_classifier.predict_proba = MagicMock(return_value=np.array([[0.20, 0.30, 0.50]]))
        
        self.mock_home_goals = MagicMock()
        self.mock_home_goals.predict = MagicMock(return_value=np.array([2.1]))
        
        self.mock_away_goals = MagicMock()
        self.mock_away_goals.predict = MagicMock(return_value=np.array([1.2]))
        
        # Instantiate MatchPredictor using mock models
        self.predictor = MatchPredictor.__new__(MatchPredictor)
        self.predictor.classifier = self.mock_classifier
        self.predictor.home_goals_model = self.mock_home_goals
        self.predictor.away_goals_model = self.mock_away_goals
        self.predictor.db_manager = None
        self.predictor.feature_engine = MagicMock()
        self.predictor.feature_engine.get_feature_columns = MagicMock(return_value=['f1', 'f2'])

    def _predictor_with(self, classes, probs):
        """Build a MatchPredictor around a mock classifier with a given class set."""
        predictor = MatchPredictor.__new__(MatchPredictor)
        predictor.db_manager = None
        predictor.home_goals_model = self.mock_home_goals
        predictor.away_goals_model = self.mock_away_goals
        predictor.feature_engine = MagicMock()
        predictor.feature_engine.get_feature_columns = MagicMock(return_value=['f1', 'f2'])

        clf = MagicMock()
        clf.classes_ = classes
        clf.predict_proba = MagicMock(return_value=probs)
        predictor.classifier = clf
        return predictor

    def test_prediction_output(self):
        feature_vector = {'f1': 1.0, 'f2': 0.5}
        pred = self.predictor.predict(feature_vector)
        
        self.assertAlmostEqual(pred['home_win_prob'], 0.50)
        self.assertAlmostEqual(pred['draw_prob'], 0.30)
        self.assertAlmostEqual(pred['away_win_prob'], 0.20)
        
        # Probs must sum to 1.0
        self.assertAlmostEqual(pred['home_win_prob'] + pred['draw_prob'] + pred['away_win_prob'], 1.0)
        
        # Expected Goals
        self.assertAlmostEqual(pred['expected_home_goals'], 2.1)
        self.assertAlmostEqual(pred['expected_away_goals'], 1.2)
        
        # Outcome label
        self.assertEqual(pred['outcome'], 'Home Win')

    def test_explainer_fallback(self):
        # Instantiate explainer with mock model
        explainer = MatchExplainer(self.mock_classifier)
        
        # Mock dataframe
        X = pd.DataFrame([{'elo_difference': 100.0, 'form_difference': 0.5, 'other_feature': 1.0}])
        feature_names = ['elo_difference', 'form_difference', 'other_feature']
        
        # predicted class 2 = Home Win
        explanation = explainer.explain_prediction(X, feature_names, predicted_class=2)
        
        # Check output structure
        self.assertIn('positive_factors', explanation)
        self.assertIn('negative_factors', explanation)
        self.assertIn('explanation_text', explanation)
        self.assertTrue(len(explanation['positive_factors']) > 0)

    def test_explainer_uses_real_shap_with_a_real_model(self):
        """SHAP values must actually be produced, not silently fall back.

        explain_prediction() wraps its SHAP path in a broad `except`, so a bug in
        that path degrades to the heuristic without surfacing an error. A
        `list.sort(ascending=...)` TypeError hid there and meant real SHAP values
        were never returned. An empty `shap_values` dict is the fallback's
        signature, so assert it is populated.
        """
        try:
            import shap  # noqa: F401
            import xgboost as xgb
        except ImportError:
            self.skipTest("shap/xgboost not installed")

        import numpy as np
        from src.features.match_features import MatchFeatureEngine

        cols = MatchFeatureEngine().get_feature_columns()
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, len(cols))), columns=cols)
        y = rng.integers(0, 3, size=120)
        model = xgb.XGBClassifier(n_estimators=8, max_depth=2, verbosity=0).fit(X, y)

        explainer = MatchExplainer(model)
        if explainer.explainer is None:
            self.skipTest("TreeExplainer unavailable for this model type")

        result = explainer.explain_prediction(X.head(1), cols, 2)
        self.assertTrue(result['shap_values'],
                        "explain_prediction fell back to the heuristic instead of using SHAP")

        # Negative factors must run most-negative first.
        negs = [f['shap_value'] for f in result['negative_factors']]
        self.assertEqual(negs, sorted(negs), "negative factors are not sorted ascending")

        # One record per feature, carrying all three class contributions.
        records = explainer.explanation_records(X.head(1), cols)
        self.assertEqual(len(records), len(cols))
        for r in records:
            self.assertIn('shap_value_home', r)
            self.assertIn('shap_value_draw', r)
            self.assertIn('shap_value_away', r)

    def test_explanation_records_empty_without_shap(self):
        """Heuristic constants must never be persisted as if they were SHAP values."""
        explainer = MatchExplainer(self.mock_classifier)
        self.assertEqual(explainer.explanation_records(pd.DataFrame(), []), [])

    def test_probabilities_mapped_by_class_not_position(self):
        """A model missing an outcome class must not mislabel the remaining ones.

        predict() previously read probs[0], probs[1], probs[2] as away/draw/home.
        A classifier trained on data containing no Draws exposes classes_ == [0, 2]
        and returns two columns, so column 1 (Home Win) was reported as the Draw
        probability and probs[2] raised IndexError.
        """
        predictor = self._predictor_with(
            classes=np.array([0, 2]),                        # no Draw class
            probs=np.array([[0.30, 0.70]]))
        result = predictor.predict({'f1': 1.0, 'f2': 2.0})

        self.assertAlmostEqual(result['away_win_prob'], 0.30)
        self.assertAlmostEqual(result['home_win_prob'], 0.70)
        self.assertAlmostEqual(result['draw_prob'], 0.0,
                               msg="absent class must be 0.0, not another class's probability")
        self.assertEqual(result['outcome'], 'Home Win')

    def test_full_class_set_still_maps_correctly(self):
        """The ordinary three-class case must be unchanged by the classes_ mapping."""
        predictor = self._predictor_with(
            classes=np.array([0, 1, 2]),
            probs=np.array([[0.20, 0.30, 0.50]]))
        result = predictor.predict({'f1': 1.0, 'f2': 2.0})
        self.assertAlmostEqual(result['away_win_prob'], 0.20)
        self.assertAlmostEqual(result['draw_prob'], 0.30)
        self.assertAlmostEqual(result['home_win_prob'], 0.50)
        self.assertEqual(result['outcome'], 'Home Win')

    def test_draw_called_when_probability_clears_threshold(self):
        """A draw must be the verdict once its probability clears the threshold.

        Argmax could effectively never return 'Draw': draw probability tops out
        around 0.42 and is usually second, so the verdict was Home or Away in
        4611 of 4652 test matches while draws are 24% of results.
        """
        # Draw clears the threshold and the match is close: away 0.31, draw 0.31,
        # home 0.38 - the leader is 0.07 ahead, inside the margin.
        predictor = self._predictor_with(
            classes=np.array([0, 1, 2]),
            probs=np.array([[0.31, 0.31, 0.38]]))
        result = predictor.predict({'f1': 1.0, 'f2': 2.0})

        self.assertEqual(result['outcome'], 'Draw')
        self.assertEqual(result['most_likely_outcome'], 'Home Win',
                         "most_likely_outcome should still report the top probability")

    def test_draw_not_called_below_threshold(self):
        """Below the threshold the verdict falls back to the stronger of home/away."""
        from src.models.predictor import DRAW_VERDICT_THRESHOLD

        p = DRAW_VERDICT_THRESHOLD - 0.05
        predictor = self._predictor_with(
            classes=np.array([0, 1, 2]),
            probs=np.array([[0.45, p, 1.0 - 0.45 - p]]))
        result = predictor.predict({'f1': 1.0, 'f2': 2.0})

        self.assertEqual(result['outcome'], 'Away Win')
        self.assertEqual(result['most_likely_outcome'], 'Away Win')

    def test_probabilities_unaffected_by_verdict_rule(self):
        """The threshold changes only the label - probabilities must be untouched."""
        predictor = self._predictor_with(
            classes=np.array([0, 1, 2]),
            probs=np.array([[0.30, 0.35, 0.35]]))
        result = predictor.predict({'f1': 1.0, 'f2': 2.0})
        self.assertAlmostEqual(result['away_win_prob'], 0.30)
        self.assertAlmostEqual(result['draw_prob'], 0.35)
        self.assertAlmostEqual(result['home_win_prob'], 0.35)

    def test_draw_not_called_when_one_side_is_clearly_favoured(self):
        """A high draw probability alone must not override a dominant favourite.

        On the threshold alone the rule fired on fixtures where the leading side
        was 64% likely, which is not a draw by any reading. The margin requires
        the match to actually be close before the verdict becomes a draw.
        """
        from src.models.predictor import DRAW_VERDICT_THRESHOLD, DRAW_VERDICT_MARGIN

        draw = DRAW_VERDICT_THRESHOLD + 0.02
        home = draw + DRAW_VERDICT_MARGIN + 0.05      # comfortably clear of the draw
        away = 1.0 - draw - home
        predictor = self._predictor_with(
            classes=np.array([0, 1, 2]),
            probs=np.array([[away, draw, home]]))
        result = predictor.predict({'f1': 1.0, 'f2': 2.0})

        self.assertEqual(result['outcome'], 'Home Win',
                         "a clear favourite must not be reported as a draw")
        self.assertGreater(result['draw_prob'], DRAW_VERDICT_THRESHOLD,
                           "draw probability is still above the threshold - only the margin blocks it")

if __name__ == '__main__':
    unittest.main()
