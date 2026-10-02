import os
import glob
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from src.database.db_manager import DatabaseManager
from src.features.match_features import MatchFeatureEngine

# Project root, resolved from this file rather than the current working directory,
# so model lookups behave the same whether the app is launched from the repo root,
# from src/app/, or by Streamlit Cloud.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A draw is a common outcome (~24% of matches) but rarely the single most likely
# one: draw probability tops out near 0.42 and usually sits second behind a home
# or away win. Choosing the verdict by argmax therefore almost never returns
# "Draw" - 32 times in 4652 test matches - even though the underlying probability
# is reasonably calibrated.
#
# The verdict is a draw when its probability clears DRAW_VERDICT_THRESHOLD *and*
# the leading side is no further than DRAW_VERDICT_MARGIN ahead of it. The margin
# matters: on the threshold alone the rule fires on fixtures where one side is
# 64% likely, which is not a draw by any reading. Requiring the match to be close
# keeps almost all of the benefit at a fraction of the cost - on the 2022+ test
# set, macro F1 0.4442 -> 0.5014 and draw F1 0.0227 -> 0.2170 for 1.0pp of
# accuracy, where the unguarded rule gave up 4.2pp for macro F1 0.5072.
#
# Tuned on 2019-2021 as a validation set, never on the test period. This changes
# only the label; the probabilities themselves are untouched.
DRAW_VERDICT_THRESHOLD = 0.28
DRAW_VERDICT_MARGIN = 0.12

# Where the training pipeline writes artifacts and the deployed app reads them.
MODEL_DIR = os.path.join(PROJECT_ROOT, 'src', 'app', 'models')


class MatchPredictor:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager
        self.feature_engine = MatchFeatureEngine()
        load = self._load_active_model_from_db if db_manager else self._load_latest_local_model
        self.classifier = load('outcome_classifier')
        self.home_goals_model = load('home_goals_regressor')
        self.away_goals_model = load('away_goals_regressor')

    def _load_active_model_from_db(self, model_name: str) -> Any:
        """Query database for the active version of a specific model name."""
        from src.database.models import ModelVersion
        with self.db_manager.get_session() as session:
            model_record = session.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.is_active == True
            ).first()
            if not model_record:
                raise FileNotFoundError(f"No active database model found with name '{model_name}'")
            stored_path = model_record.model_path

        # model_path is recorded at training time and may be relative to whatever
        # directory the pipeline ran in (or an absolute path from another machine),
        # so it is not guaranteed to resolve here. Try it, then fall back to
        # scanning MODEL_DIR.
        for candidate in (stored_path, os.path.join(PROJECT_ROOT, stored_path)):
            if candidate and os.path.isfile(candidate):
                return joblib.load(candidate)

        return self._load_latest_local_model(model_name)

    def _load_latest_local_model(self, model_name: str) -> Any:
        """Load the highest-versioned .joblib for model_name from MODEL_DIR."""
        files = sorted(glob.glob(os.path.join(MODEL_DIR, f'{model_name}*.joblib')))
        if not files:
            raise FileNotFoundError(
                f"No model file starting with '{model_name}' found in {MODEL_DIR}. "
                "Run `python run_pipeline.py` to train and save models."
            )
        return joblib.load(files[-1])

    def _probabilities_by_class(self, probs) -> Dict[int, float]:
        """
        Map predict_proba output to class labels using the model's classes_.

        Falls back to positional order when classes_ is missing or does not line
        up with the probability vector - which is also what keeps mock models in
        the tests working, since a MagicMock returns a stand-in for any attribute
        rather than raising AttributeError.
        """
        classes = getattr(self.classifier, 'classes_', None)
        labels = []
        if classes is not None:
            try:
                labels = [int(c) for c in np.asarray(classes).ravel().tolist()]
            except (TypeError, ValueError):
                labels = []

        if len(labels) != len(probs):
            labels = list(range(len(probs)))

        return {label: float(p) for label, p in zip(labels, probs)}

    def predict(self, feature_vector: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Run inference using the feature vector.
        Returns outcome probabilities, expected goals, and confidence scores.
        """
        if isinstance(feature_vector, dict):
            df = pd.DataFrame([feature_vector])
        else:
            df = feature_vector.copy()
            
        # Ensure correct column order
        cols = self.feature_engine.get_feature_columns()
        X = df[cols]
        
        # Predict W/D/L probabilities.
        # Outcome encoding: 0=Away win, 1=Draw, 2=Home win. Columns are matched to
        # labels via classes_ rather than assumed to be in that order: a model
        # trained on a slice missing an outcome returns fewer columns, and reading
        # by position would then report another class's probability as the Draw
        # and raise IndexError on the third column.
        probs = self.classifier.predict_proba(X)[0]
        prob_by_class = self._probabilities_by_class(probs)
        p_away = prob_by_class.get(0, 0.0)
        p_draw = prob_by_class.get(1, 0.0)
        p_home = prob_by_class.get(2, 0.0)
        
        # Predict Expected Goals
        exg_home = max(0.0, float(self.home_goals_model.predict(X)[0]))
        exg_away = max(0.0, float(self.away_goals_model.predict(X)[0]))
        
        # Calculate confidence metric: distance from highest probability to second highest
        sorted_probs = sorted([p_home, p_draw, p_away], reverse=True)
        confidence = float(sorted_probs[0] - sorted_probs[1])
        
        # Determine the verdict. See the notes on DRAW_VERDICT_THRESHOLD for why
        # this is not a plain argmax.
        leader_prob = max(p_home, p_away)
        is_draw = (
            p_draw > DRAW_VERDICT_THRESHOLD
            and (leader_prob - p_draw) <= DRAW_VERDICT_MARGIN
        )
        if is_draw:
            outcome_label = 'Draw'
        elif p_home > p_away:
            outcome_label = 'Home Win'
        else:
            outcome_label = 'Away Win'

        # The most likely single outcome, which can differ from the verdict above.
        # Listed away-draw-home so ties resolve to the lowest class index.
        most_likely = max(
            (('Away Win', p_away), ('Draw', p_draw), ('Home Win', p_home)),
            key=lambda item: item[1]
        )[0]
            
        return {
            'home_win_prob': p_home,
            'draw_prob': p_draw,
            'away_win_prob': p_away,
            'expected_home_goals': exg_home,
            'expected_away_goals': exg_away,
            'confidence': confidence,
            'outcome': outcome_label,
            'most_likely_outcome': most_likely
        }

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        tournament: str,
        team_features_df: pd.DataFrame,
        matches_df: pd.DataFrame,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        High-level wrapper to generate features, predict outcome, and save record to DB.
        """
        # 1. Generate feature vector
        feats = self.feature_engine.create_prediction_features(
            home_team, away_team, tournament, team_features_df, matches_df
        )
        
        # 2. Run prediction
        pred = self.predict(feats)
        pred['home_team'] = home_team
        pred['away_team'] = away_team
        
        # 3. Optionally save to DB
        if save_to_db and self.db_manager:
            active_model = self.db_manager.get_active_model()
            pred['model_id'] = active_model.model_id if active_model else None
            
            # Save predictions and return the generated ID
            pred_id = self.db_manager.save_prediction(pred)
            pred['prediction_id'] = pred_id
            
        return pred
