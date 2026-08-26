import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from src.database.db_manager import DatabaseManager
from src.features.match_features import MatchFeatureEngine

# Project root, resolved from this file rather than the current working directory,
# so model lookups behave the same whether the app is launched from the repo root,
# from src/app/, or by Streamlit Cloud.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directories searched for .joblib artifacts, in priority order.
# src/app/models is the tracked copy that the training pipeline writes to and the
# deployed app reads from, so it is checked first. models/ is only a fallback for
# older local checkouts that still have artifacts from a previous layout.
MODEL_SEARCH_DIRS = (
    os.path.join(PROJECT_ROOT, 'src', 'app', 'models'),
    os.path.join(PROJECT_ROOT, 'models'),
)


class MatchPredictor:
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        home_goals_path: Optional[str] = None,
        away_goals_path: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        self.db_manager = db_manager
        self.feature_engine = MatchFeatureEngine()
        
        # Load classifier
        if classifier_path:
            self.classifier = joblib.load(classifier_path)
        elif db_manager:
            self.classifier = self._load_active_model_from_db('outcome_classifier')
        else:
            self.classifier = self._load_latest_local_model('outcome_classifier')
            
        # Load home goals regressor
        if home_goals_path:
            self.home_goals_model = joblib.load(home_goals_path)
        elif db_manager:
            self.home_goals_model = self._load_active_model_from_db('home_goals_regressor')
        else:
            self.home_goals_model = self._load_latest_local_model('home_goals_regressor')
            
        # Load away goals regressor
        if away_goals_path:
            self.away_goals_model = joblib.load(away_goals_path)
        elif db_manager:
            self.away_goals_model = self._load_active_model_from_db('away_goals_regressor')
        else:
            self.away_goals_model = self._load_latest_local_model('away_goals_regressor')

    def _load_active_model_from_db(self, model_name: str) -> Any:
        """Query database for the active version of a specific model name."""
        if not self.db_manager:
            raise ValueError("db_manager must be set to load models from database.")
        with self.db_manager:
            session = self.db_manager.get_session()
            from src.database.models import ModelVersion
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
        # scanning the known model directories.
        for candidate in (stored_path, os.path.join(PROJECT_ROOT, stored_path)):
            if candidate and os.path.isfile(candidate):
                return joblib.load(candidate)

        return self._load_latest_local_model(model_name)

    def _find_model_dirs(self) -> List[str]:
        """Return the model search directories that actually exist."""
        return [d for d in MODEL_SEARCH_DIRS if os.path.isdir(d)]

    def _load_latest_local_model(self, model_name: str) -> Any:
        """Load the highest-versioned .joblib for model_name from the search directories."""
        searched = self._find_model_dirs()
        if not searched:
            raise FileNotFoundError(
                "No model directory found. Looked in: "
                + ", ".join(MODEL_SEARCH_DIRS)
                + ". Run `python run_pipeline.py` to train and save models."
            )

        matches = []
        for model_dir in searched:
            matches += [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.startswith(model_name) and f.endswith('.joblib')
            ]

        if not matches:
            raise FileNotFoundError(
                f"No model file starting with '{model_name}' found in: "
                + ", ".join(searched)
                + ". Run `python run_pipeline.py` to train and save models."
            )

        # Sort by filename so the highest version suffix wins.
        matches.sort(key=os.path.basename)
        return joblib.load(matches[-1])

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
        
        # Determine prediction label from the named probabilities. Listed
        # away-draw-home so that ties resolve to the lowest class index, matching
        # the argmax this replaced.
        outcome_label = max(
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
            'outcome': outcome_label
        }

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        tournament: str,
        team_features_df: pd.DataFrame,
        elo_history_df: pd.DataFrame,
        matches_df: pd.DataFrame,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        High-level wrapper to generate features, predict outcome, and save record to DB.
        """
        # 1. Generate feature vector
        feats = self.feature_engine.create_prediction_features(
            home_team, away_team, tournament, team_features_df, elo_history_df, matches_df
        )
        
        # 2. Run prediction
        pred = self.predict(feats)
        pred['home_team'] = home_team
        pred['away_team'] = away_team
        
        # 3. Optionally save to DB
        if save_to_db and self.db_manager:
            # Get active model_id if available
            model_id = None
            with self.db_manager:
                active_model = self.db_manager.get_active_model()
                if active_model:
                    model_id = active_model.model_id
                    
            pred['model_id'] = model_id
            
            # Save predictions and return the generated ID
            pred_id = self.db_manager.save_prediction(pred)
            pred['prediction_id'] = pred_id
            
        return pred
