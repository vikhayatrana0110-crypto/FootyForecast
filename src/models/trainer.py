import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, f1_score, mean_absolute_error, mean_squared_error

from src.database.db_manager import DatabaseManager
from src.features.match_features import MatchFeatureEngine

# Anchored to this file, not the working directory, so a pipeline run saves to the
# same place regardless of where it was launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trained artifacts are written where the deployed app reads them. This directory
# is tracked in git precisely because the Streamlit deployment cannot retrain:
# writing anywhere else lets the repository and the database drift apart, which is
# how the committed models previously went stale without anyone noticing.
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'src', 'app', 'models')


class ModelTrainer:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.feature_engine = MatchFeatureEngine()

    def prepare_data(self, match_features_df: pd.DataFrame, cutoff_date: str = '2022-01-01') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prepare train and test splits using a time-based cutoff.
        Returns:
            X_train, X_test, y_class_train, y_class_test, y_home_goals_train, y_home_goals_test, y_away_goals_train, y_away_goals_test
        """
        df = match_features_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Drop rows with missing feature values
        feature_cols = self.feature_engine.get_feature_columns()
        df = df.dropna(subset=feature_cols + ['result', 'home_goals', 'away_goals'])
        
        # Split train/test
        cutoff = pd.to_datetime(cutoff_date)
        train_mask = df['date'] < cutoff
        test_mask = df['date'] >= cutoff
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        
        y_class_train = train_df['result']
        y_class_test = test_df['result']
        
        y_home_goals_train = train_df['home_goals']
        y_home_goals_test = test_df['home_goals']
        
        y_away_goals_train = train_df['away_goals']
        y_away_goals_test = test_df['away_goals']
        
        return X_train, X_test, y_class_train, y_class_test, y_home_goals_train, y_home_goals_test, y_away_goals_train, y_away_goals_test

    def train_classifier(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> xgb.XGBClassifier:
        """Train an XGBoost Classifier for match outcomes (0=Away, 1=Draw, 2=Home)."""
        classifier = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric='mlogloss',
            random_state=42
        )
        
        if X_val is not None and y_val is not None:
            classifier.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            classifier.fit(X_train, y_train, verbose=False)
            
        return classifier

    def train_goals_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """Train an XGBoost Regressor for goals scored."""
        regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        regressor.fit(X_train, y_train, verbose=False)
        return regressor

    def evaluate_classifier(self, model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate outcome classifier model."""
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs)
        f1 = f1_score(y_test, preds, average='macro')
        
        return {
            'accuracy': float(acc),
            'log_loss': float(loss),
            'f1_macro': float(f1)
        }

    def evaluate_regressor(self, model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate goals regressor model."""
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        return {
            'mae': float(mae),
            'rmse': float(rmse)
        }

    def save_model(self, model: Any, name: str, version: str) -> str:
        """
        Save a model artifact using joblib.

        Returns a path relative to the project root when possible. An absolute
        path from the training machine would not resolve anywhere else, and a
        path relative to the current working directory would only resolve if the
        app happened to be launched from the same place - both of which have
        silently broken model loading before.
        """
        filename = f"{name}_v{version}.joblib"
        path = os.path.join(self.model_dir, filename)
        joblib.dump(model, path)
        print(f"Saved model to {path}")

        try:
            return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, '/')
        except ValueError:
            # Different drive on Windows; nothing portable to return.
            return path

    def run_training_pipeline(self, match_features_df: pd.DataFrame, db_manager: Optional[DatabaseManager] = None, cutoff_date: str = '2022-01-01', version: str = '1.0') -> Dict[str, Any]:
        """Run the full training pipeline, evaluate models, save artifacts, and log metrics."""
        print("Preparing dataset splits...")
        X_train, X_test, y_c_train, y_c_test, y_hg_train, y_hg_test, y_ag_train, y_ag_test = self.prepare_data(
            match_features_df, cutoff_date=cutoff_date
        )
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        print("Training match outcome classifier...")
        classifier = self.train_classifier(X_train, y_c_train, X_test, y_c_test)
        c_metrics = self.evaluate_classifier(classifier, X_test, y_c_test)
        print(f"Classifier Metrics: Accuracy={c_metrics['accuracy']:.4f}, Log Loss={c_metrics['log_loss']:.4f}, F1={c_metrics['f1_macro']:.4f}")
        
        print("Training Home expected goals regressor...")
        home_goals_model = self.train_goals_model(X_train, y_hg_train)
        hg_metrics = self.evaluate_regressor(home_goals_model, X_test, y_hg_test)
        print(f"Home Goals Regressor Metrics: MAE={hg_metrics['mae']:.4f}, RMSE={hg_metrics['rmse']:.4f}")
        
        print("Training Away expected goals regressor...")
        away_goals_model = self.train_goals_model(X_train, y_ag_train)
        ag_metrics = self.evaluate_regressor(away_goals_model, X_test, y_ag_test)
        print(f"Away Goals Regressor Metrics: MAE={ag_metrics['mae']:.4f}, RMSE={ag_metrics['rmse']:.4f}")
        
        # Save artifacts
        c_path = self.save_model(classifier, 'outcome_classifier', version)
        hg_path = self.save_model(home_goals_model, 'home_goals_regressor', version)
        ag_path = self.save_model(away_goals_model, 'away_goals_regressor', version)
        
        # Log to DB if provided
        if db_manager:
            print("Logging models to database...")
            with db_manager:
                # Log outcome classifier as main model
                db_manager.save_model_version(
                    name='outcome_classifier',
                    version=version,
                    metrics={
                        'accuracy': c_metrics['accuracy'],
                        'log_loss': c_metrics['log_loss'],
                        'f1_macro': c_metrics['f1_macro']
                    },
                    model_path=c_path
                )
                
                # Log expected goals regressors
                db_manager.save_model_version(
                    name='home_goals_regressor',
                    version=version,
                    metrics={
                        'accuracy': hg_metrics['mae'], # store MAE in accuracy column
                        'log_loss': hg_metrics['rmse'], # store RMSE in log_loss column
                        'f1_macro': 0.0
                    },
                    model_path=hg_path
                )
                
                db_manager.save_model_version(
                    name='away_goals_regressor',
                    version=version,
                    metrics={
                        'accuracy': ag_metrics['mae'],
                        'log_loss': ag_metrics['rmse'],
                        'f1_macro': 0.0
                    },
                    model_path=ag_path
                )
                
        return {
            'classifier': classifier,
            'home_goals_model': home_goals_model,
            'away_goals_model': away_goals_model,
            'metrics': {
                'classifier': c_metrics,
                'home_goals': hg_metrics,
                'away_goals': ag_metrics
            },
            'paths': {
                'classifier': c_path,
                'home_goals': hg_path,
                'away_goals': ag_path
            }
        }
