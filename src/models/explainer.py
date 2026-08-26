import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class MatchExplainer:
    def __init__(self, classifier_model: Any):
        self.model = classifier_model
        self.explainer = None
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"Error initializing SHAP TreeExplainer: {e}")
                self.explainer = None

    def explain_prediction(self, features_df: pd.DataFrame, feature_names: List[str], predicted_class: int) -> Dict[str, Any]:
        """
        Generate SHAP values for a prediction.
        Returns:
            A dictionary containing SHAP values, feature importance, and lists of positive and negative factors.
        """
        # outcome labels: 0=Away Win, 1=Draw, 2=Home Win
        class_labels = ['Away Win', 'Draw', 'Home Win']
        
        # Ensure features are in matching order
        X = features_df[feature_names]
        
        if not SHAP_AVAILABLE or self.explainer is None:
            # Fallback to feature importance heuristic if SHAP is unavailable
            return self._heuristic_explanation(X, predicted_class)
            
        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # multi-class SHAP: shap_values is a list of arrays [Away, Draw, Home]
            # or a 3D array of shape (samples, features, classes).
            if isinstance(shap_values, list):
                shap_class = shap_values[predicted_class][0]
            else:
                # 3D array: slice along classes dimension
                if len(shap_values.shape) == 3:
                    shap_class = shap_values[0, :, predicted_class]
                else:
                    shap_class = shap_values[0] # Fallback
            
            # Map features to SHAP values
            feature_contributions = []
            for name, val, shap_val in zip(feature_names, X.iloc[0].values, shap_class):
                feature_contributions.append({
                    'feature_name': name,
                    'feature_value': float(val),
                    'shap_value': float(shap_val)
                })
                
            # Sort by SHAP value magnitude
            feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            # Filter positive/negative factors
            positives = [fc for fc in feature_contributions if fc['shap_value'] > 0]
            negatives = [fc for fc in feature_contributions if fc['shap_value'] < 0]
            
            # Sort positive features descending (most positive first)
            positives.sort(key=lambda x: x['shap_value'], reverse=True)
            # Sort negative features ascending (most negative first).
            # NOTE: list.sort() takes `reverse`, not pandas' `ascending`. Passing
            # `ascending` raised TypeError on every call, which the except block
            # below swallowed - so this method always fell back to the heuristic
            # and real SHAP values were never surfaced.
            negatives.sort(key=lambda x: x['shap_value'])
            
            # Generate natural language summary
            summary = self._generate_summary_text(positives, negatives, class_labels[predicted_class])
            
            return {
                'shap_values': {class_labels[predicted_class]: [fc['shap_value'] for fc in feature_contributions]},
                'feature_importance': [(fc['feature_name'], fc['shap_value']) for fc in feature_contributions],
                'positive_factors': positives[:3],
                'negative_factors': negatives[:3],
                'explanation_text': summary
            }
        except Exception as e:
            print(f"Error running SHAP explanation: {e}")
            return self._heuristic_explanation(X, predicted_class)

    def explanation_records(self, features_df: pd.DataFrame, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Return one record per feature carrying its SHAP value for all three classes,
        shaped for DatabaseManager.save_prediction_explanations().

        Returns an empty list when SHAP is unavailable or errors: the heuristic
        fallback produces invented constants, which are fine for on-screen wording
        but must never be persisted as if they were real SHAP values.
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return []

        X = features_df[feature_names]
        try:
            shap_values = self.explainer.shap_values(X)

            # Normalise the two shapes SHAP returns for multi-class trees into
            # per_class[class_index] -> one value per feature.
            if isinstance(shap_values, list):
                per_class = [np.asarray(sv)[0] for sv in shap_values]
            elif getattr(shap_values, 'ndim', 0) == 3:
                per_class = [shap_values[0, :, i] for i in range(shap_values.shape[2])]
            else:
                return []

            if len(per_class) < 3:
                return []

            # Class order matches the model: 0=Away Win, 1=Draw, 2=Home Win
            away, draw, home = per_class[0], per_class[1], per_class[2]

            return [
                {
                    'feature_name': name,
                    'feature_value': float(value),
                    'shap_value_home': float(home[i]),
                    'shap_value_draw': float(draw[i]),
                    'shap_value_away': float(away[i]),
                }
                for i, (name, value) in enumerate(zip(feature_names, X.iloc[0].values))
            ]
        except Exception as e:
            print(f"Error building SHAP explanation records: {e}")
            return []

    def get_shap_plot_data(self, features_df: pd.DataFrame, feature_names: List[str], class_index: int) -> Dict[str, Any]:
        """Get pre-sorted data for Plotly bar chart rendering."""
        X = features_df[feature_names]
        if not SHAP_AVAILABLE or self.explainer is None:
            # Fallback heuristic
            feat_imp = self.model.feature_importances_
            sorted_idx = np.argsort(feat_imp)
            return {
                'features': [feature_names[i] for i in sorted_idx],
                'shap_values': [float(feat_imp[i]) for i in sorted_idx]
            }
            
        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_class = shap_values[class_index][0]
            else:
                if len(shap_values.shape) == 3:
                    shap_class = shap_values[0, :, class_index]
                else:
                    shap_class = shap_values[0]
            
            # Sort by absolute SHAP value
            sorted_idx = np.argsort(np.abs(shap_class))
            return {
                'features': [feature_names[i] for i in sorted_idx],
                'shap_values': [float(shap_class[i]) for i in sorted_idx]
            }
        except Exception as e:
            print(f"Error getting SHAP plot data: {e}")
            # Fallback
            feat_imp = self.model.feature_importances_
            sorted_idx = np.argsort(feat_imp)
            return {
                'features': [feature_names[i] for i in sorted_idx],
                'shap_values': [float(feat_imp[i]) for i in sorted_idx]
            }

    def _heuristic_explanation(self, X: pd.DataFrame, predicted_class: int) -> Dict[str, Any]:
        """Fallback explanation heuristic when SHAP is unavailable."""
        feature_names = list(X.columns)
        vals = X.iloc[0].values
        
        # Build explanation based on simple feature checks
        positives = []
        negatives = []
        
        # Check ELO rating difference
        elo_diff = X['elo_difference'].values[0]
        if predicted_class == 2: # Home Win predicted
            if elo_diff > 0:
                positives.append({'feature_name': 'elo_difference', 'feature_value': float(elo_diff), 'shap_value': 0.25})
            else:
                negatives.append({'feature_name': 'elo_difference', 'feature_value': float(elo_diff), 'shap_value': -0.25})
        elif predicted_class == 0: # Away Win predicted
            if elo_diff < 0:
                positives.append({'feature_name': 'elo_difference', 'feature_value': float(elo_diff), 'shap_value': 0.25})
            else:
                negatives.append({'feature_name': 'elo_difference', 'feature_value': float(elo_diff), 'shap_value': -0.25})
                
        # Check Form difference
        form_diff = X['form_difference'].values[0]
        if predicted_class == 2:
            if form_diff > 0:
                positives.append({'feature_name': 'form_difference', 'feature_value': float(form_diff), 'shap_value': 0.15})
            else:
                negatives.append({'feature_name': 'form_difference', 'feature_value': float(form_diff), 'shap_value': -0.15})
        elif predicted_class == 0:
            if form_diff < 0:
                positives.append({'feature_name': 'form_difference', 'feature_value': float(form_diff), 'shap_value': 0.15})
            else:
                negatives.append({'feature_name': 'form_difference', 'feature_value': float(form_diff), 'shap_value': -0.15})

        # Add generic features to fill lists
        for name in feature_names:
            if name not in ['elo_difference', 'form_difference']:
                val = float(X[name].values[0])
                if val > 0:
                    positives.append({'feature_name': name, 'feature_value': val, 'shap_value': 0.05})
                else:
                    negatives.append({'feature_name': name, 'feature_value': val, 'shap_value': -0.05})

        positives.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        negatives.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        class_labels = ['Away Win', 'Draw', 'Home Win']
        summary = self._generate_summary_text(positives, negatives, class_labels[predicted_class])
        
        return {
            'shap_values': {},
            'feature_importance': [(p['feature_name'], p['shap_value']) for p in positives + negatives],
            'positive_factors': positives[:3],
            'negative_factors': negatives[:3],
            'explanation_text': summary
        }

    def _generate_summary_text(self, positives: list, negatives: list, predicted_label: str) -> str:
        """Create a human readable summary paragraph explaining the match prediction."""
        if not positives:
            return f"The model predicts a {predicted_label} due to baseline statistical tendencies."
            
        top_pos = positives[0]['feature_name'].replace('_', ' ')
        top_pos_val = positives[0]['feature_value']
        
        text = f"The model's prediction of **{predicted_label}** is primarily driven by the **{top_pos}** (value: {top_pos_val:.2f}), which increases prediction confidence. "
        
        if len(positives) > 1:
            second_pos = positives[1]['feature_name'].replace('_', ' ')
            text += f"This is further reinforced by the **{second_pos}**. "
            
        if negatives:
            top_neg = negatives[0]['feature_name'].replace('_', ' ')
            top_neg_val = negatives[0]['feature_value']
            text += f"However, the prediction is slightly offset by the **{top_neg}** (value: {top_neg_val:.2f}), which acts as a counterweight against this outcome."
            
        return text
