import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Any

# Class order matches the model: 0=Away Win, 1=Draw, 2=Home Win
CLASS_LABELS = ['Away Win', 'Draw', 'Home Win']

class MatchExplainer:
    def __init__(self, classifier_model: Any):
        self.model = classifier_model
        self.explainer = shap.TreeExplainer(self.model)

    def _shap_by_class(self, X: pd.DataFrame) -> List[np.ndarray]:
        """SHAP values for the first row of X, as per_class[class_index] -> one value per feature."""
        shap_values = self.explainer.shap_values(X)
        # Multi-class SHAP comes back either as a list of (samples, features) arrays
        # or as one (samples, features, classes) array, depending on the shap version.
        if isinstance(shap_values, list):
            return [np.asarray(sv)[0] for sv in shap_values]
        return [shap_values[0, :, i] for i in range(shap_values.shape[2])]

    def explain_prediction(self, features_df: pd.DataFrame, feature_names: List[str], predicted_class: int) -> Dict[str, Any]:
        """
        Generate SHAP values for a prediction.
        Returns:
            A dictionary containing SHAP values, feature importance, and lists of positive and negative factors.
        """
        X = features_df[feature_names]
        shap_class = self._shap_by_class(X)[predicted_class]

        feature_contributions = [
            {'feature_name': name, 'feature_value': float(val), 'shap_value': float(sv)}
            for name, val, sv in zip(feature_names, X.iloc[0].values, shap_class)
        ]
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        # Most positive first / most negative first
        positives = sorted((fc for fc in feature_contributions if fc['shap_value'] > 0), key=lambda x: -x['shap_value'])
        negatives = sorted((fc for fc in feature_contributions if fc['shap_value'] < 0), key=lambda x: x['shap_value'])

        label = CLASS_LABELS[predicted_class]
        return {
            'shap_values': {label: [fc['shap_value'] for fc in feature_contributions]},
            'feature_importance': [(fc['feature_name'], fc['shap_value']) for fc in feature_contributions],
            'positive_factors': positives[:3],
            'negative_factors': negatives[:3],
            'explanation_text': self._generate_summary_text(positives, negatives, label)
        }

    def explanation_records(self, features_df: pd.DataFrame, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Return one record per feature carrying its SHAP value for all three classes,
        shaped for DatabaseManager.save_prediction_explanations().
        """
        X = features_df[feature_names]
        per_class = self._shap_by_class(X)
        if len(per_class) < 3:
            return []
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

    def get_shap_plot_data(self, features_df: pd.DataFrame, feature_names: List[str], class_index: int) -> Dict[str, Any]:
        """Get pre-sorted data for Plotly bar chart rendering."""
        shap_class = self._shap_by_class(features_df[feature_names])[class_index]
        sorted_idx = np.argsort(np.abs(shap_class))
        return {
            'features': [feature_names[i] for i in sorted_idx],
            'shap_values': [float(shap_class[i]) for i in sorted_idx]
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
