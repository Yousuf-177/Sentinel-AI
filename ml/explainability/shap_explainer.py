"""
SENTINEL-AI — SHAP Explainer Module

Generates human-readable SHAP-based explanations for model predictions.
"""

import logging
from pathlib import Path

import numpy as np
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wraps SHAP DeepExplainer for the CNN-LSTM model.
    
    Generates per-alert feature importance explanations and formats them
    as human-readable strings for the dashboard.
    
    Usage:
        explainer = SHAPExplainer(model, background_data)
        explanation = explainer.explain_alert(flow_features, feature_names)
    """

    def __init__(self, model, background_data: np.ndarray, max_background: int = 100):
        """
        Args:
            model: Trained Keras model.
            background_data: Sample of training data for SHAP background.
            max_background: Max number of background samples (controls speed).
        """
        # Subsample background if needed
        if len(background_data) > max_background:
            indices = np.random.choice(len(background_data), max_background, replace=False)
            background_data = background_data[indices]

        self.explainer = shap.DeepExplainer(model, background_data)
        self.background_mean = background_data.mean(axis=0)
        logger.info(f"SHAP DeepExplainer initialized with {len(background_data)} background samples")

    def explain_alert(self, flow_features: np.ndarray, feature_names: list, top_k: int = 3) -> dict:
        """
        Generate a SHAP explanation for a single flow.
        
        Args:
            flow_features: Feature vector for one flow (1D array).
            feature_names: List of feature names.
            top_k: Number of top contributing features to return.
        
        Returns:
            dict with 'top_features' list and 'human_readable' string.
        """
        # Ensure 2D input
        if flow_features.ndim == 1:
            flow_features = flow_features.reshape(1, -1)

        shap_values = self.explainer.shap_values(flow_features)

        # Get the predicted class
        pred_class = np.argmax(np.abs(shap_values).sum(axis=0).sum(axis=0))

        # Get SHAP values for the predicted class
        class_shap = shap_values[pred_class][0] if isinstance(shap_values, list) else shap_values[0]

        # Rank features by absolute SHAP value
        feature_importance = np.abs(class_shap)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]

        top_features = []
        for idx in top_indices:
            pct_change = ((flow_features[0, idx] - self.background_mean[idx]) / 
                         (abs(self.background_mean[idx]) + 1e-8)) * 100
            top_features.append({
                "feature": feature_names[idx],
                "shap_value": float(class_shap[idx]),
                "direction": "positive" if class_shap[idx] > 0 else "negative",
                "pct_vs_baseline": f"{pct_change:+.0f}%",
            })

        # Build human-readable explanation
        parts = []
        for feat in top_features:
            parts.append(f"{feat['feature']}: {feat['pct_vs_baseline']} vs baseline")
        human_readable = "; ".join(parts)

        return {
            "top_features": top_features,
            "human_readable": human_readable,
        }

    def explain_batch(self, flows: np.ndarray, feature_names: list, top_k: int = 3) -> list:
        """
        Generate explanations for a batch of flows.
        """
        return [self.explain_alert(flows[i], feature_names, top_k) for i in range(len(flows))]
