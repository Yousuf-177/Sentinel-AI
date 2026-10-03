"""
SENTINEL-AI — Inference Pipeline Module

Wires together: flow features → preprocessing → model prediction → SHAP explanation → decision routing.
Emits structured alert JSON objects over Socket.io.
"""

import time
import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Real-time inference pipeline.
    
    Takes flow feature dicts from the FlowAggregator, runs them through
    the preprocessing → model → SHAP → decision router pipeline, and
    emits alert JSON objects.
    
    Usage:
        pipeline = InferencePipeline(model, preprocessor, explainer, emitter)
        pipeline.process_flow(flow_features)
    """

    def __init__(self, model, preprocessor, explainer, emitter=None, confidence_threshold: float = 0.98):
        """
        Args:
            model: Trained Keras model.
            preprocessor: Fitted Preprocessor instance.
            explainer: SHAPExplainer instance.
            emitter: Callable that sends the alert dict (e.g., Socket.io emit).
            confidence_threshold: Threshold for auto-mitigation routing.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = explainer
        self.emitter = emitter
        self.confidence_threshold = confidence_threshold
        self.class_names = ["Benign", "DoS/DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration"]
        self._alert_count = 0

    def process_flow(self, flow_features: dict) -> dict:
        """
        Process a single flow through the full pipeline.
        
        Args:
            flow_features: Dict from FlowAggregator._compute_features()
        
        Returns:
            Alert dict with label, confidence, explanation, and action.
        """
        import pandas as pd

        start_time = time.time()

        # Extract metadata
        metadata = {
            "src_ip": flow_features.get("src_ip", "unknown"),
            "dst_ip": flow_features.get("dst_ip", "unknown"),
            "src_port": flow_features.get("src_port", 0),
            "dst_port": flow_features.get("dst_port", 0),
            "protocol": flow_features.get("protocol", "TCP"),
        }

        # Preprocess
        feature_keys = [k for k in flow_features.keys() if k not in 
                       ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "start_time"]]
        df = pd.DataFrame([{k: flow_features[k] for k in feature_keys}])
        X = self.preprocessor.transform(df)

        # Predict
        proba = self.model.predict(X, verbose=0)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])
        label = self.class_names[pred_class]

        # Explain (only for non-benign or high-confidence alerts)
        explanation = None
        if label != "Benign" or confidence < 0.9:
            try:
                explanation = self.explainer.explain_alert(
                    X[0], self.preprocessor.fitted_feature_names
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                explanation = {"top_features": [], "human_readable": "Explanation unavailable"}

        # Decision routing
        if label != "Benign" and confidence >= self.confidence_threshold:
            action = "MITIGATE"
        elif label != "Benign":
            action = "ALERT"
        else:
            action = "ALLOW"

        # Build alert
        latency_ms = (time.time() - start_time) * 1000
        alert = {
            "alert_id": str(uuid.uuid4())[:8],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "flow_id": str(uuid.uuid4())[:8],
            **metadata,
            "label": label,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "action": action,
            "mitigation_status": "pending" if action == "MITIGATE" else "n/a",
            "inference_latency_ms": round(latency_ms, 2),
        }

        self._alert_count += 1

        # Emit
        if self.emitter:
            self.emitter(alert)

        logger.debug(
            f"Alert #{self._alert_count}: {label} ({confidence:.2%}) → {action} "
            f"[{metadata['src_ip']}:{metadata['src_port']} → {metadata['dst_ip']}:{metadata['dst_port']}] "
            f"({latency_ms:.1f}ms)"
        )

        return alert
