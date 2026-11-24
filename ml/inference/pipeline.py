"""
SENTINEL-AI — Inference Pipeline

Classifies one flow, routes it, and optionally explains it.

Classification and explanation are separate so a slow SHAP call does not
sit on the capture thread. MITIGATE means the flow would be blocked. This
module does not change firewall state.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METADATA_KEYS = ("src_ip", "dst_ip", "src_port", "dst_port", "protocol", "start_time")
ACTIONS = ("ALLOW", "ALERT", "MITIGATE")
MITIGATION_STATUSES = ("n/a", "pending", "applied", "failed", "reversed")


def route(label: str, confidence: float, threshold: float = 0.98) -> tuple[str, str]:
    """
    Choose an action from the predicted label and confidence.

    Benign traffic is ALLOW. An attack at or above ``threshold`` is MITIGATE
    (would block; no firewall change). Any other attack is ALERT.

    Returns:
        (action, mitigation_status)
    """
    if label == "Benign":
        return "ALLOW", "n/a"
    if float(confidence) >= float(threshold):
        return "MITIGATE", "pending"
    return "ALERT", "n/a"


def build_alert(
    flow_features: dict,
    label: str,
    confidence: float,
    action: str,
    mitigation_status: str,
    explanation,
    latency_ms: float,
) -> dict:
    """Assemble one alert object matching ml/config/alert_schema.json."""
    protocol = str(flow_features.get("protocol", "TCP"))
    if protocol not in ("TCP", "UDP", "OTHER"):
        protocol = "OTHER"

    return {
        "alert_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flow_id": uuid.uuid4().hex[:8],
        "src_ip": str(flow_features.get("src_ip", "unknown")),
        "dst_ip": str(flow_features.get("dst_ip", "unknown")),
        "src_port": int(flow_features.get("src_port", 0)),
        "dst_port": int(flow_features.get("dst_port", 0)),
        "protocol": protocol,
        "label": str(label),
        "confidence": round(float(confidence), 4),
        "explanation": explanation,
        "action": action,
        "mitigation_status": mitigation_status,
        "inference_latency_ms": round(float(latency_ms), 2),
    }


def format_alert_line(alert: dict) -> str:
    """One console line: label, confidence, action, 5-tuple, explanation."""
    explanation = alert.get("explanation") or {}
    text = explanation.get("human_readable") if isinstance(explanation, dict) else None
    suffix = f" | {text}" if text else ""
    return (
        f"{alert['label']} {alert['confidence']:.4f} {alert['action']} "
        f"{alert['src_ip']}:{alert['src_port']} -> {alert['dst_ip']}:{alert['dst_port']}"
        f"{suffix}"
    )


def validate_alert(alert: dict, labels: list | None = None) -> list[str]:
    """
    Return a list of contract violations. An empty list means the alert is valid.

    ``explanation`` may be null. When it is present it must carry the Phase 3
    top-3 object.
    """
    errors = []
    required = (
        "alert_id", "timestamp", "flow_id", "src_ip", "dst_ip", "src_port",
        "dst_port", "protocol", "label", "confidence", "action",
    )
    for key in required:
        if key not in alert:
            errors.append(f"missing {key}")

    if alert.get("protocol") not in ("TCP", "UDP", "OTHER"):
        errors.append(f"bad protocol {alert.get('protocol')}")
    if alert.get("action") not in ACTIONS:
        errors.append(f"bad action {alert.get('action')}")
    if alert.get("mitigation_status", "n/a") not in MITIGATION_STATUSES:
        errors.append(f"bad mitigation_status {alert.get('mitigation_status')}")

    allowed_labels = labels or (
        "Benign", "BruteForce", "DoS/DDoS", "Infiltration", "PortScan", "WebAttack",
    )
    if alert.get("label") not in allowed_labels:
        errors.append(f"bad label {alert.get('label')}")

    confidence = alert.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"bad confidence {confidence}")

    for port_key in ("src_port", "dst_port"):
        port = alert.get(port_key)
        if not isinstance(port, int) or not 0 <= port <= 65535:
            errors.append(f"bad {port_key} {port}")

    explanation = alert.get("explanation", None)
    if explanation is not None:
        if not isinstance(explanation, dict):
            errors.append("explanation must be an object or null")
        elif "human_readable" not in explanation or "top_features" not in explanation:
            errors.append("explanation missing top_features or human_readable")

    return errors


class InferencePipeline:
    """
    Scores one flow dict from FlowAggregator.

    ``classify`` only runs the model. ``explain`` only runs SHAP. Callers
    that want both in one step use ``process_flow``.
    """

    def __init__(self, model, preprocessor, explainer=None, confidence_threshold: float = 0.98):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = explainer
        self.confidence_threshold = float(confidence_threshold)
        self._model_lock = threading.Lock()

        encoder = getattr(preprocessor, "label_encoder", None)
        classes = getattr(encoder, "classes_", None)
        if classes is None:
            raise ValueError("preprocessor.label_encoder.classes_ is required")
        self.class_names = [str(name) for name in classes]
        self.feature_names = list(preprocessor.fitted_feature_names)

    def classify(self, flow_features: dict) -> dict:
        """
        Scale the 78 CIC columns and predict.

        The row passed to the scaler is built from ``fitted_feature_names``
        only. Alias keys in the flow dict are ignored. ``X_test.npy`` is
        already scaled and must not be passed through this method.
        """
        missing = [name for name in self.feature_names if name not in flow_features]
        if missing:
            raise KeyError(f"Flow is missing {len(missing)} model features, first: {missing[0]}")

        frame = pd.DataFrame([{name: flow_features[name] for name in self.feature_names}])
        scaled = self.preprocessor.transform(frame)
        with self._model_lock:
            proba = np.asarray(self.model.predict(scaled, verbose=0))[0]
        pred_class = int(np.argmax(proba))
        return {
            "label": self.class_names[pred_class],
            "confidence": float(proba[pred_class]),
            "scaled_row": np.asarray(scaled[0], dtype=np.float32),
            "metadata": {key: flow_features.get(key) for key in METADATA_KEYS if key != "start_time"},
        }

    def explain(self, scaled_row: np.ndarray) -> dict | None:
        """Explain one already-scaled row. Returns None when no explainer is attached."""
        if self.explainer is None:
            return None
        with self._model_lock:
            return self.explainer.explain_alert(scaled_row, self.feature_names)

    def process_flow(self, flow_features: dict, explain: bool | None = None) -> dict:
        """
        Classify, route, and optionally explain one flow on the caller thread.

        Used by the single-flow SHAP check. The service uses ``classify`` and
        ``explain`` on separate workers instead.
        """
        import time

        started = time.perf_counter()
        classified = self.classify(flow_features)
        action, status = route(
            classified["label"], classified["confidence"], self.confidence_threshold
        )
        do_explain = explain if explain is not None else (action != "ALLOW" and self.explainer is not None)
        explanation = self.explain(classified["scaled_row"]) if do_explain else None
        alert = build_alert(
            flow_features,
            classified["label"],
            classified["confidence"],
            action,
            status,
            explanation,
            (time.perf_counter() - started) * 1000.0,
        )
        logger.info(format_alert_line(alert))
        return alert
