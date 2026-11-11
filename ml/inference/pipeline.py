"""
SENTINEL-AI — Inference Pipeline

Transforms one completed network flow into a model prediction and a
schema-compatible alert. SHAP and firewall mitigation are deliberately
kept out of the packet-capture hot path.
"""

import logging
import time
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Preprocess, classify, and emit alerts for completed network flows."""

    def __init__(self, model, preprocessor, explainer=None, emitter=None):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = explainer
        self.emitter = emitter
        self.class_names = [str(name) for name in preprocessor.label_encoder.classes_]
        self.feature_names = list(preprocessor.fitted_feature_names)
        self._alert_count = 0

        output_shape = getattr(model, "output_shape", None)
        if output_shape and output_shape[-1] != len(self.class_names):
            raise ValueError(
                "Model output count does not match saved label encoder: "
                f"{output_shape[-1]} outputs vs {len(self.class_names)} classes"
            )

    @staticmethod
    def _protocol_name(value):
        """Normalize Scapy protocol numbers and names to the alert contract."""
        if isinstance(value, str):
            normalized = value.upper()
            if normalized in {"TCP", "UDP"}:
                return normalized
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = None
        return {6: "TCP", 17: "UDP"}.get(number, "OTHER")

    def process_flow(self, flow_features: dict) -> dict:
        """Classify a completed flow and return its alert payload."""
        import pandas as pd

        started = time.perf_counter()
        missing = [name for name in self.feature_names if name not in flow_features]
        if missing:
            raise ValueError(
                f"Flow is missing {len(missing)} model features; "
                f"first missing fields: {missing[:5]}"
            )

        metadata = {
            "src_ip": str(flow_features.get("src_ip", "0.0.0.0")),
            "dst_ip": str(flow_features.get("dst_ip", "0.0.0.0")),
            "src_port": int(flow_features.get("src_port", 0) or 0),
            "dst_port": int(flow_features.get("dst_port", 0) or 0),
            "protocol": self._protocol_name(flow_features.get("protocol", "OTHER")),
        }

        frame = pd.DataFrame(
            [{name: flow_features[name] for name in self.feature_names}],
            columns=self.feature_names,
        )
        transformed = self.preprocessor.transform(frame)
        probabilities = np.asarray(self.model.predict(transformed, verbose=0))[0]
        if probabilities.ndim != 1 or len(probabilities) != len(self.class_names):
            raise ValueError("Model returned an invalid probability vector")

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        label = self.class_names[predicted_index]

        # Phase 4 is detection-only. A prediction must never be presented as
        # a completed firewall action; IPS requires a separately reviewed phase.
        action = "ALLOW" if label == "Benign" else "ALERT"
        explanation = None
        if self.explainer is not None and label != "Benign":
            try:
                explanation = self.explainer.explain_alert(
                    transformed[0], self.feature_names
                )
            except Exception:
                logger.exception("Optional SHAP explanation failed")

        alert = {
            "alert_id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "flow_id": str(flow_features.get("flow_id") or uuid.uuid4()),
            **metadata,
            "label": label,
            "confidence": round(confidence, 6),
            "explanation": explanation,
            "action": action,
            "mitigation_status": "n/a",
            "inference_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

        self._alert_count += 1
        if self.emitter is not None:
            try:
                self.emitter(alert)
            except Exception:
                # A backend outage must not terminate packet capture.
                logger.exception("Alert delivery failed; alert was not delivered")

        logger.info(
            "Flow %s classified as %s (%.2f%%), action=%s",
            alert["flow_id"], label, confidence * 100, action,
        )
        return alert
