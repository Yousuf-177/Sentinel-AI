"""Decision router and label-order checks. These tests do not load TensorFlow."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.inference.pipeline import InferencePipeline, route


def test_benign_is_allow_even_when_unsure():
    assert route("Benign", 0.99) == ("ALLOW", "n/a")
    assert route("Benign", 0.50) == ("ALLOW", "n/a")


def test_attack_below_threshold_is_alert_only():
    assert route("DoS/DDoS", 0.97) == ("ALERT", "n/a")
    assert route("PortScan", 0.979) == ("ALERT", "n/a")


def test_attack_at_threshold_would_mitigate():
    assert route("DoS/DDoS", 0.98) == ("MITIGATE", "pending")
    assert route("PortScan", 1.0) == ("MITIGATE", "pending")


class _Encoder:
    classes_ = np.array([
        "Benign", "BruteForce", "DoS/DDoS", "Infiltration", "PortScan", "WebAttack",
    ])


class _Preprocessor:
    """Returns a constant row and records which columns classify selected."""

    def __init__(self):
        self.fitted_feature_names = ["Destination Port", "Flow Duration"]
        self.label_encoder = _Encoder()
        self.seen_columns = None

    def transform(self, frame):
        self.seen_columns = list(frame.columns)
        return np.ones((1, 2), dtype=np.float32)


class _Model:
    def predict(self, features, verbose=0):
        # Index 4 is PortScan in the encoder. The old hard-coded list called it WebAttack.
        return np.array([[0.0, 0.0, 0.0, 0.0, 0.91, 0.09]])


def test_classify_uses_encoder_order_and_fitted_columns():
    preprocessor = _Preprocessor()
    pipeline = InferencePipeline(_Model(), preprocessor, explainer=None)
    flow = {
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 40000,
        "dst_port": 22,
        "protocol": "TCP",
        "Destination Port": 22,
        "Flow Duration": 1000,
        "syn_flag_count": 8,
    }

    classified = pipeline.classify(flow)

    assert classified["label"] == "PortScan"
    assert classified["confidence"] == 0.91
    assert preprocessor.seen_columns == ["Destination Port", "Flow Duration"]
    assert "syn_flag_count" not in preprocessor.seen_columns
