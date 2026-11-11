"""Tests for the model-to-alert inference contract."""

import numpy as np
import pandas as pd
import pytest

from ml.inference.pipeline import InferencePipeline


FEATURES = ["f1", "f2"]


class FakeEncoder:
    # Deliberately use the training CLASS_ORDER, not a hand-written inference order.
    classes_ = np.asarray([
        "Benign", "DoS/DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration"
    ])


class FakePreprocessor:
    label_encoder = FakeEncoder()
    fitted_feature_names = FEATURES

    def transform(self, frame):
        assert list(frame.columns) == FEATURES
        return frame.to_numpy(dtype=float)


class FakeModel:
    output_shape = (None, 6)

    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=float)

    def predict(self, values, verbose=0):
        assert values.shape == (1, 2)
        return self.prediction.reshape(1, -1)


def _flow(protocol="TCP"):
    return {
        "src_ip": "192.0.2.10",
        "dst_ip": "198.51.100.20",
        "src_port": 43210,
        "dst_port": 443,
        "protocol": protocol,
        "f1": 0.25,
        "f2": 1.5,
    }


def test_class_label_comes_from_saved_encoder():
    emitted = []
    model = FakeModel([0.01, 0.02, 0.03, 0.90, 0.02, 0.02])
    pipeline = InferencePipeline(model, FakePreprocessor(), emitter=emitted.append)

    alert = pipeline.process_flow(_flow())

    assert alert["label"] == "BruteForce"
    assert alert["action"] == "ALERT"
    assert alert["mitigation_status"] == "n/a"
    assert emitted == [alert]


def test_benign_prediction_is_allow():
    pipeline = InferencePipeline(
        FakeModel([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
        FakePreprocessor(),
    )

    alert = pipeline.process_flow(_flow())

    assert alert["label"] == "Benign"
    assert alert["action"] == "ALLOW"


@pytest.mark.parametrize(("protocol", "expected"), [
    ("TCP", "TCP"),
    ("UDP", "UDP"),
    (6, "TCP"),
    (17, "UDP"),
    (1, "OTHER"),
    ("unknown", "OTHER"),
])
def test_protocol_matches_alert_schema(protocol, expected):
    pipeline = InferencePipeline(
        FakeModel([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
        FakePreprocessor(),
    )

    assert pipeline.process_flow(_flow(protocol))["protocol"] == expected


def test_missing_model_feature_fails_closed():
    pipeline = InferencePipeline(
        FakeModel([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
        FakePreprocessor(),
    )
    flow = _flow()
    del flow["f2"]

    with pytest.raises(ValueError, match="missing"):
        pipeline.process_flow(flow)


def test_emitter_failure_does_not_change_prediction():
    def broken_emitter(_alert):
        raise ConnectionError("backend offline")

    pipeline = InferencePipeline(
        FakeModel([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
        FakePreprocessor(),
        emitter=broken_emitter,
    )

    alert = pipeline.process_flow(_flow())

    assert alert["action"] == "ALLOW"
