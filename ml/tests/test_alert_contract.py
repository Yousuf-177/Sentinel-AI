"""Alert JSON contract. A missing explanation is null, never an invented object."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.inference.pipeline import build_alert, route, validate_alert

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ml" / "config" / "alert_schema.json"

FIXTURE_FLOW = {
    "src_ip": "10.1.0.8",
    "dst_ip": "10.1.0.1",
    "src_port": 40000,
    "dst_port": 22,
    "protocol": "TCP",
}


def test_fixture_flow_matches_schema_enums():
    action, status = route("PortScan", 0.99)
    alert = build_alert(FIXTURE_FLOW, "PortScan", 0.99, action, status, None, 12.5)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for key in schema["required"]:
        assert key in alert
    assert alert["label"] in schema["properties"]["label"]["enum"]
    assert alert["action"] in schema["properties"]["action"]["enum"]
    assert alert["action"] == "MITIGATE"
    assert alert["mitigation_status"] == "pending"
    assert alert["explanation"] is None
    assert validate_alert(alert) == []


def test_allow_explanation_is_null():
    action, status = route("Benign", 0.995)
    alert = build_alert(FIXTURE_FLOW, "Benign", 0.995, action, status, None, 3.0)
    assert alert["action"] == "ALLOW"
    assert alert["explanation"] is None
    assert validate_alert(alert) == []


def test_present_explanation_must_carry_the_phase3_fields():
    explanation = {
        "top_features": [{
            "feature": "SYN Flag Count",
            "shap_value": 0.2,
            "direction": "positive",
            "pct_vs_baseline": "+400%",
        }],
        "human_readable": "SYN Flag Count: +400% vs baseline",
    }
    action, status = route("PortScan", 0.5)
    alert = build_alert(FIXTURE_FLOW, "PortScan", 0.5, action, status, explanation, 4.0)
    assert alert["action"] == "ALERT"
    assert validate_alert(alert) == []
