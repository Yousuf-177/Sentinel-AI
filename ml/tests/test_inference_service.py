"""Inference service queue and pcap replay.

The SHAP check loads the saved model and is skipped when the artifact is absent.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ml" / "models" / "artifacts" / "sentinel_cnn_lstm.h5"
PCAP_DIR = ROOT / "data" / "interim" / "pcaps"
SYN_SCAN = PCAP_DIR / "syn_scan.pcap"

pytest.importorskip("numpy")


def _syn_scan_path() -> Path:
    if not SYN_SCAN.exists():
        from ml.tests.test_flow_aggregator import build_pcaps

        build_pcaps(PCAP_DIR)
    return SYN_SCAN


def test_classify_queue_stays_bounded_and_drains():
    from ml.inference.service import CLASSIFY_QUEUE_MAX, InferenceService

    class SlowPipeline:
        explainer = None
        confidence_threshold = 0.98

        def classify(self, flow):
            time.sleep(0.01)
            return {"label": "PortScan", "confidence": 0.5, "scaled_row": None}

    service = InferenceService(SlowPipeline(), explain=False)
    service.start()
    submitted = 500
    for index in range(submitted):
        service.submit_flow({
            "src_ip": "10.0.0.8",
            "dst_ip": "10.0.0.1",
            "src_port": 40000,
            "dst_port": index + 1,
            "protocol": "TCP",
        })
    service.drain()

    assert service.max_depth <= CLASSIFY_QUEUE_MAX
    assert service.dropped > 0
    assert service.classified + service.dropped == submitted
    assert service.classify_queue.qsize() == 0
    assert all(alert["explanation"] is None for alert in service.alerts)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model artifact absent")
def test_syn_scan_pcap_emits_a_valid_alert():
    from ml.inference.pipeline import validate_alert
    from ml.inference.service import InferenceService, load_runtime

    pipeline = load_runtime(explain=False)
    service = InferenceService(pipeline, explain=False)
    alerts = service.replay_pcap(str(_syn_scan_path()))

    assert len(alerts) == 1
    assert validate_alert(alerts[0]) == []
    assert alerts[0]["action"] in ("ALLOW", "ALERT", "MITIGATE")
    assert alerts[0]["explanation"] is None
    assert alerts[0]["src_ip"] == "10.1.0.8"
    assert alerts[0]["dst_port"] == 22


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model artifact absent")
def test_one_suspicious_flow_includes_a_shap_sentence():
    from ml.ingestion.flow_aggregator import FlowAggregator
    from ml.ingestion.packet_sniffer import PacketSniffer
    from ml.inference.pipeline import validate_alert
    from ml.inference.service import load_runtime

    flows = []
    aggregator = FlowAggregator(callback=flows.append)
    PacketSniffer(callback=aggregator.add_packet).sniff_pcap(str(_syn_scan_path()))
    aggregator.stop()
    assert len(flows) == 1

    pipeline = load_runtime(explain=True, background_size=20)
    alert = pipeline.process_flow(flows[0], explain=True)

    assert validate_alert(alert) == []
    assert alert["explanation"] is not None
    assert alert["explanation"]["human_readable"]
    assert len(alert["explanation"]["top_features"]) == 3
