"""
SENTINEL-AI — Inference service

Replay a pcap, or sniff an interface, and print classified alerts.

    python -m ml.inference.service --pcap data/interim/pcaps/syn_scan.pcap

Pass --emit to also send each alert to the Node server as ``new_alert``.
Pass --no-explain to skip SHAP. Explanation stays off the capture thread
either way: a second worker handles at most one malicious flow at a time.
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import yaml

from ml.inference.pipeline import (
    InferencePipeline,
    build_alert,
    format_alert_line,
    route,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFY_QUEUE_MAX = 256
EXPLAIN_QUEUE_MAX = 2
_STOP = object()


def _read_threshold(default: float = 0.98) -> float:
    env = os.environ.get("CONFIDENCE_THRESHOLD")
    if env:
        return float(env)
    config_path = PROJECT_ROOT / "ml" / "config" / "model_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return float(config.get("inference", {}).get("confidence_threshold", default))
    return default


def _read_background_size(default: int = 20) -> int:
    env = os.environ.get("SHAP_BACKGROUND_SAMPLES")
    if env:
        return int(env)
    config_path = PROJECT_ROOT / "ml" / "config" / "model_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return int(config.get("inference", {}).get("shap_background_samples", default))
    return default


def load_runtime(explain: bool = True, background_size: int | None = None) -> InferencePipeline:
    """Load the saved model and preprocessor. Attach SHAP only when requested."""
    from ml.data.preprocessing import Preprocessor

    model_path = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "ml" / "models" / "artifacts" / "sentinel_cnn_lstm.h5"))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    preprocessor_path = Path(os.environ.get(
        "PREPROCESSOR_PATH", PROJECT_ROOT / "ml" / "models" / "artifacts" / "preprocessor.pkl"
    ))
    if not preprocessor_path.is_absolute():
        preprocessor_path = PROJECT_ROOT / preprocessor_path

    from tensorflow import keras

    logger.info(f"Loading model {model_path}")
    model = keras.models.load_model(model_path)
    preprocessor = Preprocessor.load(preprocessor_path)

    explainer = None
    if explain:
        from ml.explainability.shap_explainer import SHAPExplainer

        size = background_size if background_size is not None else _read_background_size()
        x_test = np.load(PROJECT_ROOT / "data" / "processed" / "X_test.npy", mmap_mode="r")
        take = min(size, len(x_test))
        index = np.random.default_rng(42).choice(len(x_test), size=take, replace=False)
        background = np.asarray(x_test[index], dtype=np.float32)
        logger.info(f"Fitting SHAP on {take} already-scaled background rows")
        explainer = SHAPExplainer(model, background, max_background=take)

    return InferencePipeline(
        model,
        preprocessor,
        explainer=explainer,
        confidence_threshold=_read_threshold(),
    )


class InferenceService:
    """
    Bounded classify queue plus a short explain queue.

    The sniffer callback only enqueues. Workers do the model work.
    """

    def __init__(self, pipeline: InferencePipeline, explain: bool = True, emit_url: str | None = None):
        self.pipeline = pipeline
        self.explain_enabled = bool(explain and pipeline.explainer is not None)
        self.emit_url = emit_url
        self.classify_queue = queue.Queue(maxsize=CLASSIFY_QUEUE_MAX)
        self.explain_queue = queue.Queue(maxsize=EXPLAIN_QUEUE_MAX)
        self.alerts = []
        self.dropped = 0
        self.classified = 0
        self.max_depth = 0
        self.classify_latencies_ms = []
        self._lock = threading.Lock()
        self._classify_thread = None
        self._explain_thread = None
        self._sio = None
        self._started = False

    def start(self):
        if self._started:
            return
        if self.emit_url:
            self._connect_socket()
        self._started = True
        self._classify_thread = threading.Thread(target=self._classify_loop, name="sentinel-classify", daemon=True)
        self._explain_thread = threading.Thread(target=self._explain_loop, name="sentinel-explain", daemon=True)
        self._classify_thread.start()
        self._explain_thread.start()

    def _connect_socket(self):
        import socketio

        self._sio = socketio.Client(reconnection=False)
        try:
            self._sio.connect(self.emit_url, wait_timeout=5)
            logger.info(f"Connected to alert socket {self.emit_url}")
        except Exception as exc:
            logger.warning(f"Alert socket unavailable ({exc}). Printing alerts only.")
            self._sio = None

    def submit_flow(self, flow_features: dict):
        """Enqueue one flow. Drop it when the classify queue is already full."""
        try:
            self.classify_queue.put_nowait(flow_features)
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return
        with self._lock:
            self.max_depth = max(self.max_depth, self.classify_queue.qsize())

    def _publish(self, alert: dict):
        logger.info(format_alert_line(alert))
        with self._lock:
            self.alerts.append(alert)
        if self._sio is not None and self._sio.connected:
            try:
                self._sio.emit("new_alert", json.loads(json.dumps(alert)))
            except Exception as exc:
                logger.warning(f"Failed to emit alert: {exc}")

    def _classify_loop(self):
        while True:
            flow = self.classify_queue.get()
            if flow is _STOP:
                self.explain_queue.put(_STOP)
                break
            started = time.perf_counter()
            try:
                classified = self.pipeline.classify(flow)
            except Exception as exc:
                logger.error(f"Classify failed: {exc}")
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0
            action, status = route(
                classified["label"], classified["confidence"], self.pipeline.confidence_threshold
            )
            alert = build_alert(
                flow,
                classified["label"],
                classified["confidence"],
                action,
                status,
                None,
                latency_ms,
            )
            with self._lock:
                self.classified += 1
                self.classify_latencies_ms.append(latency_ms)

            if action == "ALLOW" or not self.explain_enabled:
                self._publish(alert)
                continue
            try:
                self.explain_queue.put_nowait((alert, classified["scaled_row"], started))
            except queue.Full:
                self._publish(alert)

    def _explain_loop(self):
        while True:
            item = self.explain_queue.get()
            if item is _STOP:
                break
            alert, scaled_row, started = item
            try:
                alert["explanation"] = self.pipeline.explain(scaled_row)
            except Exception as exc:
                logger.warning(f"SHAP explanation failed: {exc}")
                alert["explanation"] = None
            alert["inference_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            self._publish(alert)

    def drain(self):
        """Block until every queued flow has been classified and explained."""
        self.classify_queue.put(_STOP)
        if self._classify_thread is not None:
            self._classify_thread.join()
        if self._explain_thread is not None:
            self._explain_thread.join()
        self._started = False
        if self._sio is not None:
            try:
                self._sio.disconnect()
            except Exception:
                pass
            self._sio = None

    def replay_pcap(self, path: str) -> list:
        """Replay a pcap through the sniffer and aggregator, then drain."""
        from ml.ingestion.flow_aggregator import FlowAggregator
        from ml.ingestion.packet_sniffer import PacketSniffer

        self.start()
        aggregator = FlowAggregator(timeout=120.0, callback=self.submit_flow)
        sniffer = PacketSniffer(callback=aggregator.add_packet)
        sniffer.sniff_pcap(path)
        aggregator.stop()
        self.drain()
        return list(self.alerts)

    def run_iface(self, interface: str):
        """Live capture until Ctrl+C. Optional on Windows; pcap replay is the checked path."""
        from ml.ingestion.flow_aggregator import FlowAggregator
        from ml.ingestion.packet_sniffer import PacketSniffer

        self.start()
        aggregator = FlowAggregator(timeout=120.0, callback=self.submit_flow)
        sniffer = PacketSniffer(interface=interface, callback=aggregator.add_packet)
        aggregator.start_cleanup()
        sniffer.start()
        logger.info(f"Capturing on {interface}. Press Ctrl+C to flush and stop.")
        try:
            while sniffer.is_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Stopping capture")
        sniffer.stop()
        aggregator.stop()
        self.drain()
        return list(self.alerts)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="SENTINEL-AI inference bridge")
    parser.add_argument("--pcap", help="Offline pcap to replay")
    parser.add_argument("--iface", help="Interface for a live capture")
    parser.add_argument("--emit", action="store_true", help="Emit new_alert to the Node server")
    parser.add_argument("--no-explain", action="store_true", help="Skip SHAP")
    parser.add_argument("--socket-url", default=os.environ.get("SOCKET_URL", "http://127.0.0.1:3001"))
    args = parser.parse_args()

    if not args.pcap and not args.iface:
        parser.error("Pass --pcap or --iface")

    pipeline = load_runtime(explain=not args.no_explain)
    service = InferenceService(
        pipeline,
        explain=not args.no_explain,
        emit_url=args.socket_url if args.emit else None,
    )
    if args.pcap:
        alerts = service.replay_pcap(args.pcap)
    else:
        alerts = service.run_iface(args.iface)
    logger.info(
        f"Finished. alerts={len(alerts)} classified={service.classified} dropped={service.dropped}"
    )


if __name__ == "__main__":
    main()
