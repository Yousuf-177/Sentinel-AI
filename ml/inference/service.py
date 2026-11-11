"""
SENTINEL-AI — Live IDS service.

Loads the trained model and fitted preprocessing artifact, captures IPv4
packets (or replays a PCAP), aggregates bidirectional flows, performs
inference, and streams alerts to the Node.js Socket.IO backend.
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

import joblib
import socketio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.flow_aggregator import FlowAggregator
from ml.ingestion.packet_sniffer import PacketSniffer
from ml.inference.pipeline import InferencePipeline

logger = logging.getLogger("sentinel.ids")


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class SocketAlertEmitter:
    """Persistent Socket.IO client used to send alerts to the backend."""

    def __init__(self, backend_url):
        self.backend_url = backend_url
        self.client = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            logger=False,
            engineio_logger=False,
        )

    def connect(self):
        self.client.connect(self.backend_url, wait_timeout=10)
        logger.info("Connected to backend Socket.IO at %s", self.backend_url)

    def __call__(self, alert):
        if not self.client.connected:
            raise ConnectionError("Backend Socket.IO client is disconnected")
        self.client.emit("new_alert", alert)

    def close(self):
        if self.client.connected:
            self.client.disconnect()


class SentinelIDSService:
    """Owns the model, flow aggregator, packet sniffer, and alert emitter."""

    def __init__(
        self,
        model_path,
        preprocessor_path,
        backend_url,
        interface="eth0",
        flow_timeout=120.0,
        bpf_filter="ip",
        no_backend=False,
    ):
        model_file = _project_path(model_path)
        preprocessor_file = _project_path(preprocessor_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {model_file}. Train/export the model "
                "and place the artifact at MODEL_PATH."
            )
        if not preprocessor_file.is_file():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {preprocessor_file}. Run "
                "python -m ml.models.build_preprocessor after preparing the "
                "CIC-IDS2017 data."
            )

        artifact = joblib.load(preprocessor_file)
        required = {"scaler", "label_encoder", "feature_names"}
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise ValueError(
                "Preprocessor artifact must contain scaler, label_encoder, "
                "and feature_names. Rebuild it with ml/models/build_preprocessor.py."
            )

        class PreprocessorAdapter:
            def __init__(self, values):
                self.scaler = values["scaler"]
                self.label_encoder = values["label_encoder"]
                self.fitted_feature_names = list(values["feature_names"])

            def transform(self, frame):
                missing = [
                    name for name in self.fitted_feature_names
                    if name not in frame.columns
                ]
                if missing:
                    raise ValueError(f"Missing preprocessor columns: {missing[:5]}")
                return self.scaler.transform(frame[self.fitted_feature_names].values)

        self.preprocessor = PreprocessorAdapter(artifact)
        self.feature_names = self.preprocessor.fitted_feature_names
        if len(self.feature_names) != 78:
            raise ValueError(
                f"Expected 78 trained CIC-IDS features, found {len(self.feature_names)}"
            )

        # TensorFlow is imported only after artifact validation so setup errors
        # remain clear even on machines without TensorFlow installed.
        import tensorflow as tf

        self.model = tf.keras.models.load_model(str(model_file), compile=False)
        input_shape = self.model.input_shape
        if input_shape[-1] != len(self.feature_names):
            raise ValueError(
                f"Model expects {input_shape[-1]} features but preprocessor has "
                f"{len(self.feature_names)}"
            )
        if self.model.output_shape[-1] != len(self.preprocessor.label_encoder.classes_):
            raise ValueError("Model output size does not match saved label encoder")

        self.emitter = None
        if not no_backend:
            self.emitter = SocketAlertEmitter(backend_url)
            self.emitter.connect()

        self.pipeline = InferencePipeline(
            model=self.model,
            preprocessor=self.preprocessor,
            emitter=self.emitter,
        )
        self.aggregator = FlowAggregator(
            timeout=flow_timeout,
            callback=self._on_flow,
            feature_names=self.feature_names,
        )
        self.sniffer = PacketSniffer(
            interface=interface,
            callback=self.aggregator.add_packet,
            bpf_filter=bpf_filter,
        )
        self._stopped = threading.Event()

    def _on_flow(self, flow):
        try:
            alert = self.pipeline.process_flow(flow)
            if self.emitter is None:
                print(json.dumps(alert, ensure_ascii=False), flush=True)
        except Exception:
            logger.exception("Failed to process completed flow")

    def run_pcap(self, pcap_path):
        """Replay a PCAP and flush all flows at end of file."""
        try:
            self.sniffer.sniff_pcap(pcap_path)
        finally:
            self.aggregator.stop()
            self.close()

    def run_live(self):
        """Start live capture until SIGINT/SIGTERM or capture failure."""
        self.aggregator.start_cleanup()
        self.sniffer.start()
        logger.info("Live IDS started on interface %s", self.sniffer.interface)
        try:
            while not self._stopped.wait(1):
                if not self.sniffer.is_running:
                    logger.error("Packet capture stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.stop()

    def stop(self):
        self._stopped.set()
        self.sniffer.stop()
        self.aggregator.stop()
        self.close()

    def close(self):
        if self.emitter is not None:
            self.emitter.close()


def load_runtime_config():
    config_path = _project_path(os.getenv("MODEL_CONFIG_PATH", "ml/config/model_config.yaml"))
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser(description="SENTINEL-AI live intrusion detection service")
    parser.add_argument("--pcap", help="Replay a PCAP file instead of capturing live traffic")
    parser.add_argument("--iface", default=os.getenv("CAPTURE_INTERFACE", "eth0"))
    parser.add_argument("--bpf-filter", default=os.getenv("CAPTURE_BPF_FILTER", "ip"))
    parser.add_argument("--no-backend", action="store_true", help="Print alerts as JSON instead of sending to backend")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_runtime_config()
    inference_cfg = config.get("inference", {})
    service = SentinelIDSService(
        model_path=os.getenv("MODEL_PATH", config["paths"]["model_artifact"]),
        preprocessor_path=os.getenv("PREPROCESSOR_PATH", config["paths"]["preprocessor"]),
        backend_url=os.getenv("SENTINEL_BACKEND_URL", "http://127.0.0.1:3001"),
        interface=args.iface,
        flow_timeout=float(os.getenv(
            "FLOW_TIMEOUT_SECONDS",
            inference_cfg.get("flow_timeout_seconds", 120),
        )),
        bpf_filter=args.bpf_filter,
        no_backend=args.no_backend,
    )

    if args.pcap:
        service.run_pcap(args.pcap)
    else:
        service.run_live()


if __name__ == "__main__":
    main()
