"""
SENTINEL-AI — Flow Aggregator Module

Aggregates raw packets into network flows using a hash-based flow table.
Emits feature vectors in the same order as data/processed/feature_names.json
(the 78 CIC-IDS columns the saved model was trained on) and keeps the
42 snake_case names from feature_schema.json as aliases.
"""

import json
import time
import logging
import hashlib
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# CICFlowMeter treats a gap longer than this as idle time, not active time.
ACTIVE_TIMEOUT_US = 5_000_000.0
US = 1_000_000.0

# Snake_case schema names that overlap the trained CIC columns.
SCHEMA_ALIASES = {
    "flow_duration": "Flow Duration",
    "total_fwd_packets": "Total Fwd Packets",
    "total_bwd_packets": "Total Backward Packets",
    "total_length_fwd_packets": "Total Length of Fwd Packets",
    "total_length_bwd_packets": "Total Length of Bwd Packets",
    "fwd_packet_length_max": "Fwd Packet Length Max",
    "fwd_packet_length_min": "Fwd Packet Length Min",
    "fwd_packet_length_mean": "Fwd Packet Length Mean",
    "fwd_packet_length_std": "Fwd Packet Length Std",
    "bwd_packet_length_max": "Bwd Packet Length Max",
    "bwd_packet_length_min": "Bwd Packet Length Min",
    "bwd_packet_length_mean": "Bwd Packet Length Mean",
    "bwd_packet_length_std": "Bwd Packet Length Std",
    "flow_bytes_per_s": "Flow Bytes/s",
    "flow_packets_per_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_total": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_total": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "fwd_header_length": "Fwd Header Length",
    "bwd_header_length": "Bwd Header Length",
    "fwd_packets_per_s": "Fwd Packets/s",
    "bwd_packets_per_s": "Bwd Packets/s",
    "packet_length_min": "Min Packet Length",
    "packet_length_max": "Max Packet Length",
    "packet_length_mean": "Packet Length Mean",
    "packet_length_std": "Packet Length Std",
    "packet_length_variance": "Packet Length Variance",
    "syn_flag_count": "SYN Flag Count",
    "ack_flag_count": "ACK Flag Count",
}

FLAG_FIELDS = (
    ("fin", "FIN Flag Count"),
    ("syn", "SYN Flag Count"),
    ("rst", "RST Flag Count"),
    ("psh", "PSH Flag Count"),
    ("ack", "ACK Flag Count"),
    ("urg", "URG Flag Count"),
    ("cwr", "CWE Flag Count"),
    ("ece", "ECE Flag Count"),
)

# Bits used when a caller only supplies the combined tcp_flags integer.
FLAG_BITS = {
    "fin": 0x01,
    "syn": 0x02,
    "rst": 0x04,
    "psh": 0x08,
    "ack": 0x10,
    "urg": 0x20,
    "ece": 0x40,
    "cwr": 0x80,
}


def load_feature_names(path: str | None = None) -> list:
    """Load the ordered 78 CIC-IDS names the saved model was trained on."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "processed" / "feature_names.json"
    with open(path, "r", encoding="utf-8") as handle:
        names = json.load(handle)
    if len(names) != 78:
        logger.warning(f"Expected 78 feature names, found {len(names)} in {path}")
    return names


def _length_stats(values: list) -> tuple:
    """Return min, max, mean, std for a list of lengths. Empty lists are zeros."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())


def _iat_stats_us(timestamps: list) -> tuple:
    """Return total, mean, std, max, min inter-arrival time in microseconds."""
    if len(timestamps) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    iats = np.diff(np.asarray(sorted(timestamps), dtype=np.float64)) * US
    iats = np.maximum(iats, 0.0)
    return (
        float(iats.sum()),
        float(iats.mean()),
        float(iats.std()),
        float(iats.max()),
        float(iats.min()),
    )


def _active_idle_stats(timestamps: list) -> tuple:
    """
    Split inter-arrival gaps into active and idle periods.

    A gap longer than 5 seconds is idle. Shorter gaps accumulate into the
    current active period. Returns (active_mean, std, max, min, idle_mean, std, max, min)
    in microseconds.
    """
    if len(timestamps) < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ordered = np.asarray(sorted(timestamps), dtype=np.float64)
    gaps_us = np.maximum(np.diff(ordered) * US, 0.0)
    active = []
    idle = []
    current = 0.0
    for gap in gaps_us:
        if gap > ACTIVE_TIMEOUT_US:
            if current > 0.0:
                active.append(current)
                current = 0.0
            idle.append(float(gap))
        else:
            current += float(gap)
    if current > 0.0:
        active.append(current)

    def pack(periods: list) -> tuple:
        if not periods:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(periods, dtype=np.float64)
        return float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min())

    return pack(active) + pack(idle)


class FlowAggregator:
    """
    Hash-based flow aggregation with timeout-based emission.

    Groups packets into bidirectional flows by 5-tuple, computes CICFlowMeter
    features, and emits a dict when a flow expires or times out.

    Usage:
        aggregator = FlowAggregator(timeout=120, callback=on_flow_ready)
        aggregator.add_packet(packet_info)
    """

    def __init__(self, timeout: float = 120.0, callback=None, feature_names: list | None = None):
        """
        Args:
            timeout: Flow inactivity timeout in seconds.
            callback: Function called with (flow_features: dict) when a flow is emitted.
            feature_names: Ordered model feature names. Defaults to feature_names.json.
        """
        self.timeout = float(timeout)
        self.callback = callback
        self.flow_table = {}
        self.feature_names = list(feature_names) if feature_names is not None else load_feature_names()
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._running = False

    def _flow_key(self, packet_info: dict) -> str:
        """
        Generate a bidirectional flow key from the 5-tuple.
        Ensures (A→B) and (B→A) map to the same flow.
        """
        src = (packet_info["src_ip"], packet_info["src_port"])
        dst = (packet_info["dst_ip"], packet_info["dst_port"])
        proto = packet_info.get("protocol", 6)

        if src > dst:
            src, dst = dst, src

        key_str = f"{src[0]}:{src[1]}-{dst[0]}:{dst[1]}-{proto}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def add_packet(self, packet_info: dict):
        """
        Add a packet to the flow table.
        Creates a new flow entry if needed, or updates an existing one.
        """
        flow_key = self._flow_key(packet_info)
        timestamp = packet_info["timestamp"]

        with self._lock:
            if flow_key not in self.flow_table:
                self.flow_table[flow_key] = self._create_flow(packet_info, timestamp)
            flow = self.flow_table[flow_key]
            is_forward = packet_info["src_ip"] == flow["src_ip"]
            self._update_flow(flow, packet_info, is_forward, timestamp)

    def _create_flow(self, packet_info: dict, timestamp: float) -> dict:
        """Initialize a new flow entry. The first packet defines the forward direction."""
        return {
            "src_ip": packet_info["src_ip"],
            "dst_ip": packet_info["dst_ip"],
            "src_port": packet_info["src_port"],
            "dst_port": packet_info["dst_port"],
            "protocol": packet_info.get("protocol_name", "TCP"),
            "start_time": timestamp,
            "last_time": timestamp,
            "fwd_lengths": [],
            "bwd_lengths": [],
            "fwd_timestamps": [],
            "bwd_timestamps": [],
            "fwd_header_lengths": [],
            "bwd_header_lengths": [],
            "fwd_payloads": [],
            "init_win_fwd": None,
            "init_win_bwd": None,
            "flag_counts": {name: 0 for name in FLAG_BITS},
            "fwd_psh_flags": 0,
            "bwd_psh_flags": 0,
            "fwd_urg_flags": 0,
            "bwd_urg_flags": 0,
        }

    def _flag_counts(self, packet_info: dict) -> dict:
        """Read per-flag 0/1 values, falling back to the combined tcp_flags mask."""
        flags = int(packet_info.get("tcp_flags", 0) or 0)
        counts = {}
        for name, bit in FLAG_BITS.items():
            if name in packet_info:
                counts[name] = int(bool(packet_info[name]))
            else:
                counts[name] = int(bool(flags & bit))
        return counts

    def _update_flow(self, flow: dict, packet_info: dict, is_forward: bool, timestamp: float):
        """Update flow statistics with a new packet."""
        flow["last_time"] = timestamp
        pkt_len = float(packet_info.get("ip_len", 0) or 0)
        header_len = float(packet_info.get("header_len", 20) or 0)
        payload_len = float(packet_info.get("payload_len", 0) or 0)
        window = packet_info.get("window", 0)
        flags = self._flag_counts(packet_info)

        if is_forward:
            flow["fwd_lengths"].append(pkt_len)
            flow["fwd_timestamps"].append(timestamp)
            flow["fwd_header_lengths"].append(header_len)
            flow["fwd_payloads"].append(payload_len)
            if flow["init_win_fwd"] is None and "window" in packet_info:
                flow["init_win_fwd"] = int(window or 0)
            if flags["psh"]:
                flow["fwd_psh_flags"] += 1
            if flags["urg"]:
                flow["fwd_urg_flags"] += 1
        else:
            flow["bwd_lengths"].append(pkt_len)
            flow["bwd_timestamps"].append(timestamp)
            flow["bwd_header_lengths"].append(header_len)
            if flow["init_win_bwd"] is None and "window" in packet_info:
                flow["init_win_bwd"] = int(window or 0)
            if flags["psh"]:
                flow["bwd_psh_flags"] += 1
            if flags["urg"]:
                flow["bwd_urg_flags"] += 1

        for name, value in flags.items():
            flow["flag_counts"][name] += value

    def emit_flow(self, flow_key: str):
        """Compute features for a flow and emit via callback."""
        with self._lock:
            if flow_key not in self.flow_table:
                return
            flow = self.flow_table.pop(flow_key)

        features = self._compute_features(flow)
        if self.callback:
            self.callback(features)

    def to_feature_vector(self, features: dict) -> np.ndarray:
        """Stack model features in feature_names order. Shape (78,)."""
        return np.asarray([float(features[name]) for name in self.feature_names], dtype=np.float64)

    def _compute_features(self, flow: dict) -> dict:
        """
        Compute CICFlowMeter-compatible features.

        Time features (duration, IAT, active, idle) are microseconds, matching
        the CIC-IDS2017 columns. Rates stay in per-second units.
        """
        duration_s = max(flow["last_time"] - flow["start_time"], 0.0)
        duration_us = duration_s * US
        rate_duration = duration_s if duration_s > 0.0 else 0.0

        fwd = flow["fwd_lengths"]
        bwd = flow["bwd_lengths"]
        all_lengths = fwd + bwd

        fwd_min, fwd_max, fwd_mean, fwd_std = _length_stats(fwd)
        bwd_min, bwd_max, bwd_mean, bwd_std = _length_stats(bwd)
        pkt_min, pkt_max, pkt_mean, pkt_std = _length_stats(all_lengths)

        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _iat_stats_us(
            flow["fwd_timestamps"] + flow["bwd_timestamps"]
        )
        fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _iat_stats_us(
            flow["fwd_timestamps"]
        )
        bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _iat_stats_us(
            flow["bwd_timestamps"]
        )
        active_mean, active_std, active_max, active_min, idle_mean, idle_std, idle_max, idle_min = (
            _active_idle_stats(flow["fwd_timestamps"] + flow["bwd_timestamps"])
        )

        total_fwd_bytes = float(sum(fwd)) if fwd else 0.0
        total_bwd_bytes = float(sum(bwd)) if bwd else 0.0
        total_bytes = total_fwd_bytes + total_bwd_bytes
        total_fwd = len(fwd)
        total_bwd = len(bwd)
        total_pkts = total_fwd + total_bwd

        def per_second(count: float) -> float:
            if rate_duration <= 0.0:
                return 0.0
            return float(count) / rate_duration

        fwd_header = float(sum(flow["fwd_header_lengths"])) if flow["fwd_header_lengths"] else 0.0
        bwd_header = float(sum(flow["bwd_header_lengths"])) if flow["bwd_header_lengths"] else 0.0
        min_seg = float(min(flow["fwd_header_lengths"])) if flow["fwd_header_lengths"] else 0.0
        act_data = sum(1 for size in flow["fwd_payloads"] if size > 0)
        down_up = (total_bwd / total_fwd) if total_fwd else 0.0
        avg_packet = (total_bytes / total_pkts) if total_pkts else 0.0

        features = {
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "protocol": flow["protocol"],
            "start_time": flow["start_time"],
            "Destination Port": flow["dst_port"],
            "Flow Duration": duration_us,
            "Total Fwd Packets": total_fwd,
            "Total Backward Packets": total_bwd,
            "Total Length of Fwd Packets": total_fwd_bytes,
            "Total Length of Bwd Packets": total_bwd_bytes,
            "Fwd Packet Length Max": fwd_max,
            "Fwd Packet Length Min": fwd_min,
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": fwd_std,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Min": bwd_min,
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": bwd_std,
            "Flow Bytes/s": per_second(total_bytes),
            "Flow Packets/s": per_second(total_pkts),
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            "Fwd IAT Total": fwd_iat_total,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Total": bwd_iat_total,
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd PSH Flags": flow["fwd_psh_flags"],
            "Bwd PSH Flags": flow["bwd_psh_flags"],
            "Fwd URG Flags": flow["fwd_urg_flags"],
            "Bwd URG Flags": flow["bwd_urg_flags"],
            "Fwd Header Length": fwd_header,
            "Bwd Header Length": bwd_header,
            "Fwd Packets/s": per_second(total_fwd),
            "Bwd Packets/s": per_second(total_bwd),
            "Min Packet Length": pkt_min,
            "Max Packet Length": pkt_max,
            "Packet Length Mean": pkt_mean,
            "Packet Length Std": pkt_std,
            "Packet Length Variance": pkt_std ** 2,
            "FIN Flag Count": flow["flag_counts"]["fin"],
            "SYN Flag Count": flow["flag_counts"]["syn"],
            "RST Flag Count": flow["flag_counts"]["rst"],
            "PSH Flag Count": flow["flag_counts"]["psh"],
            "ACK Flag Count": flow["flag_counts"]["ack"],
            "URG Flag Count": flow["flag_counts"]["urg"],
            "CWE Flag Count": flow["flag_counts"]["cwr"],
            "ECE Flag Count": flow["flag_counts"]["ece"],
            "Down/Up Ratio": down_up,
            "Average Packet Size": avg_packet,
            "Avg Fwd Segment Size": fwd_mean,
            "Avg Bwd Segment Size": bwd_mean,
            "Fwd Header Length.1": fwd_header,
            # Bulk statistics need a full CICFlowMeter bulk detector. They are
            # zero for the large majority of CIC-IDS2017 rows, so live flows
            # use that same fill value.
            "Fwd Avg Bytes/Bulk": 0.0,
            "Fwd Avg Packets/Bulk": 0.0,
            "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0,
            "Bwd Avg Packets/Bulk": 0.0,
            "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": total_fwd,
            "Subflow Fwd Bytes": total_fwd_bytes,
            "Subflow Bwd Packets": total_bwd,
            "Subflow Bwd Bytes": total_bwd_bytes,
            "Init_Win_bytes_forward": 0 if flow["init_win_fwd"] is None else flow["init_win_fwd"],
            "Init_Win_bytes_backward": 0 if flow["init_win_bwd"] is None else flow["init_win_bwd"],
            "act_data_pkt_fwd": act_data,
            "min_seg_size_forward": min_seg,
            "Active Mean": active_mean,
            "Active Std": active_std,
            "Active Max": active_max,
            "Active Min": active_min,
            "Idle Mean": idle_mean,
            "Idle Std": idle_std,
            "Idle Max": idle_max,
            "Idle Min": idle_min,
        }

        for alias, source in SCHEMA_ALIASES.items():
            features[alias] = features[source]
        return features

    def start_cleanup(self):
        """Start background thread to emit expired flows."""
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info(f"Flow cleanup thread started (timeout={self.timeout}s)")

    def _cleanup_loop(self):
        """Periodically check for expired flows and emit them."""
        while self._running:
            time.sleep(min(self.timeout / 4, 10))
            now = time.time()
            expired_keys = []
            with self._lock:
                for key, flow in self.flow_table.items():
                    if now - flow["last_time"] > self.timeout:
                        expired_keys.append(key)
            for key in expired_keys:
                self.emit_flow(key)

    def stop(self):
        """Stop the cleanup thread and emit all remaining flows."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

        remaining_keys = list(self.flow_table.keys())
        for key in remaining_keys:
            self.emit_flow(key)

        logger.info("Flow aggregator stopped. All remaining flows emitted.")
