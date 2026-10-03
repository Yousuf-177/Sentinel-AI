"""
SENTINEL-AI — Flow Aggregator Module

Aggregates raw packets into network flows using a hash-based flow table.
Emits feature vectors matching the feature_schema.json when flows expire.
"""

import time
import logging
import hashlib
import threading
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class FlowAggregator:
    """
    Hash-based flow aggregation with timeout-based emission.
    
    Groups packets into bidirectional flows by 5-tuple, computes statistical
    features (CICFlowMeter-compatible), and emits feature vectors when a
    flow expires or times out.
    
    Usage:
        aggregator = FlowAggregator(timeout=120, callback=on_flow_ready)
        aggregator.add_packet(packet_info)  # Called by the sniffer
    """

    def __init__(self, timeout: float = 120.0, callback=None):
        """
        Args:
            timeout: Flow inactivity timeout in seconds.
            callback: Function called with (flow_features: dict) when a flow is emitted.
        """
        self.timeout = timeout
        self.callback = callback
        self.flow_table = {}
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

        # Sort to make bidirectional
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
        is_forward = self._is_forward(packet_info, flow_key)
        timestamp = packet_info["timestamp"]

        with self._lock:
            if flow_key not in self.flow_table:
                self.flow_table[flow_key] = self._create_flow(packet_info, timestamp)

            flow = self.flow_table[flow_key]
            self._update_flow(flow, packet_info, is_forward, timestamp)

    def _is_forward(self, packet_info: dict, flow_key: str) -> bool:
        """Determine if this packet is in the forward or backward direction."""
        with self._lock:
            if flow_key in self.flow_table:
                return packet_info["src_ip"] == self.flow_table[flow_key]["src_ip"]
        return True  # First packet defines forward direction

    def _create_flow(self, packet_info: dict, timestamp: float) -> dict:
        """Initialize a new flow entry."""
        return {
            "src_ip": packet_info["src_ip"],
            "dst_ip": packet_info["dst_ip"],
            "src_port": packet_info["src_port"],
            "dst_port": packet_info["dst_port"],
            "protocol": packet_info.get("protocol_name", "TCP"),
            "start_time": timestamp,
            "last_time": timestamp,
            # Forward direction stats
            "fwd_packets": [],
            "fwd_timestamps": [],
            "fwd_psh_flags": 0,
            "fwd_urg_flags": 0,
            "fwd_header_lengths": [],
            # Backward direction stats
            "bwd_packets": [],
            "bwd_timestamps": [],
            "bwd_header_lengths": [],
            # Overall flags
            "syn_count": 0,
            "ack_count": 0,
        }

    def _update_flow(self, flow: dict, packet_info: dict, is_forward: bool, timestamp: float):
        """Update flow statistics with a new packet."""
        flow["last_time"] = timestamp
        pkt_len = packet_info.get("ip_len", 0)
        header_len = packet_info.get("header_len", 20)
        tcp_flags = packet_info.get("tcp_flags", 0)

        if is_forward:
            flow["fwd_packets"].append(pkt_len)
            flow["fwd_timestamps"].append(timestamp)
            flow["fwd_header_lengths"].append(header_len)
            if tcp_flags & 0x08:  # PSH flag
                flow["fwd_psh_flags"] += 1
            if tcp_flags & 0x20:  # URG flag
                flow["fwd_urg_flags"] += 1
        else:
            flow["bwd_packets"].append(pkt_len)
            flow["bwd_timestamps"].append(timestamp)
            flow["bwd_header_lengths"].append(header_len)

        # Count TCP flags
        if tcp_flags & 0x02:  # SYN
            flow["syn_count"] += 1
        if tcp_flags & 0x10:  # ACK
            flow["ack_count"] += 1

    def emit_flow(self, flow_key: str):
        """
        Compute features for a flow and emit via callback.
        """
        with self._lock:
            if flow_key not in self.flow_table:
                return
            flow = self.flow_table.pop(flow_key)

        features = self._compute_features(flow)
        if self.callback:
            self.callback(features)

    def _compute_features(self, flow: dict) -> dict:
        """
        Compute CICFlowMeter-compatible features from accumulated flow data.
        Returns a dict matching feature_schema.json field names.
        """
        duration = max(flow["last_time"] - flow["start_time"], 1e-6)
        fwd = flow["fwd_packets"]
        bwd = flow["bwd_packets"]
        all_packets = fwd + bwd

        def safe_stats(arr):
            """Compute min, max, mean, std for an array, handling empty."""
            if not arr:
                return 0.0, 0.0, 0.0, 0.0
            a = np.array(arr, dtype=np.float64)
            return float(a.min()), float(a.max()), float(a.mean()), float(a.std())

        def compute_iat(timestamps):
            """Compute inter-arrival time stats."""
            if len(timestamps) < 2:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            ts = np.array(sorted(timestamps))
            iats = np.diff(ts)
            return float(iats.sum()), float(iats.mean()), float(iats.std()), float(iats.max()), float(iats.min())

        fwd_min, fwd_max, fwd_mean, fwd_std = safe_stats(fwd)
        bwd_min, bwd_max, bwd_mean, bwd_std = safe_stats(bwd)
        pkt_min, pkt_max, pkt_mean, pkt_std = safe_stats(all_packets)

        flow_iat_total, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = compute_iat(
            flow["fwd_timestamps"] + flow["bwd_timestamps"]
        )
        fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = compute_iat(flow["fwd_timestamps"])
        bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = compute_iat(flow["bwd_timestamps"])

        total_bytes = sum(all_packets) if all_packets else 0
        total_pkts = len(all_packets)

        return {
            # Metadata (not model features — used for display/routing)
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "protocol": flow["protocol"],
            "start_time": flow["start_time"],
            # Model features (match feature_schema.json)
            "flow_duration": duration * 1e6,  # Convert to microseconds
            "total_fwd_packets": len(fwd),
            "total_bwd_packets": len(bwd),
            "total_length_fwd_packets": sum(fwd) if fwd else 0.0,
            "total_length_bwd_packets": sum(bwd) if bwd else 0.0,
            "fwd_packet_length_max": fwd_max,
            "fwd_packet_length_min": fwd_min,
            "fwd_packet_length_mean": fwd_mean,
            "fwd_packet_length_std": fwd_std,
            "bwd_packet_length_max": bwd_max,
            "bwd_packet_length_min": bwd_min,
            "bwd_packet_length_mean": bwd_mean,
            "bwd_packet_length_std": bwd_std,
            "flow_bytes_per_s": total_bytes / duration if duration > 0 else 0.0,
            "flow_packets_per_s": total_pkts / duration if duration > 0 else 0.0,
            "flow_iat_mean": flow_iat_mean,
            "flow_iat_std": flow_iat_std,
            "flow_iat_max": flow_iat_max,
            "flow_iat_min": flow_iat_min,
            "fwd_iat_total": fwd_iat_total,
            "fwd_iat_mean": fwd_iat_mean,
            "fwd_iat_std": fwd_iat_std,
            "fwd_iat_max": fwd_iat_max,
            "fwd_iat_min": fwd_iat_min,
            "bwd_iat_total": bwd_iat_total,
            "bwd_iat_mean": bwd_iat_mean,
            "bwd_iat_std": bwd_iat_std,
            "bwd_iat_max": bwd_iat_max,
            "bwd_iat_min": bwd_iat_min,
            "fwd_psh_flags": flow["fwd_psh_flags"],
            "fwd_urg_flags": flow["fwd_urg_flags"],
            "fwd_header_length": sum(flow["fwd_header_lengths"]) if flow["fwd_header_lengths"] else 0.0,
            "bwd_header_length": sum(flow["bwd_header_lengths"]) if flow["bwd_header_lengths"] else 0.0,
            "fwd_packets_per_s": len(fwd) / duration if duration > 0 else 0.0,
            "bwd_packets_per_s": len(bwd) / duration if duration > 0 else 0.0,
            "packet_length_min": pkt_min,
            "packet_length_max": pkt_max,
            "packet_length_mean": pkt_mean,
            "packet_length_std": pkt_std,
            "packet_length_variance": pkt_std ** 2,
            "syn_flag_count": flow["syn_count"],
            "ack_flag_count": flow["ack_count"],
        }

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

        # Emit any remaining flows
        remaining_keys = list(self.flow_table.keys())
        for key in remaining_keys:
            self.emit_flow(key)

        logger.info("Flow aggregator stopped. All remaining flows emitted.")
