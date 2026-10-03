"""
SENTINEL-AI — Packet Sniffer Module

Scapy-based packet capture from a network interface.
Passes captured packets to the flow aggregator for feature extraction.
"""

import logging
import threading
from collections import defaultdict

from scapy.all import sniff, IP, TCP, UDP

logger = logging.getLogger(__name__)


class PacketSniffer:
    """
    Captures packets from a network interface using Scapy.
    
    Extracts the 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol)
    and TCP flags from each packet, then passes to a callback function.
    
    Usage:
        sniffer = PacketSniffer(interface="eth0", callback=my_callback)
        sniffer.start()   # Starts capture in a background thread
        sniffer.stop()    # Stops capture
    """

    def __init__(self, interface: str = "eth0", callback=None, bpf_filter: str = "ip"):
        """
        Args:
            interface: Network interface to capture on.
            callback: Function called with (packet_info: dict) for each captured packet.
            bpf_filter: BPF filter string (default: capture only IP packets).
        """
        self.interface = interface
        self.callback = callback
        self.bpf_filter = bpf_filter
        self._running = False
        self._thread = None
        self._packet_count = 0

    def _process_packet(self, packet):
        """Extract relevant fields from a captured packet."""
        if not packet.haslayer(IP):
            return

        self._packet_count += 1

        packet_info = {
            "timestamp": float(packet.time),
            "src_ip": packet[IP].src,
            "dst_ip": packet[IP].dst,
            "protocol": packet[IP].proto,
            "ip_len": packet[IP].len,
            "ttl": packet[IP].ttl,
            "src_port": 0,
            "dst_port": 0,
            "tcp_flags": 0,
            "payload_len": len(packet[IP].payload),
            "header_len": packet[IP].ihl * 4,
        }

        if packet.haslayer(TCP):
            packet_info["src_port"] = packet[TCP].sport
            packet_info["dst_port"] = packet[TCP].dport
            packet_info["tcp_flags"] = int(packet[TCP].flags)
            packet_info["protocol_name"] = "TCP"
        elif packet.haslayer(UDP):
            packet_info["src_port"] = packet[UDP].sport
            packet_info["dst_port"] = packet[UDP].dport
            packet_info["protocol_name"] = "UDP"
        else:
            packet_info["protocol_name"] = "OTHER"

        if self.callback:
            self.callback(packet_info)

    def start(self):
        """Start packet capture in a background thread."""
        if self._running:
            logger.warning("Sniffer is already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Packet sniffer started on interface '{self.interface}'")

    def _capture_loop(self):
        """Main capture loop."""
        try:
            sniff(
                iface=self.interface,
                prn=self._process_packet,
                filter=self.bpf_filter,
                store=False,
                stop_filter=lambda _: not self._running,
            )
        except Exception as e:
            logger.error(f"Capture error: {e}")
            self._running = False

    def stop(self):
        """Stop packet capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Sniffer stopped. Total packets captured: {self._packet_count}")

    @property
    def is_running(self) -> bool:
        return self._running
