"""
SENTINEL-AI — Packet Sniffer Module

Scapy-based packet capture from a network interface or an offline pcap.
Passes captured packets to the flow aggregator for feature extraction.
"""

import argparse
import logging
import threading
from pathlib import Path

from scapy.all import IP, TCP, UDP, rdpcap, sniff

logger = logging.getLogger(__name__)

# TCP flag bits, in the order they are printed.
TCP_FLAG_BITS = (
    ("FIN", 0x01),
    ("SYN", 0x02),
    ("RST", 0x04),
    ("PSH", 0x08),
    ("ACK", 0x10),
    ("URG", 0x20),
    ("ECE", 0x40),
    ("CWR", 0x80),
)


class PacketSniffer:
    """
    Captures packets from a network interface or replays a pcap using Scapy.

    Extracts the 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol),
    TCP flags, and the TCP window, then passes each packet to a callback.

    Usage:
        sniffer = PacketSniffer(interface="eth0", callback=my_callback)
        sniffer.start()
        sniffer.stop()

        sniffer.sniff_pcap("capture.pcap")
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
        info = packet_to_info(packet)
        if info is None:
            return

        self._packet_count += 1
        if self.callback:
            self.callback(info)

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

    def sniff_pcap(self, path: str):
        """
        Replay an offline pcap through the same parser used for live capture.

        Does not open a network interface, so it runs on Windows without a
        capture driver.
        """
        pcap_path = Path(path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"pcap not found: {pcap_path}")

        logger.info(f"Replaying pcap {pcap_path}")
        for packet in rdpcap(str(pcap_path)):
            self._process_packet(packet)
        logger.info(f"Pcap replay finished. Packets parsed: {self._packet_count}")

    def stop(self):
        """Stop packet capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Sniffer stopped. Total packets captured: {self._packet_count}")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def packet_count(self) -> int:
        return self._packet_count


def packet_to_info(packet) -> dict | None:
    """
    Convert one Scapy packet into the dict the flow aggregator consumes.

    Returns None when the packet is not IPv4.
    """
    if not packet.haslayer(IP):
        return None

    ip_header_len = int(packet[IP].ihl) * 4
    info = {
        "timestamp": float(packet.time),
        "src_ip": packet[IP].src,
        "dst_ip": packet[IP].dst,
        "protocol": int(packet[IP].proto),
        "ip_len": int(packet[IP].len),
        "ttl": int(packet[IP].ttl),
        "src_port": 0,
        "dst_port": 0,
        "tcp_flags": 0,
        "window": 0,
        "payload_len": 0,
        "header_len": ip_header_len,
        "fin": 0,
        "syn": 0,
        "rst": 0,
        "psh": 0,
        "ack": 0,
        "urg": 0,
        "ece": 0,
        "cwr": 0,
    }

    if packet.haslayer(TCP):
        tcp = packet[TCP]
        flags = int(tcp.flags)
        tcp_header_len = int(tcp.dataofs) * 4
        info.update({
            "src_port": int(tcp.sport),
            "dst_port": int(tcp.dport),
            "tcp_flags": flags,
            "window": int(tcp.window),
            "payload_len": len(bytes(tcp.payload)),
            "header_len": ip_header_len + tcp_header_len,
            "protocol_name": "TCP",
        })
        for name, bit in TCP_FLAG_BITS:
            info[name.lower()] = int(bool(flags & bit))
    elif packet.haslayer(UDP):
        udp = packet[UDP]
        info.update({
            "src_port": int(udp.sport),
            "dst_port": int(udp.dport),
            "payload_len": len(bytes(udp.payload)),
            "header_len": ip_header_len + 8,
            "protocol_name": "UDP",
        })
    else:
        info["protocol_name"] = "OTHER"
        info["payload_len"] = max(int(packet[IP].len) - ip_header_len, 0)

    return info


def format_packet_line(info: dict) -> str:
    """Format one packet as ``src:port -> dst:port PROTO flags=...``."""
    raised = [name for name, bit in TCP_FLAG_BITS if info.get("tcp_flags", 0) & bit]
    flag_text = ",".join(raised) if raised else "-"
    return (
        f"{info['src_ip']}:{info['src_port']} -> {info['dst_ip']}:{info['dst_port']} "
        f"{info.get('protocol_name', 'OTHER')} flags={flag_text} window={info.get('window', 0)}"
    )


def _print_packet(info: dict):
    print(format_packet_line(info))


def main():
    """Print the 5-tuple and TCP flags for a pcap, or a short live capture."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Print 5-tuple and TCP flags from a pcap or interface")
    parser.add_argument("--pcap", help="Offline pcap to replay")
    parser.add_argument("--iface", default="eth0", help="Interface for a live capture")
    parser.add_argument("--count", type=int, default=20, help="Live packets to print")
    args = parser.parse_args()

    sniffer = PacketSniffer(interface=args.iface, callback=_print_packet)
    if args.pcap:
        sniffer.sniff_pcap(args.pcap)
        return

    logger.info(f"Live capture on {args.iface} for {args.count} packets")
    sniff(
        iface=args.iface,
        prn=sniffer._process_packet,
        filter=sniffer.bpf_filter,
        store=False,
        count=args.count,
    )


if __name__ == "__main__":
    main()
