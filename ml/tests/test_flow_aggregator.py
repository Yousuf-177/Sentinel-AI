"""
SENTINEL-AI — Flow aggregator tests against synthetic pcaps.

Builds a short benign TCP exchange and a SYN scan, replays both through
PacketSniffer, and checks the emitted CIC feature vectors.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PCAP_DIR = PROJECT_ROOT / "data" / "interim" / "pcaps"
FEATURE_NAMES_PATH = PROJECT_ROOT / "data" / "processed" / "feature_names.json"


def _stamp(packet, when: float):
    packet.time = when
    return packet


def build_pcaps(directory: Path) -> tuple[Path, Path]:
    """Write a benign handshake and an 8-packet SYN scan. Returns both paths."""
    from scapy.all import IP, Raw, TCP, wrpcap

    directory.mkdir(parents=True, exist_ok=True)
    benign_path = directory / "benign_http.pcap"
    scan_path = directory / "syn_scan.pcap"

    base = 1_700_000_000.0
    benign = [
        _stamp(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=12345, dport=80, flags="S", seq=1, window=65535), base),
        _stamp(IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=12345, flags="SA", seq=100, ack=2, window=29200), base + 0.01),
        _stamp(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=12345, dport=80, flags="A", seq=2, ack=101, window=65535), base + 0.02),
        _stamp(
            IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=12345, dport=80, flags="PA", seq=2, ack=101, window=65535)
            / Raw(load=b"GET / HTTP/1.1\r\n"),
            base + 0.03,
        ),
        _stamp(
            IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=12345, flags="PA", seq=101, ack=18, window=29200)
            / Raw(load=b"HTTP/1.1 200 OK\r\n"),
            base + 0.05,
        ),
    ]
    scan = []
    for i in range(8):
        scan.append(
            _stamp(
                IP(src="10.1.0.8", dst="10.1.0.1") / TCP(sport=40000, dport=22, flags="S", seq=1000 + i, window=1024),
                base + i * 0.001,
            )
        )
    wrpcap(str(benign_path), benign)
    wrpcap(str(scan_path), scan)
    return benign_path, scan_path


def _replay(pcap_path: Path) -> list:
    from ml.ingestion.flow_aggregator import FlowAggregator
    from ml.ingestion.packet_sniffer import PacketSniffer

    emitted = []
    aggregator = FlowAggregator(timeout=120.0, callback=emitted.append)
    sniffer = PacketSniffer(callback=aggregator.add_packet)
    sniffer.sniff_pcap(str(pcap_path))
    aggregator.stop()
    return emitted


@pytest.fixture(scope="module")
def pcaps():
    return build_pcaps(PCAP_DIR)


def test_sniffer_prints_five_tuple(pcaps):
    from ml.ingestion.packet_sniffer import PacketSniffer, format_packet_line

    lines = []
    sniffer = PacketSniffer(callback=lambda info: lines.append(format_packet_line(info)))
    sniffer.sniff_pcap(str(pcaps[0]))

    assert sniffer.packet_count == 5
    assert lines[0].startswith("10.0.0.1:12345 -> 10.0.0.2:80 TCP flags=SYN")
    assert "window=65535" in lines[0]
    assert "flags=SYN,ACK" in lines[1]


def test_benign_flow_is_bidirectional(pcaps):
    flows = _replay(pcaps[0])
    assert len(flows) == 1
    flow = flows[0]
    assert flow["Total Fwd Packets"] >= 1
    assert flow["Total Backward Packets"] >= 1
    assert flow["ACK Flag Count"] >= 1
    assert flow["src_ip"] == "10.0.0.1"
    assert flow["dst_port"] == 80
    # 0.05 seconds of the crafted exchange, stored as microseconds.
    assert flow["Flow Duration"] == pytest.approx(50_000.0, rel=0.01)


def test_syn_scan_collapses_to_one_flow(pcaps):
    flows = _replay(pcaps[1])
    assert len(flows) == 1
    flow = flows[0]
    assert flow["SYN Flag Count"] == 8
    assert flow["ACK Flag Count"] == 0
    assert flow["Total Backward Packets"] == 0
    assert flow["syn_flag_count"] == flow["SYN Flag Count"]
    assert flow["Init_Win_bytes_forward"] == 1024


def test_vectors_match_trained_feature_order(pcaps):
    with open(FEATURE_NAMES_PATH, encoding="utf-8") as handle:
        expected = json.load(handle)

    from ml.ingestion.flow_aggregator import FlowAggregator

    for pcap in pcaps:
        flows = _replay(pcap)
        aggregator = FlowAggregator()
        vector = aggregator.to_feature_vector(flows[0])
        assert list(aggregator.feature_names) == expected
        assert vector.shape == (78,)
        assert len(expected) == 78
        for name in expected:
            assert name in flows[0]
