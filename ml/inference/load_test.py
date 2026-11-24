"""
SENTINEL-AI — Bounded classify-queue check

Pushes several thousand single-packet SYN flows through the same queue the
live service uses. SHAP stays off. This is the in-process stand-in for a
lab flood: the queue is capped, drops are counted, and depth returns to
zero after the producer stops.

    python -m ml.inference.load_test
    python -m ml.inference.load_test --flows 2000
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from ml.inference.service import CLASSIFY_QUEUE_MAX, InferenceService, load_runtime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase4" / "load_test.json"


def syn_packet(index: int, base_time: float = 1_700_000_000.0) -> dict:
    """One SYN packet. Each index is its own flow because the destination port differs."""
    return {
        "timestamp": base_time + index * 1e-4,
        "src_ip": "10.9.0.8",
        "dst_ip": "10.9.0.1",
        "src_port": 40000,
        "dst_port": 1 + (index % 65535),
        "protocol": 6,
        "protocol_name": "TCP",
        "ip_len": 40,
        "header_len": 40,
        "payload_len": 0,
        "tcp_flags": 0x02,
        "syn": 1,
        "window": 1024,
    }


def run_load(service: InferenceService, n_flows: int) -> dict:
    """Emit ``n_flows`` through the aggregator into ``service`` and drain it."""
    from ml.ingestion.flow_aggregator import FlowAggregator

    service.start()
    emitted = {"count": 0}

    def enqueue(flow):
        emitted["count"] += 1
        service.submit_flow(flow)

    aggregator = FlowAggregator(timeout=120.0, callback=enqueue)
    started = time.perf_counter()
    for index in range(n_flows):
        aggregator.add_packet(syn_packet(index))
    aggregator.stop()
    service.drain()
    elapsed = time.perf_counter() - started

    latencies = np.asarray(service.classify_latencies_ms, dtype=np.float64)
    summary = {
        "flows_emitted": int(emitted["count"]),
        "flows_classified": int(service.classified),
        "flows_dropped": int(service.dropped),
        "max_queue_depth": int(service.max_depth),
        "final_queue_depth": int(service.classify_queue.qsize()),
        "queue_cap": CLASSIFY_QUEUE_MAX,
        "shap_enabled": False,
        "elapsed_seconds": round(elapsed, 3),
        "classify_latency_ms": {
            "mean": round(float(latencies.mean()), 2) if latencies.size else None,
            "p50": round(float(np.percentile(latencies, 50)), 2) if latencies.size else None,
            "p95": round(float(np.percentile(latencies, 95)), 2) if latencies.size else None,
        },
    }
    if summary["flows_classified"] + summary["flows_dropped"] != summary["flows_emitted"]:
        raise RuntimeError(
            f"Accounting mismatch: classified {summary['flows_classified']} + "
            f"dropped {summary['flows_dropped']} != emitted {summary['flows_emitted']}"
        )
    if summary["max_queue_depth"] > CLASSIFY_QUEUE_MAX:
        raise RuntimeError(f"Queue depth {summary['max_queue_depth']} exceeded {CLASSIFY_QUEUE_MAX}")
    if summary["final_queue_depth"] != 0:
        raise RuntimeError(f"Queue still holds {summary['final_queue_depth']} flows after drain")
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Bounded classify-queue load check")
    parser.add_argument("--flows", type=int, default=2000)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    pipeline = load_runtime(explain=False)
    service = InferenceService(pipeline, explain=False)
    logger.info(f"Classifying {args.flows} SYN flows with queue cap {CLASSIFY_QUEUE_MAX}")
    summary = run_load(service, args.flows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(
        f"classified={summary['flows_classified']} dropped={summary['flows_dropped']} "
        f"max_depth={summary['max_queue_depth']} mean_ms={summary['classify_latency_ms']['mean']}"
    )
    logger.info(f"Wrote {output}")


if __name__ == "__main__":
    main()
