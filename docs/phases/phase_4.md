# Phase 4 — Inference bridge

**Status:** Complete  
**Date:** 7 November 2026

Phase 4 connects a captured flow to the saved CNN-LSTM and prints `{label, confidence, explanation}`. `MITIGATE` is only a decision. Nothing in this phase inserts a firewall rule.

## Run it

From the project root:

```powershell
.\venv\Scripts\Activate.ps1
$env:OPENBLAS_NUM_THREADS = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
python -m ml.inference.service --pcap data/interim/pcaps/syn_scan.pcap
```

`--no-explain` skips SHAP. `--emit` also sends each alert to `http://127.0.0.1:3001` as the Socket.io event `new_alert`. The Node server in `backend/src/server.js` already rebroadcasts that event as `alert`. Without `--emit` the service only prints, so it runs when Node is down. `--iface eth0` is the live capture path and is optional on Windows. Pcap replay does not need Npcap.

The bounded-queue check:

```powershell
python -m ml.inference.load_test --flows 2000
```

That writes `reports/phase4/load_test.json`.

## How the process is split

`ml/inference/service.py` keeps two queues:

| Queue | Cap | Who consumes it |
| --- | ---: | --- |
| Classify | 256 | One worker. Scales the 78 CIC columns and calls `model.predict`. |
| Explain | 2 | One worker. Runs SHAP for `ALERT` and `MITIGATE` only. |

The sniffer callback only enqueues. A full classify queue drops the flow and increments a counter. A full explain queue still emits the alert, with `explanation: null`, so a burst cannot wait on SHAP.

SHAP stays off the capture thread because Phase 3 measured KernelExplainer at about 3.8 seconds per flow. DeepExplainer and GradientExplainer still cannot differentiate this LSTM. `ALLOW` never asks for an explanation. The model lock is shared, so a SHAP call and a predict do not run at the same time inside TensorFlow.

Class names come from `preprocessor.pkl`: `Benign`, `BruteForce`, `DoS/DDoS`, `Infiltration`, `PortScan`, `WebAttack`. The previous hard-coded list assigned those indexes to the wrong names.

`route()` in `ml/inference/pipeline.py`:

- Benign -> `ALLOW`, mitigation `n/a`
- attack below 0.98 -> `ALERT`, mitigation `n/a`
- attack at or above 0.98 -> `MITIGATE`, mitigation `pending`

The threshold is `CONFIDENCE_THRESHOLD` or `inference.confidence_threshold` in `ml/config/model_config.yaml`. Background size for SHAP defaults to 20 (`SHAP_BACKGROUND_SAMPLES`).

## What a replay prints

`data/interim/pcaps/syn_scan.pcap` is eight SYNs with one 5-tuple and no reply. The service emits one alert:

`Benign 0.9235 ALLOW 10.1.0.8:40000 -> 10.1.0.1:22`

The wiring is doing what the model says. This synthetic flow is not a CICFlowMeter row, so the saved network does not call it a port scan. Held-out CIC rows are a different question; those metrics are in `reports/evaluation_v2/evaluation_metrics.json`. A forced SHAP call on this same flow still returns a top-3 `human_readable` sentence (`test_one_suspicious_flow_includes_a_shap_sentence`). On the normal service path that sentence is skipped, because the action is `ALLOW`.

## Load check

2,000 one-packet SYN flows, SHAP off, classify queue capped at 256. Numbers from `reports/phase4/load_test.json`:

| | |
| --- | ---: |
| Emitted | 2000 |
| Classified | 257 |
| Dropped because the queue was full | 1743 |
| Max queue depth | 256 |
| Depth after the producer stopped | 0 |
| Mean classify time | 95.33 ms |
| Median classify time | 92.62 ms |
| 95th percentile | 114.74 ms |
| Wall time | 24.6 s |

Depth never passed 256, and the queue was empty once the producer finished. Those single-packet flows were also scored Benign (about 0.83), so they took the `ALLOW` branch and did not enter the explain queue. The hping3 procedure for an isolated lab VM is already in `docs/lab_setup_guide.md`. This check does not send packets at a host.

## Contract

`docs/architecture/alert_schema.md` freezes the JSON that `new_alert` carries. `ml/config/alert_schema.json` is unchanged. Tests that do not load TensorFlow cover the three router branches, the encoder's class order, and a fixture alert whose missing explanation is null.

## Tests

```powershell
python -m pytest ml/tests/test_decision_router.py ml/tests/test_alert_contract.py ml/tests/test_inference_service.py -q
```

The two tests that load `sentinel_cnn_lstm.h5` skip themselves when that file is absent.
