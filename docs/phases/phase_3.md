# Phase 3 — XAI Integration and Live Ingestion

**Status:** Complete  
**Date:** 6 November 2026

Phase 3 produces a SHAP explanation for a scored flow and turns packets into the same 78-feature vector the saved CNN-LSTM expects. It does not wire capture into `model.predict`. That connection is Phase 4.

## What was built

| Piece | Location | What it does |
| --- | --- | --- |
| Explainer | `ml/explainability/shap_explainer.py` | Ranks features for the class `model.predict` chose and returns a top-3 sentence |
| Batch runner | `ml/explainability/run_shap_batch.py` | Explains 10 held-out test rows and writes the latency report |
| Latency report | `reports/phase3/shap_latency.json` | Init time, per-flow time, explainer name, and the 10 explanations |
| Sniffer | `ml/ingestion/packet_sniffer.py` | Prints the 5-tuple, TCP flags, and window from a pcap or a live interface |
| Flow table | `ml/ingestion/flow_aggregator.py` | Emits the 78 CIC-IDS columns, plus the 42 schema names as aliases |
| Tests | `ml/tests/test_flow_aggregator.py` | Replays a synthetic benign exchange and a SYN scan |

## Run the SHAP batch (the latency job)

From the project root, with the project virtualenv active:

```powershell
.\venv\Scripts\Activate.ps1
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
python ml/explainability/run_shap_batch.py
```

`OPENBLAS_NUM_THREADS=1` matters on this machine. The first attempt died during import with `OpenBLAS error: Memory allocation still failed` while several other Python processes were still resident. Pinning OpenBLAS to one thread let the import succeed.

The script loads:

- `ml/models/artifacts/sentinel_cnn_lstm.h5`
- `ml/models/artifacts/preprocessor.pkl` (class names only)
- `data/processed/X_test.npy` and `y_test.npy` (memory-mapped; rows are already scaled)
- `data/processed/feature_names.json`

It explains 10 stratified test rows, prints each sentence, and overwrites `reports/phase3/shap_latency.json`. Expect a few minutes. KernelExplainer evaluates the network once per coalition, and the run log is verbose because SHAP logs every coalition weight.

To print a capture without scoring it:

```powershell
python -m ml.ingestion.packet_sniffer --pcap data/interim/pcaps/benign_http.pcap
```

Live capture is optional and needs a working interface plus Npcap:

```powershell
python -m ml.ingestion.packet_sniffer --iface eth0 --count 20
```

Aggregator tests, which also rebuild those two pcaps:

```powershell
python -m pytest ml/tests/test_flow_aggregator.py -q
```

## Explainer that actually ran

`SHAPExplainer` tries three backends in order:

1. `shap.DeepExplainer` — constructed, then failed. TensorFlow 2 has no gradient for SHAP's `shap_TensorListStack` op, which this LSTM hits because `recurrent_dropout` is set in `ml/config/model_config.yaml`.
2. `shap.GradientExplainer` — failed the same way (`shap_StridedSlice` is missing from the gradient registry).
3. `shap.KernelExplainer` on `model.predict` — this is the backend that produced the report. Coalition count is capped at `min(200, 2 * n_features + 1)` so a 78-feature call stays bounded.

Each explanation is for the class `argmax(model.predict)` returned, not the class with the largest summed SHAP magnitude. The sentence format is `Feature Name: +400% vs baseline`, where the percent is the scaled feature versus the mean of the background rows.

## Latency

Measured on the CPU (TensorFlow has no GPU build on native Windows). Ten flows, one SHAP call for the batch. Per-flow time is batch time divided by 10.

| Background rows | Init | Per flow | Within 500 ms budget |
| --- | ---: | ---: | --- |
| 100 | 6.6 s | 18035 ms | No |
| 20 | 2.3 s | 3814 ms | No |

The report keeps the smaller background. `reports/phase3/shap_latency.json` records `explainer = KernelExplainer`, `background_size = 20`, `background_reduced = true`, `within_budget = false`, `per_flow_ms = 3814.04`.

That number is the dashboard budget for Phase 5: a SHAP sentence cannot be computed inline on every flow if the UI has to stay near one second. Phase 4 should explain alerts (non-benign, or low-confidence benign), not every benign flow, and it should not block packet capture on the KernelExplainer call.

The percent-versus-baseline figures are large because the background mean of a StandardScaler feature sits near zero, so the denominator in the percent formula is tiny. The ranking itself uses the absolute SHAP value. Read the percent as a relative deviation from the background, not as a raw packet-count change.

## Ten sample explanations

Class order is the encoder stored in `preprocessor.pkl`: `0=Benign, 1=BruteForce, 2=DoS/DDoS, 3=Infiltration, 4=PortScan, 5=WebAttack`. That is not the order hard-coded in `ml/inference/pipeline.py`. Phase 4 must read the encoder.

| True label | Predicted | Confidence | Top-3 sentence |
| --- | --- | ---: | --- |
| Benign | Benign | 0.978 | Init_Win_bytes_backward: +26045% vs baseline; Down/Up Ratio: -317% vs baseline; PSH Flag Count: +586% vs baseline |
| BruteForce | BruteForce | 0.999 | PSH Flag Count: +586% vs baseline; Init_Win_bytes_forward: +1979% vs baseline; Fwd IAT Std: -73% vs baseline |
| DoS/DDoS | DoS/DDoS | 0.993 | Init_Win_bytes_forward: +1979% vs baseline; Average Packet Size: +2060% vs baseline; Avg Bwd Segment Size: +1775% vs baseline |
| Infiltration | Infiltration | 0.998 | Init_Win_bytes_forward: +13% vs baseline; Init_Win_bytes_backward: -605% vs baseline; Packet Length Std: -104% vs baseline |
| PortScan | PortScan | 0.997 | PSH Flag Count: +586% vs baseline; Init_Win_bytes_forward: +1979% vs baseline; ACK Flag Count: -381% vs baseline |
| WebAttack | WebAttack | 0.954 | Init_Win_bytes_backward: -605% vs baseline; Subflow Fwd Packets: +43999% vs baseline; Packet Length Std: +2083% vs baseline |
| PortScan | PortScan | 0.997 | PSH Flag Count: +586% vs baseline; URG Flag Count: -272% vs baseline; Average Packet Size: -226% vs baseline |
| Benign | BruteForce | 0.933 | Min Packet Length: -408% vs baseline; Init_Win_bytes_forward: -725% vs baseline; ACK Flag Count: +1143% vs baseline |
| DoS/DDoS | DoS/DDoS | 0.915 | Down/Up Ratio: -317% vs baseline; ACK Flag Count: +1143% vs baseline; PSH Flag Count: -391% vs baseline |
| PortScan | PortScan | 0.997 | Bwd Packet Length Mean: -209% vs baseline; PSH Flag Count: +586% vs baseline; Bwd Packet Length Min: -1470% vs baseline |

Nine of the ten rows match the held-out label. One Benign row is scored as BruteForce at 0.93, which is consistent with the Week 2 evaluation (BruteForce precision was about 0.15). Full vectors and SHAP numbers are in the JSON file.

## Feature contract for Phase 4

The saved model was trained on **78 original CIC-IDS column names** in `data/processed/feature_names.json` (for example `SYN Flag Count`, `Flow Duration`). `ml/config/feature_schema.json` lists **42 snake_case names**. Preprocessing fell back to every numeric column, so the schema was not the training contract. Do not retrain to close that gap.

`FlowAggregator` emits both:

- The 78 names, in `feature_names.json` order. `to_feature_vector()` returns that `float64` vector. Phase 4 must scale it with `preprocessor.pkl` before `model.predict`. The SHAP runner skips that step because `X_test.npy` is already scaled.
- The 42 schema names as aliases of the overlapping CIC columns (`syn_flag_count` is `SYN Flag Count`).

Time features (flow duration, inter-arrival times, active time, idle time) are microseconds, matching CIC-IDS2017. Packet and byte rates stay in per-second units. A zero-duration flow gets a rate of 0 so the vector does not contain infinities.

Columns a packet capture does not reconstruct are filled with 0: `Fwd/Bwd Avg Bytes/Bulk`, `Fwd/Bwd Avg Packets/Bulk`, and `Fwd/Bwd Avg Bulk Rate`. Those columns are zero on most CIC-IDS2017 rows. Subflow packet and byte counts copy the forward and backward totals, which is what CICFlowMeter stores when a flow has a single subflow. Active versus idle uses a 5-second gap, the CICFlowMeter active timeout.

There are no CIC raw pcaps in `data/raw` (only the flow CSVs). The unit test writes two Scapy pcaps under `data/interim/pcaps/` (gitignored):

- `benign_http.pcap` — one bidirectional TCP exchange. The emitted flow has packets in both directions, a non-zero ACK count, and a duration of 50,000 microseconds.
- `syn_scan.pcap` — eight SYNs with the same 5-tuple and no reply. They collapse to one flow with `SYN Flag Count == 8`, `ACK Flag Count == 0`, and no backward packets.

Both vectors have length 78 and the same column order as `feature_names.json`.
