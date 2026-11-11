# Phase 4 — Live IDS runtime

## What this phase wires together

`PacketSniffer -> FlowAggregator -> InferencePipeline -> Socket.IO backend`

The service supports live IPv4 capture and offline PCAP replay. It uses the exact
feature order and class order stored in the fitted preprocessing artifact.

## Required artifacts

The repository intentionally excludes datasets and trained binary artifacts.
Before starting the service, provide:

- `ml/models/artifacts/sentinel_cnn_lstm.h5` (or set `MODEL_PATH`)
- `ml/models/artifacts/preprocessor.pkl` (or set `PREPROCESSOR_PATH`)

Build the preprocessor from the same CIC-IDS2017 data and split used to train
the committed model:

```bash
python -m ml.models.build_preprocessor
```

The preprocessor artifact must contain `scaler`, `label_encoder`, and
`feature_names`. The runtime validates that there are 78 features and that
the model input/output dimensions match the saved artifact. Do not pair a model
with a preprocessor from a different training run.

## Run

Install Python dependencies:

```bash
python -m pip install -r ml/requirements.txt
```

Start the Node backend, then run a PCAP replay (no capture privileges required):

```bash
python -m ml.inference.service --pcap data/interim/pcaps/benign_http.pcap
```

To print alerts to stdout without a backend:

```bash
python -m ml.inference.service --pcap path/to/capture.pcap --no-backend
```

For live capture on Linux (usually requires root or appropriate packet-capture
capabilities):

```bash
sudo CAPTURE_INTERFACE=eth0 python -m ml.inference.service
```

Set `SENTINEL_BACKEND_URL` to the backend URL reachable from the inference
process. The default is `http://127.0.0.1:3001`.

## Safety and current limitations

- This is detection-only. Non-benign predictions emit `ALERT`; benign
  predictions emit `ALLOW`. No firewall rule is inserted or modified.
- SHAP is not run synchronously for every flow. The measured KernelExplainer
  latency is unsuitable for the packet-capture callback.
- The trained model and fitted preprocessing artifact are not present in the
  Git repository because binary artifacts and datasets are gitignored. The
  runtime cannot produce real predictions until matching artifacts are supplied.
- The current flow aggregator is a CICFlowMeter-like approximation. Live
  feature parity with the exact training pipeline still needs evaluation on
  representative PCAPs; passing shape checks does not establish model quality.
- Validate in an isolated lab. Do not capture third-party traffic without
  authorization or deploy automatic blocking based on these model metrics.

## Validation

Run the focused tests:

```bash
pytest ml/tests/test_inference_pipeline.py ml/tests/test_flow_aggregator.py -q
```

Live interface capture, backend reachability, and model/feature parity require
the target Linux environment and the matching model artifacts.
