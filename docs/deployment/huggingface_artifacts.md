# Publishing and loading SENTINEL-AI model artifacts with Hugging Face

This repository keeps large trained binaries out of Git. The model and its
matching fitted preprocessing artifact are published to a **public Hugging Face
model repository** and downloaded by the inference service at startup.

## 1. Create a public model repository

1. Sign in at https://huggingface.co/new
2. Choose a model repository name, for example `sentinel-ai-models`.
3. Set visibility to **Public**.
4. Keep the repository ID (format: `your-username/sentinel-ai-models`).

Public downloads do not need a Hugging Face token. Uploading does.

## 2. Create a write token for publishing

Create a Hugging Face access token with write permission at
https://huggingface.co/settings/tokens. Keep it private. Do not paste it into
GitHub, source code, Dockerfiles, issue comments, or commit it in `.env`.

In a local terminal, set it only for the upload command:

Linux/macOS:
```bash
export HF_TOKEN="hf_your_write_token"
```

PowerShell:
```powershell
$env:HF_TOKEN = "hf_your_write_token"
```

## 3. Prepare the matched artifacts

The publisher expects the trained Keras model and the fitted preprocessor
created from the **same training run**:

- `ml/models/artifacts/sentinel_cnn_lstm.h5`
- `ml/models/artifacts/preprocessor.pkl`

The preprocessor must contain `scaler`, `label_encoder`, and
`feature_names`. The publisher checks for 78 features and six classes,
extracts the exact encoder class order, and writes a manifest with SHA-256
checksums.

Do not rebuild the preprocessor using a different split, dataset revision, or
training run and pair it with an older model. The model and preprocessor must
remain a matched pair.

## 4. Upload the artifacts

From the repository root:

```bash
python -m pip install -r ml/requirements.txt
python -m ml.models.publish_hf_artifacts \
  --repo-id your-username/sentinel-ai-models \
  --model ml/models/artifacts/sentinel_cnn_lstm.h5 \
  --preprocessor ml/models/artifacts/preprocessor.pkl \
  --model-version 1.0.0
```

The script creates/updates the repository as public by default and uploads only
the model, preprocessor, manifest, and model card. It prints the full Hub commit
SHA at the end. Record that SHA for reproducible deployments.

To deliberately create a private repository instead, pass `--private`; this
is not the selected deployment configuration for SENTINEL-AI.

## 5. Configure runtime downloads

Set these variables in your deployment environment (not in committed secrets):

```dotenv
HF_MODEL_REPO_ID=your-username/sentinel-ai-models
HF_MODEL_REVISION=<full-hugging-face-commit-sha>
HF_MODEL_FILENAME=sentinel_cnn_lstm.h5
HF_PREPROCESSOR_FILENAME=preprocessor.pkl
HF_HUB_CACHE_DIR=/app/.cache/huggingface
```

Use the exact full commit SHA printed by the publisher for `HF_MODEL_REVISION`.
Using `main` is convenient for development but means a later Hub upload can
change what a new deployment downloads.

No `HF_TOKEN` is needed for this public model repository. For gated/private
repositories only, inject a read token using your hosting provider's secret
manager. Never bake the token into an image.

The inference service now:
1. Downloads only the manifest, model, and preprocessor files.
2. Verifies the manifest and both SHA-256 hashes.
3. Checks manifest feature names and class order against the deserialized
   preprocessor.
4. Checks the model input/output dimensions before starting packet capture.

The Hub client caches downloads. Set `HF_HUB_CACHE_DIR` to a persistent cache
volume in Docker/production if you want to avoid downloading artifacts on every
container replacement.

## 6. Run the IDS

```bash
python -m ml.inference.service --pcap path/to/capture.pcap --no-backend
```

For live capture, run on an authorized Linux lab interface with the required
packet-capture privileges:

```bash
sudo python -m ml.inference.service --iface eth0
```

## Security and reproducibility

- Public means anyone can download the model and preprocessor. Do not include
  training data, private captures, tokens, or personally identifiable data.
- A SHA-256 manifest detects accidental or uncoordinated artifact changes; it
  does not establish who published the files. Pin a trusted Hub commit SHA.
- `preprocessor.pkl` is a Python pickle. Loading pickle can execute arbitrary
  code. Only load it from the trusted SENTINEL-AI repository and do not accept
  arbitrary user-supplied pickle files.
- Preserve the model/preprocessor pair and record its version/Hub commit in
  deployment records.
- This artifact flow does not validate live feature parity or model accuracy.
  Those require separate evaluation on representative PCAP data.
