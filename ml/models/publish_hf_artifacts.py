"""
Publish a matched SENTINEL-AI Keras model and fitted preprocessor to Hugging Face.

Example:
  HF_TOKEN=hf_... python -m ml.models.publish_hf_artifacts \
      --repo-id your-hf-username/sentinel-ai-models \
      --model ml/models/artifacts/sentinel_cnn_lstm.h5 \
      --preprocessor ml/models/artifacts/preprocessor.pkl

The repository is created as PUBLIC. Never upload credentials, raw datasets,
or unrelated training files. Only upload trusted pickle files: pickle loading
can execute code.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from huggingface_hub import HfApi, create_repo, upload_folder

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Publish SENTINEL-AI model artifacts")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo, e.g. user/sentinel-ai-models")
    parser.add_argument("--model", default="ml/models/artifacts/sentinel_cnn_lstm.h5")
    parser.add_argument("--preprocessor", default="ml/models/artifacts/preprocessor.pkl")
    parser.add_argument("--model-version", default="1.0.0")
    parser.add_argument("--private", action="store_true", help="Create a private repo (default is public)")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        parser.error("Set HF_TOKEN to a Hugging Face WRITE token before uploading.")

    model_path = Path(args.model).expanduser()
    preprocessor_path = Path(args.preprocessor).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not preprocessor_path.is_absolute():
        preprocessor_path = PROJECT_ROOT / preprocessor_path

    for path in (model_path, preprocessor_path):
        if not path.is_file():
            parser.error(f"Required artifact does not exist: {path}")

    # Inspect the fitted preprocessor before publishing it.
    artifact = joblib.load(preprocessor_path)
    required = {"scaler", "label_encoder", "feature_names"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        parser.error(
            "preprocessor.pkl must contain scaler, label_encoder, and feature_names"
        )
    feature_names = list(artifact["feature_names"])
    class_order = [str(value) for value in artifact["label_encoder"].classes_]
    if len(feature_names) != 78:
        parser.error(f"Expected 78 model features; found {len(feature_names)}")
    if len(class_order) != 6:
        parser.error(f"Expected 6 encoded classes; found {len(class_order)}")

    model_name = "sentinel_cnn_lstm.h5"
    preprocessor_name = "preprocessor.pkl"
    manifest = {
        "artifact_version": args.model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_file": model_name,
        "preprocessor_file": preprocessor_name,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "class_order": class_order,
        "sha256": {
            model_name: sha256_file(model_path),
            preprocessor_name: sha256_file(preprocessor_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    with tempfile.TemporaryDirectory(prefix="sentinel-hf-upload-") as tmp:
        staging = Path(tmp)
        shutil.copy2(model_path, staging / model_name)
        shutil.copy2(preprocessor_path, staging / preprocessor_name)
        (staging / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# SENTINEL-AI model artifacts\n\n"
            "Public release of the SENTINEL-AI CNN-LSTM IDS model and its "
            "matching fitted preprocessing artifact.\n\n"
            f"- Artifact version: {args.model_version}\n"
            f"- Features: {len(feature_names)}\n"
            f"- Classes: {', '.join(class_order)}\n"
            "- Verify artifact_manifest.json before loading files.\n"
            "- The preprocessor is a Python pickle and must only be loaded "
            "from a trusted repository.\n",
            encoding="utf-8",
        )
        info = upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(staging),
            token=token,
            commit_message=f"Publish SENTINEL-AI artifacts v{args.model_version}",
        )

    print(f"Published public model repository: https://huggingface.co/{args.repo_id}")
    print(f"Hub commit: {info.oid}")
    print("Pin this full commit SHA in HF_MODEL_REVISION for reproducible deployment.")
    print("Manifest SHA-256 checksums were generated from the uploaded files.")


if __name__ == "__main__":
    main()
