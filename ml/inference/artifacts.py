"""
SENTINEL-AI artifact resolver for public Hugging Face Hub model repositories.

A public repository can be downloaded without a token. The optional token is
only needed for gated/private repositories. The manifest is verified before
deserializing the pickle preprocessor.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "artifact_manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_manifest(snapshot_dir: str | Path) -> dict:
    """Validate required files and their SHA-256 hashes before model loading."""
    snapshot = Path(snapshot_dir)
    manifest_path = snapshot / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face artifact repository is missing {MANIFEST_NAME}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_file = manifest.get("model_file")
    preprocessor_file = manifest.get("preprocessor_file")
    checksums = manifest.get("sha256")

    if not isinstance(model_file, str) or not model_file:
        raise ValueError("Artifact manifest must define model_file")
    if not isinstance(preprocessor_file, str) or not preprocessor_file:
        raise ValueError("Artifact manifest must define preprocessor_file")
    if not isinstance(checksums, dict):
        raise ValueError("Artifact manifest must define a sha256 object")

    # Do not permit paths outside the downloaded snapshot.
    for filename in (model_file, preprocessor_file):
        if Path(filename).name != filename:
            raise ValueError(f"Artifact filename must be a simple filename: {filename}")
        artifact_path = snapshot / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact file missing: {artifact_path}")
        expected = checksums.get(filename)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing or invalid SHA-256 for {filename}")
        actual = sha256_file(artifact_path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
            )

    if not isinstance(manifest.get("feature_count"), int):
        raise ValueError("Artifact manifest must define integer feature_count")
    if not isinstance(manifest.get("class_order"), list) or not manifest["class_order"]:
        raise ValueError("Artifact manifest must define non-empty class_order")

    return manifest


def resolve_artifact_paths(
    local_model_path: str | Path,
    local_preprocessor_path: str | Path,
    repo_id: str | None = None,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> tuple[Path, Path, dict | None]:
    """
    Resolve artifacts from Hugging Face when repo_id is set, otherwise local disk.

    For production, set revision to a full immutable Hub commit SHA. When no
    repo_id is configured, existing local artifact paths remain supported.
    """
    if not repo_id:
        model_path = Path(local_model_path).expanduser()
        preprocessor_path = Path(local_preprocessor_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}. Set HF_MODEL_REPO_ID "
                "or provide the file locally via MODEL_PATH."
            )
        if not preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {preprocessor_path}. Set "
                "HF_MODEL_REPO_ID or provide it locally via PREPROCESSOR_PATH."
            )
        return model_path, preprocessor_path, None

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=[
            MANIFEST_NAME,
            os.getenv("HF_MODEL_FILENAME", "sentinel_cnn_lstm.h5"),
            os.getenv("HF_PREPROCESSOR_FILENAME", "preprocessor.pkl"),
        ],
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
    ))
    manifest = verify_artifact_manifest(snapshot)

    expected_model = os.getenv("HF_MODEL_FILENAME", "sentinel_cnn_lstm.h5")
    expected_preprocessor = os.getenv("HF_PREPROCESSOR_FILENAME", "preprocessor.pkl")
    if manifest["model_file"] != expected_model:
        raise ValueError(
            f"Manifest model_file={manifest['model_file']!r} does not match "
            f"HF_MODEL_FILENAME={expected_model!r}"
        )
    if manifest["preprocessor_file"] != expected_preprocessor:
        raise ValueError(
            f"Manifest preprocessor_file={manifest['preprocessor_file']!r} "
            f"does not match HF_PREPROCESSOR_FILENAME={expected_preprocessor!r}"
        )

    logger.info(
        "Resolved SENTINEL-AI artifacts from %s@%s (artifact version %s)",
        repo_id,
        revision,
        manifest.get("artifact_version", "unspecified"),
    )
    return snapshot / manifest["model_file"], snapshot / manifest["preprocessor_file"], manifest
