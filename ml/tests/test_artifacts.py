"""Unit tests for Hugging Face artifact manifest validation."""

import hashlib
import json

import pytest

from ml.inference.artifacts import verify_artifact_manifest


def _write_artifacts(folder):
    model = folder / "sentinel_cnn_lstm.h5"
    preprocessor = folder / "preprocessor.pkl"
    model.write_bytes(b"fake-keras-model")
    preprocessor.write_bytes(b"fake-preprocessor")
    manifest = {
        "artifact_version": "test",
        "model_file": model.name,
        "preprocessor_file": preprocessor.name,
        "feature_count": 78,
        "class_order": ["Benign", "BruteForce", "DoS/DDoS", "Infiltration", "PortScan", "WebAttack"],
        "sha256": {
            model.name: hashlib.sha256(model.read_bytes()).hexdigest(),
            preprocessor.name: hashlib.sha256(preprocessor.read_bytes()).hexdigest(),
        },
    }
    (folder / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_manifest_validates_matching_hashes(tmp_path):
    expected = _write_artifacts(tmp_path)

    actual = verify_artifact_manifest(tmp_path)

    assert actual["artifact_version"] == expected["artifact_version"]
    assert actual["feature_count"] == 78


def test_manifest_rejects_modified_model(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "sentinel_cnn_lstm.h5").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifact_manifest(tmp_path)


def test_manifest_rejects_path_traversal(tmp_path):
    manifest = _write_artifacts(tmp_path)
    manifest["model_file"] = "../outside.h5"
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="simple filename"):
        verify_artifact_manifest(tmp_path)


def test_manifest_requires_both_files(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "preprocessor.pkl").unlink()

    with pytest.raises(FileNotFoundError, match="preprocessor.pkl"):
        verify_artifact_manifest(tmp_path)
