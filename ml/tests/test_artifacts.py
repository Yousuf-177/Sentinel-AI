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



def test_local_artifact_resolution_does_not_require_huggingface(tmp_path):
    from ml.inference.artifacts import resolve_artifact_paths

    model = tmp_path / "model.h5"
    preprocessor = tmp_path / "preprocessor.pkl"
    model.write_bytes(b"model")
    preprocessor.write_bytes(b"preprocessor")

    resolved_model, resolved_preprocessor, manifest = resolve_artifact_paths(
        model, preprocessor
    )

    assert resolved_model == model
    assert resolved_preprocessor == preprocessor
    assert manifest is None


def test_hub_artifact_resolution_uses_pinned_revision_and_manifest(tmp_path, monkeypatch):
    import sys
    import types
    from ml.inference.artifacts import resolve_artifact_paths

    _write_artifacts(tmp_path)
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    model, preprocessor, manifest = resolve_artifact_paths(
        "local-model.h5",
        "local-preprocessor.pkl",
        repo_id="test-user/sentinel-ai-models",
        revision="a" * 40,
    )

    assert model.name == "sentinel_cnn_lstm.h5"
    assert preprocessor.name == "preprocessor.pkl"
    assert manifest["feature_count"] == 78
    assert calls["revision"] == "a" * 40
    assert calls["repo_id"] == "test-user/sentinel-ai-models"
