"""
SENTINEL-AI — SHAP batch runner

Explains a small stratified slice of the held-out test set and records how
long each explanation takes. This is the Phase 3 latency note.

Run from the project root:
    python ml/explainability/run_shap_batch.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability.shap_explainer import SHAPExplainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL = ARTIFACTS_DIR / "sentinel_cnn_lstm.h5"
DEFAULT_PREPROCESSOR = ARTIFACTS_DIR / "preprocessor.pkl"
DEFAULT_X_TEST = PROCESSED_DIR / "X_test.npy"
DEFAULT_Y_TEST = PROCESSED_DIR / "y_test.npy"
DEFAULT_FEATURES = PROCESSED_DIR / "feature_names.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "phase3" / "shap_latency.json"

N_EXPLAIN = 10
LATENCY_BUDGET_MS = 500.0
BACKGROUND_SIZES = (100, 20)


def stratified_indices(y: np.ndarray, n_total: int, seed: int = 42) -> np.ndarray:
    """Pick up to n_total rows, spread across the classes present in y."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, n_total // len(classes))
    chosen = []
    for class_id in classes:
        pool = np.flatnonzero(y == class_id)
        take = min(per_class, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    chosen = chosen[:n_total]
    if len(chosen) < n_total:
        rest = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=int), assume_unique=False)
        need = min(n_total - len(chosen), len(rest))
        chosen.extend(rng.choice(rest, size=need, replace=False).tolist())
    return np.asarray(chosen[:n_total], dtype=int)


def _load_model(path: Path):
    from tensorflow import keras

    return keras.models.load_model(path)


def explain_sample(model, x_test, feature_names, indices, background_size: int) -> dict:
    """Fit an explainer on a background slice and explain ``indices``."""
    rng = np.random.default_rng(42)
    bg_count = min(background_size, len(x_test))
    bg_index = rng.choice(len(x_test), size=bg_count, replace=False)
    background = np.asarray(x_test[bg_index], dtype=np.float32)
    rows = np.asarray(x_test[indices], dtype=np.float32)

    started = time.perf_counter()
    explainer = SHAPExplainer(model, background, max_background=background_size)
    init_seconds = time.perf_counter() - started

    started = time.perf_counter()
    explanations = explainer.explain_batch(rows, feature_names, top_k=3)
    batch_seconds = time.perf_counter() - started
    per_flow_ms = (batch_seconds / max(len(rows), 1)) * 1000.0

    proba = model.predict(rows, verbose=0)
    pred = np.argmax(proba, axis=1)

    return {
        "explainer": explainer.explainer_name,
        "background_size": int(len(explainer.background)),
        "init_seconds": round(init_seconds, 3),
        "batch_seconds": round(batch_seconds, 3),
        "per_flow_ms": round(per_flow_ms, 2),
        "explanations": explanations,
        "predicted_class": pred.astype(int).tolist(),
        "confidence": [round(float(proba[i, pred[i]]), 4) for i in range(len(rows))],
    }


def main() -> None:
    logger.info("Loading model, labels, and a memory-mapped test matrix")
    model = _load_model(DEFAULT_MODEL)
    x_test = np.load(DEFAULT_X_TEST, mmap_mode="r")
    y_test = np.load(DEFAULT_Y_TEST)
    with open(DEFAULT_FEATURES, "r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    bundle = joblib.load(DEFAULT_PREPROCESSOR)
    class_names = [str(name) for name in bundle["label_encoder"].classes_]

    if x_test.shape[1] != len(feature_names):
        raise ValueError(
            f"X_test has {x_test.shape[1]} columns but feature_names.json has {len(feature_names)}"
        )

    indices = stratified_indices(y_test, N_EXPLAIN)
    logger.info(f"Explaining {len(indices)} stratified test rows")

    background_reduced = False
    result = None
    for size in BACKGROUND_SIZES:
        logger.info(f"Trying background size {size}")
        result = explain_sample(model, x_test, feature_names, indices, size)
        logger.info(
            f"{result['explainer']}: {result['per_flow_ms']:.1f} ms/flow "
            f"(init {result['init_seconds']:.1f}s)"
        )
        if result["per_flow_ms"] <= LATENCY_BUDGET_MS:
            break
        background_reduced = True
        logger.warning(
            f"Per-flow explanation {result['per_flow_ms']:.1f} ms exceeds "
            f"{LATENCY_BUDGET_MS:.0f} ms. Reducing the background set."
        )

    samples = []
    for row_i, index in enumerate(indices):
        explanation = result["explanations"][row_i]
        true_name = class_names[int(y_test[index])]
        pred_name = class_names[int(result["predicted_class"][row_i])]
        samples.append({
            "row_index": int(index),
            "true_label": true_name,
            "predicted_label": pred_name,
            "confidence": result["confidence"][row_i],
            "human_readable": explanation["human_readable"],
            "top_features": explanation["top_features"],
        })
        logger.info(
            f"[{row_i}] true={true_name} pred={pred_name} "
            f"conf={result['confidence'][row_i]:.2%} | {explanation['human_readable']}"
        )

    report = {
        "explainer": result["explainer"],
        "background_size": result["background_size"],
        "background_reduced": background_reduced and result["per_flow_ms"] <= LATENCY_BUDGET_MS,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "within_budget": result["per_flow_ms"] <= LATENCY_BUDGET_MS,
        "n_flows": len(samples),
        "init_seconds": result["init_seconds"],
        "batch_seconds": result["batch_seconds"],
        "per_flow_ms": result["per_flow_ms"],
        "feature_count": len(feature_names),
        "class_names": class_names,
        "samples": samples,
    }
    # If we reduced the background and it still missed the budget, say so.
    if background_reduced and result["per_flow_ms"] > LATENCY_BUDGET_MS:
        report["background_reduced"] = True

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
