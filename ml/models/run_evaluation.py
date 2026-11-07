"""
SENTINEL-AI — Model Evaluation Runner
=====================================

Evaluate the trained CNN-LSTM on the held-out test split.

This script:
    1. Loads the trained model (sentinel_cnn_lstm.h5) and the test arrays
       (X_test.npy, y_test.npy)
    2. Runs model.predict() and prints a classification report
    3. Saves metrics as JSON: accuracy, macro F1, weighted F1, and per-class
       precision / recall / F1
    4. Saves a confusion-matrix heatmap (raw counts + row-normalised) as PNG
    5. Reports the False Positive Rate (FPR) for every class

Run (from the project root):
    python ml/models/run_evaluation.py
    python ml/models/run_evaluation.py --output-dir reports/evaluation_v2

Output (default directory: reports/evaluation/):
    - evaluation_metrics.json
    - confusion_matrix.png
    - classification_report.txt

CLASS ORDER — READ THIS
-----------------------
y_test.npy holds integers produced by sklearn's LabelEncoder, which sorts its
labels alphabetically:

    0=Benign  1=BruteForce  2=DoS/DDoS  3=Infiltration  4=PortScan  5=WebAttack

That is NOT the order hard-coded in Trainer.class_names,
InferencePipeline.class_names or feature_schema.json. Using those lists would
attach the wrong names to every per-class number. This script therefore reads
the class order from the label_encoder stored in preprocessor.pkl (the same
encoder that produced y_test.npy) and warns if it differs from the repo's
hard-coded order.

FALSE POSITIVE RATE
-------------------
FPR is computed one-vs-rest from the confusion matrix:

    FPR_k = FP_k / (FP_k + TN_k)

where FP_k = samples of any *other* class that were predicted as class k.
For an attack class this is the rate at which unrelated traffic is wrongly
flagged as that attack (relevant because SENTINEL-AI auto-mitigates with
iptables). For Benign it is the rate at which attack flows slip through
labelled as Benign.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Quiet TensorFlow's C++ logging unless the user has already set a level.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: works on servers, Docker, CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

# Add project root to path (same pattern as ml/data/run_preprocessing.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.preprocessing import CLASS_ORDER  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default locations (match ml/config/model_config.yaml and run_preprocessing.py)
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL = ARTIFACTS_DIR / "sentinel_cnn_lstm.h5"
DEFAULT_PREPROCESSOR = ARTIFACTS_DIR / "preprocessor.pkl"
DEFAULT_X_TEST = PROCESSED_DIR / "X_test.npy"
DEFAULT_Y_TEST = PROCESSED_DIR / "y_test.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation"


def _display_path(path: Path) -> str:
    """Show paths relative to the project root when possible."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_test_data(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate X_test / y_test. Returns (float32 features, int64 labels)."""
    for path in (x_path, y_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Generate the splits first with: "
                "python ml/data/run_preprocessing.py --dataset cicids"
            )

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(f"X_test must be 2-D (samples, features); got shape {X.shape}")

    # Accept one-hot labels too, but y_test.npy is normally integer class ids.
    if y.ndim == 2:
        if y.shape[1] > 1:
            logger.info("y_test looks one-hot encoded; converting with argmax")
            y = np.argmax(y, axis=1)
        else:
            y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y_test must be 1-D class ids; got shape {y.shape}")

    if len(X) != len(y):
        raise ValueError(f"X_test has {len(X):,} rows but y_test has {len(y):,}")
    if not np.isfinite(X).all():
        raise ValueError("X_test contains NaN/inf; predictions would be meaningless")
    if not np.array_equal(y, y.astype(np.int64)):
        raise ValueError("y_test must contain integer class ids")

    return X.astype(np.float32, copy=False), y.astype(np.int64)


def load_model(model_path: Path):
    """Load the trained Keras model for inference."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first "
            "(see ml/models/trainer.py) or pass --model."
        )
    # Imported lazily so the metric helpers below work without TensorFlow.
    from tensorflow import keras

    # compile=False: inference only, so optimizer/loss need not be rebuilt
    # (avoids deserialisation issues across Keras versions).
    model = keras.models.load_model(model_path, compile=False)
    logger.info(f"Model loaded from {_display_path(model_path)} ({model.count_params():,} params)")
    return model


def resolve_class_names(
    preprocessor_path: Path, num_classes: int, override: str | None = None
) -> tuple[list[str], str]:
    """
    Work out which class name belongs to each integer label.

    Priority:
        1. --class-names override
        2. label_encoder.classes_ inside preprocessor.pkl (ground truth: this
           is the encoder that produced y_test.npy)
        3. sorted(CLASS_ORDER), which is what LabelEncoder.fit() yields

    Returns (class_names, human-readable source).
    """
    if override:
        names = [n.strip() for n in override.split(",") if n.strip()]
        source = "--class-names"
    elif Path(preprocessor_path).exists():
        # Trusted project artifact written by this repo's own preprocessing step.
        data = joblib.load(preprocessor_path)
        names = [str(c) for c in data["label_encoder"].classes_]
        source = f"label_encoder in {_display_path(preprocessor_path)}"
    else:
        names = sorted(CLASS_ORDER)
        source = "sorted(CLASS_ORDER) fallback (preprocessor.pkl not found)"
        logger.warning(
            f"{_display_path(preprocessor_path)} not found; assuming LabelEncoder's "
            "alphabetical order. Pass --class-names to override."
        )

    if len(names) != num_classes:
        raise ValueError(
            f"Got {len(names)} class names ({names}) but the model has {num_classes} outputs"
        )

    if names != CLASS_ORDER:
        logger.warning(
            "Class order from %s is %s, which differs from the order hard-coded in "
            "Trainer.class_names / InferencePipeline.class_names / feature_schema.json "
            "(%s). Metrics below use the encoder's order, which matches y_test.npy. "
            "Trainer.evaluate() and InferencePipeline will mislabel classes until "
            "their lists are fixed.",
            source,
            names,
            CLASS_ORDER,
        )
    return names, source


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """
    Compute overall metrics, per-class precision/recall/F1/FPR and the confusion matrix.

    Every class is always reported (labels=range(n)), even if it is absent from
    the test set. Macro averages only include classes that actually occur in
    y_true, so an empty class cannot silently drag the average down.
    """
    n = len(class_names)
    labels = list(range(n))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # One-vs-rest counts for each class k
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp  # predicted k, actually something else
    fn = cm.sum(axis=1) - tp  # actually k, predicted something else
    tn = cm.sum() - tp - fp - fn
    fpr = np.divide(fp, fp + tn, out=np.zeros(n, dtype=float), where=(fp + tn) > 0)

    present = [i for i in labels if support[i] > 0]
    if len(present) < n:
        missing = [class_names[i] for i in labels if support[i] == 0]
        logger.warning(f"No test samples for class(es) {missing}; excluded from macro averages")

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
    }

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "false_positive_rate": float(fpr[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "tn": int(tn[i]),
        }

    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
    )

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": {"labels": class_names, "matrix": cm.tolist()},
        "_report_text": report_text,  # stripped before JSON is written
    }


def format_fpr_table(per_class: dict) -> str:
    """Console-friendly table of one-vs-rest false positive rates."""
    width = max(len(name) for name in per_class)
    lines = [f"{'Class':<{width}}  {'FP':>9}  {'TN':>11}  {'FPR':>8}", "-" * (width + 33)]
    for name, m in per_class.items():
        lines.append(
            f"{name:<{width}}  {m['fp']:>9,}  {m['tn']:>11,}  {m['false_positive_rate']:>8.4%}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    """
    Save a two-panel heatmap: raw counts (left) and row-normalised recall (right).

    The normalised panel matters because the classes are heavily imbalanced:
    in the raw-count panel Benign dominates the colour scale and hides errors
    on small classes.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    panels = [
        (axes[0], cm, "d", "Confusion Matrix (counts)"),
        (axes[1], cm_norm, ".1%", "Confusion Matrix (row-normalised: % of true class)"),
    ]
    for ax, data, fmt, title in panels:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("SENTINEL-AI CNN-LSTM — Test Set Evaluation", fontsize=15, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_evaluation(args: argparse.Namespace) -> dict:
    X_test, y_test = load_test_data(args.x_test, args.y_test)
    logger.info(f"Test data: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

    model = load_model(args.model)

    expected_features = model.input_shape[-1]
    if X_test.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features but X_test has {X_test.shape[1]}. "
            "Was the model trained on a different feature set?"
        )
    num_classes = int(model.output_shape[-1])
    if y_test.min() < 0 or y_test.max() >= num_classes:
        raise ValueError(
            f"y_test labels span [{y_test.min()}, {y_test.max()}] but the model has "
            f"{num_classes} outputs"
        )

    class_names, names_source = resolve_class_names(args.preprocessor, num_classes, args.class_names)

    logger.info("Running model.predict() ...")
    y_proba = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(y_proba, axis=1)

    result = compute_metrics(y_test, y_pred, class_names)
    report_text = result.pop("_report_text")

    # ---- Console output ----
    print("\n" + "=" * 70)
    print("  SENTINEL-AI — Test Set Evaluation")
    print("=" * 70)
    print(f"  Samples:      {len(y_test):,}")
    print(f"  Class labels: {names_source}")
    print("-" * 70)
    print(report_text)
    print("-" * 70)
    print("  Overall metrics")
    for key, value in result["overall"].items():
        print(f"    {key:<16} {value:.4f}")
    print("-" * 70)
    print("  False Positive Rate per class (one-vs-rest)")
    print(format_fpr_table(result["per_class"]))
    print("=" * 70 + "\n")

    # ---- Save artifacts ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "metadata": {
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_path": _display_path(args.model),
            "x_test_path": _display_path(args.x_test),
            "y_test_path": _display_path(args.y_test),
            "num_samples": int(len(y_test)),
            "num_features": int(X_test.shape[1]),
            "num_classes": num_classes,
            "model_parameters": int(model.count_params()),
            "class_names": class_names,
            "class_names_source": names_source,
            "class_order_matches_repo_constants": class_names == CLASS_ORDER,
        },
        **result,
    }

    json_path = out_dir / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    png_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(np.array(result["confusion_matrix"]["matrix"]), class_names, png_path)

    txt_path = out_dir / "classification_report.txt"
    txt_path.write_text(report_text)

    print("Saved:")
    for p in (json_path, png_path, txt_path):
        print(f"  - {_display_path(p)}")

    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the SENTINEL-AI CNN-LSTM on the held-out test set."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the .h5 model")
    parser.add_argument("--x-test", type=Path, default=DEFAULT_X_TEST, help="Path to X_test.npy")
    parser.add_argument("--y-test", type=Path, default=DEFAULT_Y_TEST, help="Path to y_test.npy")
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=DEFAULT_PREPROCESSOR,
        help="Path to preprocessor.pkl (source of the class-name order)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write the outputs"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for predict()")
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Comma-separated class names in label order (overrides preprocessor.pkl)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run_evaluation(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())