#!/usr/bin/env python3
"""

Standalone script that (re)trains the CNN-LSTM from the preprocessed arrays and
saves everything needed to audit or reproduce the run.

It:
    1. Loads and validates the preprocessed .npy splits
       (X/y for train, val and test)
    2. Builds the CNN-LSTM via ``Trainer`` using ml/config/model_config.yaml
    3. Trains with class weights ("balanced"), EarlyStopping and ReduceLROnPlateau
       (the callbacks come from ``get_training_callbacks``)
    4. Saves the model, then evaluates the SAVED file on the test split using
       the same code as ml/models/run_evaluation.py
    5. Saves the training history, training curves and a run summary
       (config, seed, library versions, git commit, data hashes, class weights,
       metrics)



"""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

# Quiet TensorFlow's C++ logging unless the user already chose a level.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import MaxNLocator

# Add project root to path (same pattern as the other ml/ scripts)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.preprocessing import CLASS_ORDER  # noqa: E402

try:
    from ml.models.run_evaluation import resolve_class_names, run_evaluation  # noqa: E402
except ImportError as e:  # pragma: no cover - friendly message for a missing file
    raise SystemExit(
        f"Could not import ml/models/run_evaluation.py ({e}). "
        "run_training.py reuses it for evaluation; make sure it is in ml/models/."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "ml" / "config" / "model_config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --smoke-test sizes (stratified subsets) and epochs
SMOKE_TRAIN, SMOKE_VAL, SMOKE_TEST, SMOKE_EPOCHS = 20_000, 4_000, 5_000, 2


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Show paths relative to the project root when possible."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _json_default(obj):
    """Make numpy scalars/arrays and Paths JSON-serialisable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def sha256_of(path: Path, chunk_size: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    """Commit hash of the code that produced the run (None if not a git checkout)."""
    try:
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=15, check=True
        ).stdout.strip()
        return {
            "commit": run("rev-parse", "HEAD"),
            "has_uncommitted_changes": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except Exception:
        return {"commit": None, "has_uncommitted_changes": None}


def environment_info() -> dict:
    import sklearn
    import tensorflow as tf
    from tensorflow import keras

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_split(x_path: Path, y_path: Path, split: str, num_classes: int):
    """Load and validate one split. Returns (X float32, y int64, file metadata)."""
    for path in (x_path, y_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Generate the splits first with: "
                "python ml/data/run_preprocessing.py --dataset cicids"
            )

    X = np.load(x_path)
    y = np.load(y_path)
    meta = {
        "x_file": {"path": _rel(x_path), "shape": list(X.shape), "dtype": str(X.dtype)},
        "y_file": {"path": _rel(y_path), "shape": list(y.shape), "dtype": str(y.dtype)},
    }

    if X.ndim != 2:
        raise ValueError(f"X_{split} must be 2-D (samples, features); got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y_{split} must be 1-D integer class ids; got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X_{split} has {len(X):,} rows but y_{split} has {len(y):,}")
    if not np.isfinite(X).all():
        raise ValueError(f"X_{split} contains NaN/inf; fix preprocessing before training")
    if not np.array_equal(y, y.astype(np.int64)):
        raise ValueError(f"y_{split} must contain integer class ids")
    y = y.astype(np.int64)
    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError(
            f"y_{split} labels span [{y.min()}, {y.max()}] but the model has {num_classes} classes"
        )

    meta["x_file"]["sha256"] = sha256_of(x_path)
    meta["y_file"]["sha256"] = sha256_of(y_path)
    return X.astype(np.float32, copy=False), y, meta


def stratified_subset(X, y, n, rng, min_per_class=30):
    """Random subset of ~n rows that keeps every class represented (smoke test only)."""
    picks = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(len(idx), max(min_per_class, int(round(n * len(idx) / len(y)))))
        picks.append(rng.choice(idx, size=k, replace=False))
    sel = np.sort(np.concatenate(picks))
    return X[sel], y[sel]


def class_counts(y: np.ndarray, class_names: list) -> dict:
    return {name: int((y == i).sum()) for i, name in enumerate(class_names)}


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_training_curves(history: dict, best_epoch: int, out_path: Path) -> None:
    """Loss / accuracy / learning-rate curves, with the best (restored) epoch marked."""
    lr_key = next((k for k in ("learning_rate", "lr") if k in history), None)
    panels = [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]
    n_panels = len(panels) + (1 if lr_key else 0)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    axes = np.atleast_1d(axes)
    epochs = np.arange(1, len(history["loss"]) + 1)

    for ax, (train_key, val_key, title) in zip(axes, panels):
        if train_key in history:
            ax.plot(epochs, history[train_key], marker="o", ms=3, label="train")
        if val_key in history:
            ax.plot(epochs, history[val_key], marker="o", ms=3, label="validation")
        ax.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"best epoch ({best_epoch})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if lr_key:
        ax = axes[-1]
        ax.plot(epochs, history[lr_key], marker="o", ms=3, color="tab:green")
        ax.set_yscale("log")
        ax.set_title("Learning rate")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("SENTINEL-AI CNN-LSTM — Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc)
    t_run = time.perf_counter()

    # ---- Config and effective settings ----
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_cfg, arch = config["training"], config["model"]["architecture"]
    paths_cfg = config.get("paths", {})

    model_artifact = Path(paths_cfg.get("model_artifact", "ml/models/artifacts/sentinel_cnn_lstm.h5"))
    out_dir = args.output_dir or (PROJECT_ROOT / model_artifact.parent)
    if args.smoke_test:
        out_dir = out_dir / "smoke_test"
    model_path = out_dir / model_artifact.name

    preprocessor_path = args.preprocessor or (
        PROJECT_ROOT / paths_cfg.get("preprocessor", "ml/models/artifacts/preprocessor.pkl")
    )

    epochs = args.epochs if args.epochs is not None else (
        SMOKE_EPOCHS if args.smoke_test else int(train_cfg["epochs"])
    )
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    seed = args.seed if args.seed is not None else int(config.get("data", {}).get("random_state", 42))
    use_class_weights = not args.no_class_weights
    num_classes = int(arch.get("num_classes", len(CLASS_ORDER)))

    if epochs < 1 or batch_size < 1:
        raise ValueError("--epochs and --batch-size must be >= 1")

    # Fail fast: get_training_callbacks() always creates a TensorBoard callback, and
    # without the package training would crash at epoch 1, after data loading.
    if importlib.util.find_spec("tensorboard") is None:
        raise ValueError(
            "TensorBoard is not installed, but get_training_callbacks() always uses it. "
            "Install the project requirements: pip install -r ml/requirements.txt"
        )
    if train_cfg.get("class_weights") not in ("balanced", None):
        logger.warning(
            "Config training.class_weights=%r is not supported by Trainer (it only "
            "implements 'balanced'); using 'balanced'.", train_cfg.get("class_weights"),
        )

    # ---- Safety: never silently replace an existing model ----
    if model_path.exists() and not args.smoke_test and not args.overwrite:
        raise FileExistsError(
            f"{_rel(model_path)} already exists. Re-run with --overwrite to replace it, "
            "or use --output-dir to train into a different folder."
        )

    logger.info("=" * 70)
    logger.info("SENTINEL-AI training run%s", "  [SMOKE TEST]" if args.smoke_test else "")
    logger.info("=" * 70)

    # ---- 1. Data ----
    logger.info("[1/6] Loading data from %s", _rel(args.data_dir))
    data, data_meta = {}, {}
    for split in ("train", "val", "test"):
        X, y, meta = load_split(
            args.data_dir / f"X_{split}.npy", args.data_dir / f"y_{split}.npy", split, num_classes
        )
        data[split], data_meta[split] = (X, y), meta
        logger.info("  %-5s %9s samples x %d features", split, f"{len(X):,}", X.shape[1])

    n_features = {X.shape[1] for X, _ in data.values()}
    if len(n_features) != 1:
        raise ValueError(f"Splits disagree on feature count: {sorted(n_features)}")
    num_features = n_features.pop()

    class_names, names_source = resolve_class_names(preprocessor_path, num_classes)

    if args.smoke_test:
        rng = np.random.default_rng(seed)
        sizes = {"train": SMOKE_TRAIN, "val": SMOKE_VAL, "test": SMOKE_TEST}
        for split, n in sizes.items():
            data[split] = stratified_subset(*data[split], n=n, rng=rng)
        logger.warning(
            "SMOKE TEST: using stratified subsets (%s) and %d epochs. Results are NOT meaningful.",
            {s: len(data[s][0]) for s in data}, epochs,
        )

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    counts = {s: class_counts(y, class_names) for s, (_, y) in data.items()}
    for cls in class_names:
        if counts["train"][cls] == 0:
            logger.warning("Class %r has no training samples; the model cannot learn it.", cls)
    logger.info("  Train class counts: %s", counts["train"])

    # ---- Working directory ----
    # get_training_callbacks() hard-codes relative paths for the checkpoint
    # (ml/models/checkpoints/) and TensorBoard logs (ml/logs/). Run from the
    # project root so they always land in the same place, or, for a smoke test,
    # from the smoke folder so a throwaway run cannot touch real checkpoints/logs.
    workdir = out_dir if args.smoke_test else PROJECT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    Path("ml/models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("ml/logs").mkdir(parents=True, exist_ok=True)

    # ---- 2. Seeds & determinism ----
    logger.info("[2/6] Seeding (seed=%d, deterministic_ops=%s)", seed, args.deterministic)
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(seed)  # seeds Python, NumPy and TensorFlow
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    # ---- 3. Build ----
    logger.info("[3/6] Building model via Trainer")
    from ml.models.trainer import Trainer  # imported here so --help and data errors are fast

    trainer = Trainer(num_features=num_features, num_classes=num_classes, config_path=str(args.config))
    trainer.build()
    n_params = int(trainer.model.count_params())
    if float(arch.get("lstm_recurrent_dropout", 0)) > 0:
        logger.info(
            "  Note: lstm_recurrent_dropout > 0 disables TensorFlow's cuDNN LSTM kernels, "
            "so training is slow even on a GPU."
        )

    class_weights = trainer.compute_class_weights(y_train) if use_class_weights else None
    named_weights = (
        {class_names[int(k)]: float(v) for k, v in class_weights.items()} if class_weights else None
    )

    # ---- 4. Train ----
    logger.info("[4/6] Training: up to %d epochs, batch_size=%d, class_weights=%s",
                epochs, batch_size, use_class_weights)
    t_train = time.perf_counter()
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, use_class_weights=use_class_weights,
    )
    train_seconds = time.perf_counter() - t_train

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    epochs_run = len(hist["loss"])
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    lr_key = next((k for k in ("learning_rate", "lr") if k in hist), None)

    # ---- 5. Save model, history, curves ----
    logger.info("[5/6] Saving artifacts to %s", _rel(out_dir))
    trainer.save_model(str(model_path))
    _write_json(out_dir / "training_history.json", hist)
    plot_training_curves(hist, best_epoch, out_dir / "training_curves.png")

    # ---- 6. Evaluate the SAVED model on the test split ----
    logger.info("[6/6] Evaluating saved model on the test split")
    if args.smoke_test:
        # run_evaluation reads arrays from disk, so persist the test subset.
        smoke_data = out_dir / "data"
        smoke_data.mkdir(exist_ok=True)
        x_test_path, y_test_path = smoke_data / "X_test.npy", smoke_data / "y_test.npy"
        np.save(x_test_path, X_test)
        np.save(y_test_path, y_test)
    else:
        x_test_path, y_test_path = args.data_dir / "X_test.npy", args.data_dir / "y_test.npy"

    eval_dir = out_dir / "evaluation"
    eval_metrics, eval_error = None, None
    try:
        eval_metrics = run_evaluation(Namespace(
            model=model_path, x_test=x_test_path, y_test=y_test_path,
            preprocessor=preprocessor_path, output_dir=eval_dir,
            batch_size=1024, class_names=None,
        ))
    except (FileNotFoundError, ValueError) as e:
        eval_error = str(e)
        logger.error("Evaluation failed: %s", e)

    # ---- Run summary ----
    summary = {
        "run": {
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_seconds": round(time.perf_counter() - t_run, 1),
            "smoke_test": args.smoke_test,
            "command": " ".join(sys.argv),
            "seed": seed,
            "deterministic_ops": args.deterministic,
            "git": git_info(),
        },
        "environment": environment_info(),
        "config": {"file": _rel(args.config), "contents": config},
        "settings": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "use_class_weights": use_class_weights,
        },
        "data": {
            "directory": _rel(args.data_dir),
            "num_features": num_features,
            "num_classes": num_classes,
            "class_names": class_names,
            "class_names_source": names_source,
            "class_counts_used": counts,
            "files": data_meta,
        },
        "class_weights": named_weights,
        "training": {
            "epochs_run": epochs_run,
            "stopped_early": epochs_run < epochs,
            "best_epoch": best_epoch,
            "best_val_loss": min(hist["val_loss"]),
            "val_accuracy_at_best_epoch": hist["val_accuracy"][best_epoch - 1],
            "final_learning_rate": hist[lr_key][-1] if lr_key else None,
            "training_seconds": round(train_seconds, 1),
        },
        "model": {
            "path": _rel(model_path),
            "parameters": n_params,
            "sha256": sha256_of(model_path),
        },
        "evaluation": {
            "directory": _rel(eval_dir),
            "error": eval_error,
            "overall": eval_metrics["overall"] if eval_metrics else None,
        },
    }
    _write_json(out_dir / "training_summary.json", summary)

    # ---- Console wrap-up ----
    logger.info("=" * 70)
    logger.info("Done in %.1f min. Epochs run: %d (best: %d%s)",
                (time.perf_counter() - t_run) / 60, epochs_run, best_epoch,
                ", stopped early" if epochs_run < epochs else "")
    if eval_metrics:
        o = eval_metrics["overall"]
        logger.info("Test accuracy %.4f | macro F1 %.4f | weighted F1 %.4f",
                    o["accuracy"], o["macro_f1"], o["weighted_f1"])
    logger.info("Saved to %s:", _rel(out_dir))
    for name in (model_path.name, "training_history.json", "training_curves.png",
                 "training_summary.json", "evaluation/"):
        logger.info("  - %s", name)
    logger.info("=" * 70)

    return 1 if eval_error else 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the SENTINEL-AI CNN-LSTM reproducibly and save everything.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tip: run with --smoke-test first to verify the whole pipeline in minutes.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Model config YAML")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Folder with X_/y_ train, val, test .npy files (default: data/processed)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to save outputs (default: folder of paths.model_artifact in the config)")
    p.add_argument("--preprocessor", type=Path, default=None,
                   help="preprocessor.pkl used for class-name order (default: from config)")
    p.add_argument("--epochs", type=int, default=None, help="Max epochs (default: from config)")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: data.random_state from config)")
    p.add_argument("--no-class-weights", action="store_true",
                   help="Train WITHOUT class weights (default: balanced class weights)")
    p.add_argument("--deterministic", action="store_true",
                   help="Enable TensorFlow deterministic ops (slower, more repeatable)")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow replacing an existing model file")
    p.add_argument("--smoke-test", action="store_true",
                   help=f"Quick pipeline check: stratified subsets, {SMOKE_EPOCHS} epochs, "
                        "isolated output folder")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Resolve user-supplied paths against the CALLER's directory now, because
    # run() changes the working directory.
    for name in ("config", "data_dir", "output_dir", "preprocessor"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        logger.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
