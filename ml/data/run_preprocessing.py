"""
SENTINEL-AI — Data Preprocessing Pipeline Executor
====================================================

Phase 1 deliverable: Execute the full preprocessing pipeline on downloaded datasets.

This script:
    1. Loads CIC-IDS2017 and/or UNSW-NB15 datasets
    2. Cleans data (inf/NaN handling)
    3. Maps labels to the 6-class schema
    4. Selects features per feature_schema.json
    5. Fits StandardScaler on training data
    6. Creates stratified train/val/test splits (70/15/15)
    7. Saves processed data as .npy files and fitted preprocessor as .pkl

Run:
    python ml/data/run_preprocessing.py --dataset cicids
    python ml/data/run_preprocessing.py --dataset unsw
    python ml/data/run_preprocessing.py --dataset both

Output:
    - data/processed/X_train.npy, X_val.npy, X_test.npy
    - data/processed/y_train.npy, y_val.npy, y_test.npy
    - data/processed/preprocessing_report.json
    - ml/models/artifacts/preprocessor.pkl
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.data_loader import DataLoader
from ml.data.preprocessing import Preprocessor, CICIDS_LABEL_MAP, UNSW_LABEL_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Output directories
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"


def process_dataset(dataset: str = "cicids"):
    """
    Run the full preprocessing pipeline on the specified dataset.
    
    Args:
        dataset: "cicids", "unsw", or "both"
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_dir=str(PROJECT_ROOT / "data" / "raw"))
    preprocessor = Preprocessor(
        feature_schema_path=str(PROJECT_ROOT / "ml" / "config" / "feature_schema.json")
    )

    report = {
        "dataset": dataset,
        "preprocessing_steps": [],
    }

    # ==================== LOAD DATA ====================
    print("\n" + "=" * 70)
    print("  SENTINEL-AI — Preprocessing Pipeline")
    print("=" * 70)

    dfs = []

    if dataset in ("cicids", "both"):
        print("\n📂 Loading CIC-IDS2017...")
        try:
            df_cic = loader.load_cicids2017()
            print(f"   Loaded: {df_cic.shape[0]:,} rows, {df_cic.shape[1]} columns")

            # Find label column
            label_col = None
            for candidate in ["Label", " Label"]:
                if candidate in df_cic.columns:
                    label_col = candidate
                    break

            if label_col is None:
                raise ValueError("Could not find label column in CIC-IDS2017")

            # Map labels
            df_cic = preprocessor.map_labels(df_cic, label_col, dataset="cicids")
            report["preprocessing_steps"].append({
                "step": "load_cicids2017",
                "rows": df_cic.shape[0],
                "cols": df_cic.shape[1],
            })
            dfs.append(df_cic)
        except FileNotFoundError as e:
            print(f"   ⚠ {e}")
            if dataset == "cicids":
                print("   Cannot proceed without CIC-IDS2017. Run download script first.")
                return

    if dataset in ("unsw", "both"):
        print("\n📂 Loading UNSW-NB15...")
        try:
            df_unsw = loader.load_unsw_nb15()
            print(f"   Loaded: {df_unsw.shape[0]:,} rows, {df_unsw.shape[1]} columns")

            label_col = None
            for candidate in ["attack_cat", "Label", "label"]:
                if candidate in df_unsw.columns:
                    label_col = candidate
                    break

            if label_col is None:
                raise ValueError("Could not find label column in UNSW-NB15")

            df_unsw = preprocessor.map_labels(df_unsw, label_col, dataset="unsw")
            report["preprocessing_steps"].append({
                "step": "load_unsw_nb15",
                "rows": df_unsw.shape[0],
                "cols": df_unsw.shape[1],
            })
            dfs.append(df_unsw)
        except FileNotFoundError as e:
            print(f"   ⚠ {e}")
            if dataset == "unsw":
                print("   Cannot proceed without UNSW-NB15. Run download script first.")
                return

    if not dfs:
        print("\n❌ No datasets loaded. Exiting.")
        return

    # Combine if both
    if len(dfs) > 1:
        # Find common numeric columns
        common_cols = set(dfs[0].select_dtypes(include=[np.number]).columns)
        for df in dfs[1:]:
            common_cols &= set(df.select_dtypes(include=[np.number]).columns)
        common_cols = sorted(common_cols)
        print(f"\n🔀 Combining datasets ({len(common_cols)} common numeric features)...")
        df = pd.concat([d[list(common_cols) + ["label"]] for d in dfs], ignore_index=True)
    else:
        df = dfs[0]

    print(f"\n📊 Combined dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    report["combined_shape"] = list(df.shape)

    # ==================== CLEAN DATA ====================
    print("\n🧹 Cleaning data...")
    before = df.shape[0]
    df = preprocessor.clean_data(df)
    after = df.shape[0]
    dropped = before - after
    print(f"   Dropped {dropped:,} rows ({dropped/before*100:.2f}%) with inf/NaN")
    report["preprocessing_steps"].append({
        "step": "clean_data",
        "rows_before": before,
        "rows_after": after,
        "rows_dropped": dropped,
        "pct_dropped": round(dropped / before * 100, 2),
    })

    # ==================== CLASS DISTRIBUTION ====================
    print("\n📊 Final class distribution:")
    dist = df["label"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"   {label:20s} {count:>10,} ({pct:.2f}%)")

    report["class_distribution"] = dist.to_dict()

    # ==================== FIT & TRANSFORM ====================
    print("\n⚙️  Running fit_transform (stratified 70/15/15 split)...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(
        df, label_col="label", test_size=0.15, val_size=0.15, random_state=42
    )

    print(f"\n   Split Sizes:")
    print(f"     Train: {X_train.shape[0]:>10,} samples, {X_train.shape[1]} features")
    print(f"     Val:   {X_val.shape[0]:>10,} samples")
    print(f"     Test:  {X_test.shape[0]:>10,} samples")

    report["splits"] = {
        "train": X_train.shape[0],
        "val": X_val.shape[0],
        "test": X_test.shape[0],
        "features": X_train.shape[1],
    }

    # Verify stratification
    print(f"\n   Stratification Check (class proportions):")
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        props = counts / len(y_split) * 100
        prop_str = ", ".join([f"C{u}={p:.1f}%" for u, p in zip(unique, props)])
        print(f"     {name:6s}: {prop_str}")

    # ==================== SAVE ARTIFACTS ====================
    print("\n💾 Saving processed data...")

    # Save numpy arrays
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    print(f"   ✅ Saved to: {PROCESSED_DIR}")

    # Save preprocessor
    preprocessor.save(str(ARTIFACTS_DIR / "preprocessor.pkl"))
    print(f"   ✅ Preprocessor saved to: {ARTIFACTS_DIR / 'preprocessor.pkl'}")

    # Save feature names
    feature_names_path = PROCESSED_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(preprocessor.fitted_feature_names, f, indent=2)
    print(f"   ✅ Feature names saved to: {feature_names_path}")

    # Save report
    report_path = PROCESSED_DIR / "preprocessing_report.json"
    report["feature_names"] = preprocessor.fitted_feature_names
    report["num_features"] = len(preprocessor.fitted_feature_names)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"   ✅ Report saved to: {report_path}")

    # ==================== VERIFICATION ====================
    print("\n🔍 Verification:")
    X_train_loaded = np.load(PROCESSED_DIR / "X_train.npy")
    y_train_loaded = np.load(PROCESSED_DIR / "y_train.npy")
    print(f"   Loaded X_train: shape={X_train_loaded.shape}, dtype={X_train_loaded.dtype}")
    print(f"   Loaded y_train: shape={y_train_loaded.shape}, unique classes={np.unique(y_train_loaded)}")
    print(f"   Feature range: [{X_train_loaded.min():.4f}, {X_train_loaded.max():.4f}]")
    print(f"   Mean ≈ 0: {abs(X_train_loaded.mean()):.6f}")
    print(f"   Std ≈ 1:  {X_train_loaded.std():.6f}")

    print("\n" + "=" * 70)
    print("  ✅ Preprocessing Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL-AI Preprocessing Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["cicids", "unsw", "both"],
        default="cicids",
        help="Which dataset to preprocess (default: cicids)",
    )
    args = parser.parse_args()
    process_dataset(args.dataset)
