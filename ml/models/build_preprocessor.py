"""
SENTINEL-AI — Build Preprocessor Artifact
==========================================

Rebuilds the preprocessor.pkl from the already-processed training data.

Since X_train.npy is already scaled, this script:
  1. Loads X_train.npy to fit a StandardScaler on it (identity-like since it's already scaled)
  OR
  2. Re-fits from raw data to get the ORIGINAL scaler parameters

This script uses Method 2 (re-fit from raw) to get the correct scaler that can
transform NEW unseen data. It loads the raw CIC-IDS2017 CSV, cleans it, and
fits the scaler on the UNSCALED training features.

Run:
    python ml/models/build_preprocessor.py

Output:
    ml/models/artifacts/preprocessor.pkl
"""

import os
import sys
import json
import logging
from pathlib import Path

# Fix OpenBLAS memory issue on low-RAM machines
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Paths
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"
FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_names.json"

# Label mapping (same as preprocessing.py)
CICIDS_LABEL_MAP = {
    "BENIGN": "Benign",
    "Bot": "Infiltration",
    "DDoS": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS Hulk": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Heartbleed": "WebAttack",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "Web Attack \u2013 Brute Force": "WebAttack",
    "Web Attack \u2013 Sql Injection": "WebAttack",
    "Web Attack \u2013 XSS": "WebAttack",
}

CLASS_ORDER = ["Benign", "DoS/DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration"]


def main():
    print("=" * 60)
    print("  SENTINEL-AI -- Build preprocessor.pkl")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature names (from previous preprocessing run)
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)
    print(f"\n[1/5] Loaded {len(feature_names)} feature names from {FEATURE_NAMES_PATH.name}")

    # Load all CIC-IDS2017 CSV files
    print(f"\n[2/5] Loading CIC-IDS2017 raw CSVs from {RAW_DIR}...")
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    dfs = []
    for f in csv_files:
        print(f"  Loading {f.name}...")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Map labels
    print(f"\n[3/5] Mapping labels to 6-class schema...")
    df["label"] = df["Label"].map(CICIDS_LABEL_MAP)
    df = df.dropna(subset=["label"])

    for cls in CLASS_ORDER:
        count = (df["label"] == cls).sum()
        print(f"  {cls:20s} {count:>10,}")

    # Clean: replace inf with NaN, drop NaN rows
    print(f"\n[4/5] Cleaning data & fitting scaler...")
    numeric_cols = [c for c in feature_names if c in df.columns]
    print(f"  Matching features: {len(numeric_cols)} of {len(feature_names)}")

    X = df[numeric_cols].copy()
    y = df["label"].copy()

    # Replace inf -> NaN, drop rows
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"  After cleaning: {X.shape[0]:,} rows")

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_ORDER)
    y_encoded = label_encoder.transform(y.values)

    # Stratified split (same params as original preprocessing)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X.values, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    relative_val_size = 0.15 / (1 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val_size, random_state=42, stratify=y_trainval
    )
    print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"  Scaler fitted on {X_train.shape[0]:,} training samples, {X_train.shape[1]} features")
    print(f"  Scaler mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Scaler std range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    # Save preprocessor
    print(f"\n[5/5] Saving preprocessor.pkl...")
    preprocessor_data = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": numeric_cols,
        "class_order": CLASS_ORDER,
    }

    save_path = ARTIFACTS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor_data, save_path)
    print(f"  Saved to: {save_path}")
    print(f"  File size: {save_path.stat().st_size / 1024:.1f} KB")

    # Verify by loading back
    print(f"\n-- Verification --")
    loaded = joblib.load(save_path)
    print(f"  Loaded keys: {list(loaded.keys())}")
    print(f"  Features: {len(loaded['feature_names'])}")
    print(f"  Classes: {list(loaded['label_encoder'].classes_)}")

    # Quick sanity check: transform a sample and compare
    X_train_scaled = loaded["scaler"].transform(X_train[:5])
    print(f"  Sample transform OK - shape: {X_train_scaled.shape}")
    print(f"  Sample mean (should be ~0): {X_train_scaled.mean():.6f}")

    print(f"\n{'=' * 60}")
    print(f"  preprocessor.pkl built successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
