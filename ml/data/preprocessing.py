"""
SENTINEL-AI — Preprocessing Module

Handles data cleaning, feature engineering, label encoding to the 6-class schema,
scaling, and train/val/test splitting.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

logger = logging.getLogger(__name__)

# Mapping from raw CIC-IDS2017 labels to our 6-class schema
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
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack – XSS": "WebAttack",
}

# Mapping from raw UNSW-NB15 labels to our 6-class schema
UNSW_LABEL_MAP = {
    "Normal": "Benign",
    "DoS": "DoS/DDoS",
    "Exploits": "WebAttack",
    "Fuzzers": "WebAttack",
    "Generic": "DoS/DDoS",
    "Reconnaissance": "PortScan",
    "Analysis": "Infiltration",
    "Backdoor": "Infiltration",
    "Backdoors": "Infiltration",
    "Shellcode": "Infiltration",
    "Worms": "Infiltration",
}

CLASS_ORDER = ["Benign", "DoS/DDoS", "PortScan", "BruteForce", "WebAttack", "Infiltration"]


class Preprocessor:
    """
    Preprocessing pipeline for SENTINEL-AI.
    
    Handles:
    - Inf/NaN cleaning
    - Label mapping to 6-class schema
    - Feature selection (based on feature_schema.json)
    - StandardScaler fitting/transforming
    - Stratified train/val/test splitting
    - Saving/loading the fitted preprocessor
    
    Usage:
        preprocessor = Preprocessor(feature_schema_path="ml/config/feature_schema.json")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(df, label_col="Label")
        preprocessor.save("ml/models/artifacts/preprocessor.pkl")
    """

    def __init__(self, feature_schema_path: str = "ml/config/feature_schema.json"):
        self.feature_schema_path = Path(feature_schema_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False

        # Load feature schema
        if self.feature_schema_path.exists():
            with open(self.feature_schema_path, "r") as f:
                schema = json.load(f)
            self.feature_names = [feat["name"] for feat in schema["features"]]
            logger.info(f"Loaded feature schema: {len(self.feature_names)} features")
        else:
            logger.warning(f"Feature schema not found at {self.feature_schema_path}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset: handle inf values, NaN values, and invalid entries.
        """
        logger.info(f"Cleaning data: {df.shape[0]} rows")

        # Replace inf with NaN, then drop NaN rows
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        before = df.shape[0]
        df = df.dropna()
        after = df.shape[0]

        logger.info(f"  Dropped {before - after} rows with inf/NaN ({(before-after)/before*100:.2f}%)")
        return df.reset_index(drop=True)

    def map_labels(self, df: pd.DataFrame, label_col: str, dataset: str = "cicids") -> pd.DataFrame:
        """
        Map raw dataset labels to the 6-class SENTINEL-AI schema.
        
        Args:
            df: DataFrame with raw labels.
            label_col: Name of the label column.
            dataset: "cicids" or "unsw" to select the right mapping.
        
        Returns:
            DataFrame with a new 'label' column containing mapped labels.
        """
        label_map = CICIDS_LABEL_MAP if dataset == "cicids" else UNSW_LABEL_MAP

        df["label"] = df[label_col].map(label_map)

        # Check for unmapped labels
        unmapped = df[df["label"].isna()][label_col].unique()
        if len(unmapped) > 0:
            logger.warning(f"Unmapped labels found: {unmapped}. Dropping these rows.")
            df = df.dropna(subset=["label"])

        logger.info(f"Label mapping complete. Distribution:")
        for cls, count in df["label"].value_counts().items():
            logger.info(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        """
        Fit the preprocessor and transform data into train/val/test splits.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Select features
        available_features = [f for f in self.feature_names if f in df.columns]
        if not available_features:
            # Fallback: use all numeric columns except the label
            available_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col]
            logger.warning(f"Feature schema columns not found. Using {len(available_features)} numeric columns.")
        else:
            logger.info(f"Using {len(available_features)} of {len(self.feature_names)} schema features")

        X = df[available_features].values
        y = df[label_col].values

        # Encode labels
        self.label_encoder.fit(CLASS_ORDER)
        y_encoded = self.label_encoder.transform(y)

        # Stratified split: first train+val vs test, then train vs val
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=relative_val_size, random_state=random_state, stratify=y_trainval
        )

        # Fit scaler on training data only
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.is_fitted = True
        self.fitted_feature_names = available_features

        logger.info(f"Split sizes — Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the already-fitted preprocessor.
        Used for inference on live data.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before transform. Call fit_transform() first.")

        X = df[self.fitted_feature_names].values
        return self.scaler.transform(X)

    def save(self, path: str):
        """Save the fitted preprocessor to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.fitted_feature_names,
            "class_order": CLASS_ORDER,
        }, save_path)
        logger.info(f"Preprocessor saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        """Load a fitted preprocessor from disk."""
        data = joblib.load(path)
        preprocessor = cls.__new__(cls)
        preprocessor.scaler = data["scaler"]
        preprocessor.label_encoder = data["label_encoder"]
        preprocessor.fitted_feature_names = data["feature_names"]
        preprocessor.is_fitted = True
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
