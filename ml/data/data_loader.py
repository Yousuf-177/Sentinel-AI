"""
SENTINEL-AI — Data Loader Module

Handles loading CIC-IDS2017 and UNSW-NB15 datasets from disk,
basic validation, and conversion to a unified DataFrame format.
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and validates network intrusion detection datasets.
    
    Supports:
    - CIC-IDS2017 (CSV files per day)
    - UNSW-NB15 (CSV files in parts)
    
    Usage:
        loader = DataLoader(data_dir="data/raw")
        df = loader.load_cicids2017()
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")

    def load_cicids2017(self) -> pd.DataFrame:
        """
        Load all CIC-IDS2017 CSV files and concatenate into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined dataset with all days of traffic.
        """
        cicids_dir = self.data_dir / "cicids2017"
        if not cicids_dir.exists():
            raise FileNotFoundError(
                f"CIC-IDS2017 directory not found at {cicids_dir}. "
                "Run 'python infra/scripts/download_datasets.py' first."
            )

        csv_files = sorted(cicids_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {cicids_dir}")

        logger.info(f"Loading {len(csv_files)} CIC-IDS2017 files...")
        dfs = []
        for f in csv_files:
            logger.info(f"  Loading {f.name}...")
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            # Strip whitespace from column names (CIC-IDS2017 has leading spaces)
            df.columns = df.columns.str.strip()
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"CIC-IDS2017 loaded: {combined.shape[0]} rows, {combined.shape[1]} columns")
        return combined

    def load_unsw_nb15(self) -> pd.DataFrame:
        """
        Load UNSW-NB15 dataset files and concatenate.
        
        Returns:
            pd.DataFrame: Combined UNSW-NB15 dataset.
        """
        unsw_dir = self.data_dir / "unsw_nb15"
        if not unsw_dir.exists():
            raise FileNotFoundError(
                f"UNSW-NB15 directory not found at {unsw_dir}. "
                "Run 'python infra/scripts/download_datasets.py' first."
            )

        csv_files = sorted(unsw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {unsw_dir}")

        logger.info(f"Loading {len(csv_files)} UNSW-NB15 files...")
        dfs = []
        for f in csv_files:
            logger.info(f"  Loading {f.name}...")
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"UNSW-NB15 loaded: {combined.shape[0]} rows, {combined.shape[1]} columns")
        return combined

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Run basic validation checks on a loaded dataset.
        
        Returns:
            dict: Validation report with shape, dtypes, missing values, etc.
        """
        report = {
            "dataset": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": df.isnull().sum().sum(),
            "missing_by_column": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "inf_values": np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
        }

        # Check for label column
        label_candidates = ["Label", "label", "attack_cat", "Attack"]
        found_label = [c for c in label_candidates if c in df.columns]
        report["label_column"] = found_label[0] if found_label else None

        if report["label_column"]:
            report["class_distribution"] = df[report["label_column"]].value_counts().to_dict()

        logger.info(f"Validation report for {dataset_name}:")
        logger.info(f"  Shape: {report['shape']}")
        logger.info(f"  Missing values: {report['missing_values']}")
        logger.info(f"  Inf values: {report['inf_values']}")
        logger.info(f"  Duplicate rows: {report['duplicate_rows']}")

        return report
