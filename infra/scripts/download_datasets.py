"""
SENTINEL-AI — Dataset Download Script

Downloads CIC-IDS2017 and UNSW-NB15 datasets to data/raw/.
Run: python infra/scripts/download_datasets.py
"""

import os
import sys
from pathlib import Path


def main():
    """
    Prints download instructions for the datasets.
    
    NOTE: These datasets require manual download from their official sources
    due to licensing and hosting constraints. This script provides the links
    and expected directory structure.
    """
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SENTINEL-AI — Dataset Download Instructions")
    print("=" * 70)
    print()
    print("Please download the following datasets manually:")
    print()

    # CIC-IDS2017
    cicids_dir = data_dir / "cicids2017"
    cicids_dir.mkdir(parents=True, exist_ok=True)
    print("1. CIC-IDS2017")
    print("   URL: https://www.unb.ca/cic/datasets/ids-2017.html")
    print("   Alternative: https://www.kaggle.com/datasets/cicdataset/cicids2017")
    print(f"   Place CSV files in: {cicids_dir.resolve()}")
    print("   Expected files:")
    print("     - Monday-WorkingHours.pcap_ISCX.csv")
    print("     - Tuesday-WorkingHours.pcap_ISCX.csv")
    print("     - Wednesday-workingHours.pcap_ISCX.csv")
    print("     - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    print("     - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    print("     - Friday-WorkingHours-Morning.pcap_ISCX.csv")
    print("     - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    print("     - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    print()

    # UNSW-NB15
    unsw_dir = data_dir / "unsw_nb15"
    unsw_dir.mkdir(parents=True, exist_ok=True)
    print("2. UNSW-NB15")
    print("   URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("   Alternative: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
    print(f"   Place CSV files in: {unsw_dir.resolve()}")
    print("   Expected files:")
    print("     - UNSW-NB15_1.csv through UNSW-NB15_4.csv")
    print("     - UNSW-NB15_GT.csv (ground truth)")
    print("     - UNSW_NB15_training-set.csv")
    print("     - UNSW_NB15_testing-set.csv")
    print()

    # Data directory structure
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 70)
    print("Data directory structure created:")
    print(f"  {data_dir.resolve()}")
    print(f"  ├── cicids2017/     (place CIC-IDS2017 CSVs here)")
    print(f"  └── unsw_nb15/      (place UNSW-NB15 CSVs here)")
    print(f"  {processed_dir.resolve()}")
    print(f"  {interim_dir.resolve()}")
    print()
    print("After downloading, run the EDA notebook:")
    print("  jupyter notebook ml/notebooks/01_eda.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
