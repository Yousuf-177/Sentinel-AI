# Data Directory

This directory stores datasets. It is **gitignored** — datasets are too large to commit.

## Structure

```
data/
├── raw/              # Original downloaded datasets
│   ├── cicids2017/   # CIC-IDS2017 CSV files
│   └── unsw_nb15/    # UNSW-NB15 CSV files
├── processed/        # Cleaned, preprocessed data (parquet)
└── interim/          # Intermediate processing artifacts
```

## Setup

Run the download script for instructions:
```bash
python infra/scripts/download_datasets.py
```

Or download manually from:
- **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
