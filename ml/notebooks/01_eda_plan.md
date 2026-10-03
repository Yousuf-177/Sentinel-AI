# SENTINEL-AI — EDA Notebook

This notebook will contain the Exploratory Data Analysis for CIC-IDS2017 and UNSW-NB15 datasets.

## Planned Analysis (Phase 1)

1. **Load & inspect** both datasets
2. **Class distribution** — bar chart (log scale) showing label imbalance
3. **Feature statistics** — missing values, inf values, ranges
4. **Correlation analysis** — heatmap of feature correlations
5. **Key feature boxplots** — SYN flag count, packet length variance, flow duration (benign vs attack)
6. **Feature importance** — mutual information / RF importance to identify top discriminative features
7. **Data quality report** — document all cleaning decisions

## To Run

```bash
# Ensure datasets are downloaded first
python infra/scripts/download_datasets.py

# Then start the notebook
jupyter notebook ml/notebooks/01_eda.ipynb
```

*This markdown file is a placeholder. The actual Jupyter notebook (.ipynb) will be created during Phase 1 execution.*
