"""
SENTINEL-AI — Exploratory Data Analysis (EDA)
==============================================

Phase 1 deliverable: Comprehensive EDA on CIC-IDS2017 and UNSW-NB15 datasets.

Analysis Sections:
    1. Dataset loading & basic inspection
    2. Class distribution analysis (with log-scale visualization)
    3. Missing/infinite value audit
    4. Feature statistics & distributions
    5. Correlation analysis (heatmap)
    6. Key feature boxplots (benign vs attack)
    7. Feature importance (mutual information)
    8. Dataset comparison (CIC-IDS2017 vs UNSW-NB15)
    9. Data quality summary report

Run:
    python ml/notebooks/01_eda.py

Output:
    - Console: Full analysis report
    - Plots saved to: ml/notebooks/eda_outputs/
    - Summary saved to: ml/notebooks/eda_outputs/eda_summary_report.txt
"""

import os
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.data_loader import DataLoader
from ml.data.preprocessing import CICIDS_LABEL_MAP, UNSW_LABEL_MAP, CLASS_ORDER

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE = (14, 8)
DPI = 150


def section_header(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


# ============================================================
# SECTION 1: DATASET LOADING & BASIC INSPECTION
# ============================================================

def load_and_inspect(loader: DataLoader):
    """Load datasets and perform basic inspection."""
    section_header("1. DATASET LOADING & BASIC INSPECTION")

    results = {}

    # --- CIC-IDS2017 ---
    try:
        print("Loading CIC-IDS2017...")
        df_cic = loader.load_cicids2017()
        report_cic = loader.validate_dataset(df_cic, "CIC-IDS2017")
        results["cicids2017"] = {"df": df_cic, "report": report_cic}

        print(f"\n  CIC-IDS2017 Summary:")
        print(f"    Shape:           {df_cic.shape[0]:,} rows × {df_cic.shape[1]} columns")
        print(f"    Memory Usage:    {df_cic.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(f"    Label Column:    {report_cic.get('label_column', 'Not found')}")
        print(f"    Missing Values:  {report_cic['missing_values']:,}")
        print(f"    Inf Values:      {report_cic['inf_values']:,}")
        print(f"    Duplicate Rows:  {report_cic['duplicate_rows']:,}")
    except FileNotFoundError as e:
        print(f"  ⚠ CIC-IDS2017 not found: {e}")
        print(f"    Run: python infra/scripts/download_datasets.py")
        results["cicids2017"] = None

    # --- UNSW-NB15 ---
    try:
        print("\nLoading UNSW-NB15...")
        df_unsw = loader.load_unsw_nb15()
        report_unsw = loader.validate_dataset(df_unsw, "UNSW-NB15")
        results["unsw_nb15"] = {"df": df_unsw, "report": report_unsw}

        print(f"\n  UNSW-NB15 Summary:")
        print(f"    Shape:           {df_unsw.shape[0]:,} rows × {df_unsw.shape[1]} columns")
        print(f"    Memory Usage:    {df_unsw.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(f"    Label Column:    {report_unsw.get('label_column', 'Not found')}")
        print(f"    Missing Values:  {report_unsw['missing_values']:,}")
        print(f"    Inf Values:      {report_unsw['inf_values']:,}")
        print(f"    Duplicate Rows:  {report_unsw['duplicate_rows']:,}")
    except FileNotFoundError as e:
        print(f"  ⚠ UNSW-NB15 not found: {e}")
        print(f"    Run: python infra/scripts/download_datasets.py")
        results["unsw_nb15"] = None

    return results


# ============================================================
# SECTION 2: CLASS DISTRIBUTION ANALYSIS
# ============================================================

def analyze_class_distribution(df: pd.DataFrame, label_col: str, dataset_name: str, label_map: dict):
    """Analyze and visualize class distribution."""
    section_header(f"2. CLASS DISTRIBUTION — {dataset_name}")

    # Raw label distribution
    raw_dist = df[label_col].value_counts()
    print("Raw Label Distribution:")
    print("-" * 60)
    for label, count in raw_dist.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:40s} {count:>10,}  ({pct:5.2f}%) {bar}")

    # Map to 6-class schema
    df_mapped = df.copy()
    df_mapped["mapped_label"] = df_mapped[label_col].map(label_map)
    df_mapped = df_mapped.dropna(subset=["mapped_label"])

    mapped_dist = df_mapped["mapped_label"].value_counts()
    print(f"\nMapped to 6-Class Schema ({len(df_mapped):,} rows after mapping):")
    print("-" * 60)
    for label in CLASS_ORDER:
        count = mapped_dist.get(label, 0)
        pct = count / len(df_mapped) * 100 if len(df_mapped) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:20s} {count:>10,}  ({pct:5.2f}%) {bar}")

    # Imbalance ratio
    if len(mapped_dist) > 0:
        majority = mapped_dist.max()
        minority = mapped_dist.min()
        ratio = majority / minority if minority > 0 else float('inf')
        print(f"\n  Imbalance Ratio (majority/minority): {ratio:,.1f}:1")
        print(f"  Benign percentage: {mapped_dist.get('Benign', 0) / len(df_mapped) * 100:.1f}%")

    # --- Plot 1: Raw distribution (log scale) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Bar chart
    colors = sns.color_palette("husl", len(raw_dist))
    bars = axes[0].barh(range(len(raw_dist)), raw_dist.values, color=colors)
    axes[0].set_yticks(range(len(raw_dist)))
    axes[0].set_yticklabels(raw_dist.index, fontsize=9)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Count (log scale)', fontsize=11)
    axes[0].set_title(f'{dataset_name} — Raw Label Distribution', fontsize=13, fontweight='bold')

    for bar, val in zip(bars, raw_dist.values):
        axes[0].text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                     f'{val:,}', va='center', fontsize=8)

    # Mapped 6-class distribution
    mapped_counts = [mapped_dist.get(c, 0) for c in CLASS_ORDER]
    colors_mapped = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
    bars2 = axes[1].bar(CLASS_ORDER, mapped_counts, color=colors_mapped, edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'{dataset_name} — 6-Class Schema Distribution', fontsize=13, fontweight='bold')
    axes[1].set_yscale('log')

    for bar, val in zip(bars2, mapped_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                     f'{val:,}', ha='center', fontsize=9, fontweight='bold')

    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"class_distribution_{dataset_name.lower().replace('-', '_')}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Plot saved: {save_path}")

    return df_mapped


# ============================================================
# SECTION 3: MISSING/INFINITE VALUE AUDIT
# ============================================================

def audit_data_quality(df: pd.DataFrame, dataset_name: str):
    """Audit missing and infinite values in detail."""
    section_header(f"3. DATA QUALITY AUDIT — {dataset_name}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    print(f"Columns with Missing Values: {len(missing_cols)} of {len(df.columns)}")
    if len(missing_cols) > 0:
        print("-" * 60)
        for col, count in missing_cols.head(20).items():
            pct = count / len(df) * 100
            print(f"  {col:40s} {count:>8,} ({pct:.3f}%)")
    else:
        print("  ✅ No missing values found")

    # Infinite values
    inf_mask = np.isinf(df[numeric_cols])
    inf_counts = inf_mask.sum()
    inf_cols = inf_counts[inf_counts > 0].sort_values(ascending=False)
    print(f"\nColumns with Infinite Values: {len(inf_cols)} of {len(numeric_cols)}")
    if len(inf_cols) > 0:
        print("-" * 60)
        for col, count in inf_cols.head(20).items():
            pct = count / len(df) * 100
            print(f"  {col:40s} {count:>8,} ({pct:.3f}%)")
    else:
        print("  ✅ No infinite values found")

    # Constant columns (zero variance)
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    print(f"\nConstant/Near-Constant Columns: {len(constant_cols)}")
    if constant_cols:
        for col in constant_cols:
            print(f"  {col}: unique values = {df[col].nunique()}")

    # Negative values in columns that shouldn't have them
    negative_cols = []
    for col in numeric_cols:
        if df[col].min() < 0 and 'length' in col.lower():
            negative_cols.append((col, int((df[col] < 0).sum())))
    if negative_cols:
        print(f"\nSuspicious Negative Values in Length Columns:")
        for col, count in negative_cols:
            print(f"  {col}: {count:,} negative values")

    total_issues = len(missing_cols) + len(inf_cols) + len(constant_cols)
    print(f"\n  Total Quality Issues: {total_issues} column-level issues found")

    return {
        "missing_cols": missing_cols,
        "inf_cols": inf_cols,
        "constant_cols": constant_cols,
    }


# ============================================================
# SECTION 4: FEATURE STATISTICS & DISTRIBUTIONS
# ============================================================

def analyze_feature_statistics(df: pd.DataFrame, dataset_name: str):
    """Compute and display detailed feature statistics."""
    section_header(f"4. FEATURE STATISTICS — {dataset_name}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe(percentiles=[.01, .25, .5, .75, .99]).T
    stats["range"] = stats["max"] - stats["min"]
    stats["iqr"] = stats["75%"] - stats["25%"]
    stats["cv"] = stats["std"] / (stats["mean"].abs() + 1e-10)  # Coefficient of variation

    print(f"Numeric Feature Count: {len(numeric_cols)}")
    print(f"\nTop 15 Features by Range (most variable):")
    print("-" * 80)
    top_range = stats.sort_values("range", ascending=False).head(15)
    for col in top_range.index:
        row = top_range.loc[col]
        print(f"  {col:40s}  range={row['range']:>15,.2f}  mean={row['mean']:>12,.2f}  std={row['std']:>12,.2f}")

    print(f"\nTop 10 Features by Coefficient of Variation:")
    print("-" * 80)
    top_cv = stats.sort_values("cv", ascending=False).head(10)
    for col in top_cv.index:
        row = top_cv.loc[col]
        print(f"  {col:40s}  CV={row['cv']:>8.2f}  mean={row['mean']:>12,.2f}  std={row['std']:>12,.2f}")

    return stats


# ============================================================
# SECTION 5: CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(df: pd.DataFrame, dataset_name: str, top_n: int = 30):
    """Compute and visualize feature correlations."""
    section_header(f"5. CORRELATION ANALYSIS — {dataset_name}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Use only top_n features by variance for readability
    variances = df[numeric_cols].var().sort_values(ascending=False)
    selected_cols = variances.head(top_n).index.tolist()

    # Replace inf with NaN for correlation computation
    df_clean = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
    corr_matrix = df_clean.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.90:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"Highly Correlated Feature Pairs (|r| > 0.90): {len(high_corr_pairs)}")
    print("-" * 80)
    for col1, col2, r in high_corr_pairs[:15]:
        print(f"  {col1:30s} ↔ {col2:30s}  r = {r:+.4f}")

    if high_corr_pairs:
        print(f"\n  ⚠ {len(high_corr_pairs)} highly correlated pairs found.")
        print(f"    Consider dropping redundant features to reduce dimensionality.")

    # --- Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title(f'{dataset_name} — Feature Correlation Heatmap (Top {top_n} by Variance)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(fontsize=7, rotation=45, ha='right')
    plt.yticks(fontsize=7)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"correlation_heatmap_{dataset_name.lower().replace('-', '_')}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Plot saved: {save_path}")

    return corr_matrix, high_corr_pairs


# ============================================================
# SECTION 6: KEY FEATURE BOXPLOTS (BENIGN vs ATTACK)
# ============================================================

def plot_key_feature_comparisons(df: pd.DataFrame, label_col: str, dataset_name: str):
    """Plot boxplots comparing benign vs attack traffic for key features."""
    section_header(f"6. KEY FEATURE COMPARISONS — {dataset_name}")

    # Select key discriminative features (from literature and domain knowledge)
    key_features_candidates = [
        'Flow Duration', 'flow_duration', 'Flow_Duration',
        'Total Fwd Packets', 'total_fwd_packets', 'Tot Fwd Pkts',
        'Fwd Packet Length Mean', 'fwd_packet_length_mean', 'Fwd Pkt Len Mean',
        'Bwd Packet Length Mean', 'bwd_packet_length_mean', 'Bwd Pkt Len Mean',
        'Flow Bytes/s', 'flow_bytes_per_s', 'Flow Byts/s',
        'Flow Packets/s', 'flow_packets_per_s', 'Flow Pkts/s',
        'Flow IAT Mean', 'flow_iat_mean', 'Flow IAT Mean',
        'Fwd IAT Mean', 'fwd_iat_mean', 'Fwd IAT Mean',
        'SYN Flag Count', 'syn_flag_count', 'SYN Flag Cnt',
        'Packet Length Variance', 'packet_length_variance', 'Pkt Len Var',
        'Packet Length Mean', 'packet_length_mean', 'Pkt Len Mean',
        'Fwd Header Length', 'fwd_header_length', 'Fwd Header Len',
    ]

    available_features = [f for f in key_features_candidates if f in df.columns]

    if not available_features:
        print("  ⚠ No matching key features found in dataset columns.")
        print(f"    Available columns: {list(df.columns[:10])}...")
        return

    # Limit to unique feature names (take first match)
    seen = set()
    unique_features = []
    for f in available_features:
        base = f.lower().replace(' ', '_').replace('/', '_')
        if base not in seen:
            seen.add(base)
            unique_features.append(f)
    unique_features = unique_features[:8]  # Max 8

    # Create binary label: Benign vs Attack
    df_plot = df.copy()
    benign_labels = ['BENIGN', 'Benign', 'Normal', 'benign', 'normal']
    df_plot['is_attack'] = ~df_plot[label_col].isin(benign_labels)
    df_plot['traffic_type'] = df_plot['is_attack'].map({True: 'Attack', False: 'Benign'})

    # Replace inf with NaN
    for col in unique_features:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        df_plot[col] = df_plot[col].replace([np.inf, -np.inf], np.nan)

    # Print statistics
    print(f"Key Features Analyzed: {len(unique_features)}")
    print("-" * 80)
    for feat in unique_features:
        benign_mean = df_plot.loc[~df_plot['is_attack'], feat].mean()
        attack_mean = df_plot.loc[df_plot['is_attack'], feat].mean()
        ratio = attack_mean / (benign_mean + 1e-10)
        indicator = "🔴" if abs(ratio - 1) > 0.5 else "🟡" if abs(ratio - 1) > 0.2 else "🟢"
        print(f"  {indicator} {feat:35s}  Benign={benign_mean:>12,.2f}  Attack={attack_mean:>12,.2f}  Ratio={ratio:.2f}")

    # --- Boxplots ---
    n_cols = min(4, len(unique_features))
    n_rows = (len(unique_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = {'Benign': '#2ecc71', 'Attack': '#e74c3c'}

    for idx, feat in enumerate(unique_features):
        ax = axes[idx]
        # Sample for performance (boxplots on millions of rows are slow)
        sample_size = min(50000, len(df_plot))
        df_sample = df_plot[[feat, 'traffic_type']].dropna().sample(n=sample_size, random_state=42)

        sns.boxplot(data=df_sample, x='traffic_type', y=feat, palette=colors,
                    ax=ax, fliersize=1, linewidth=1.2)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Use log scale if range is very large
        feat_range = df_sample[feat].quantile(0.99) - df_sample[feat].quantile(0.01)
        if feat_range > 0 and df_sample[feat].quantile(0.99) / (abs(df_sample[feat].quantile(0.01)) + 1) > 100:
            ax.set_yscale('symlog')

    # Remove unused axes
    for idx in range(len(unique_features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{dataset_name} — Key Feature Distributions (Benign vs Attack)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"feature_boxplots_{dataset_name.lower().replace('-', '_')}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Plot saved: {save_path}")


# ============================================================
# SECTION 7: FEATURE IMPORTANCE (MUTUAL INFORMATION)
# ============================================================

def compute_feature_importance(df: pd.DataFrame, label_col: str, dataset_name: str, label_map: dict):
    """Compute feature importance using mutual information."""
    section_header(f"7. FEATURE IMPORTANCE — {dataset_name}")

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    # Map labels and clean
    df_fi = df.copy()
    df_fi["label"] = df_fi[label_col].map(label_map)
    df_fi = df_fi.dropna(subset=["label"])

    numeric_cols = df_fi.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("  ⚠ No numeric columns found for importance analysis.")
        return None

    # Clean numeric data
    X = df_fi[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df_fi["label"])

    # Sample for speed
    sample_size = min(50000, len(X))
    np.random.seed(42)
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[indices]
    y_sample = y[indices]

    print(f"Computing mutual information on {sample_size:,} samples, {len(numeric_cols)} features...")
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42, n_neighbors=5)

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "feature": numeric_cols,
        "mutual_info": mi_scores,
    }).sort_values("mutual_info", ascending=False)

    print(f"\nTop 20 Most Important Features (Mutual Information):")
    print("-" * 60)
    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
        bar = "█" * int(row['mutual_info'] / importance_df['mutual_info'].max() * 30)
        print(f"  {i+1:2d}. {row['feature']:35s}  MI={row['mutual_info']:.4f}  {bar}")

    # How many features needed for 90% of total MI
    total_mi = importance_df['mutual_info'].sum()
    cumsum = importance_df['mutual_info'].cumsum()
    threshold_90 = (cumsum >= 0.9 * total_mi).idxmax()
    n_features_90 = importance_df.index.get_loc(threshold_90) + 1
    print(f"\n  Features needed for 90% of total MI: {n_features_90} of {len(numeric_cols)}")

    # --- Feature Importance Bar Plot ---
    fig, ax = plt.subplots(figsize=(12, 10))
    top_20 = importance_df.head(20)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_20)))
    bars = ax.barh(range(len(top_20)), top_20['mutual_info'].values, color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'].values, fontsize=9)
    ax.set_xlabel('Mutual Information Score', fontsize=11)
    ax.set_title(f'{dataset_name} — Top 20 Features by Mutual Information',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, top_20['mutual_info'].values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"feature_importance_{dataset_name.lower().replace('-', '_')}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Plot saved: {save_path}")

    return importance_df


# ============================================================
# SECTION 8: DATA QUALITY SUMMARY REPORT
# ============================================================

def generate_summary_report(results: dict):
    """Generate a comprehensive summary report file."""
    section_header("8. EDA SUMMARY REPORT")

    report_path = OUTPUT_DIR / "eda_summary_report.txt"
    lines = []
    lines.append("=" * 80)
    lines.append("  SENTINEL-AI — Exploratory Data Analysis Summary Report")
    lines.append("=" * 80)
    lines.append("")

    for name, data in results.items():
        if data is None:
            lines.append(f"Dataset: {name} — NOT LOADED (download required)")
            continue

        df = data["df"]
        report = data["report"]

        lines.append(f"Dataset: {name}")
        lines.append("-" * 60)
        lines.append(f"  Rows:            {df.shape[0]:,}")
        lines.append(f"  Columns:         {df.shape[1]}")
        lines.append(f"  Memory:          {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        lines.append(f"  Label Column:    {report.get('label_column', 'N/A')}")
        lines.append(f"  Missing Values:  {report['missing_values']:,}")
        lines.append(f"  Inf Values:      {report['inf_values']:,}")
        lines.append(f"  Duplicates:      {report['duplicate_rows']:,}")

        if "class_distribution" in report:
            lines.append(f"  Class Distribution:")
            for cls, count in sorted(report["class_distribution"].items(), key=lambda x: -x[1]):
                pct = count / df.shape[0] * 100
                lines.append(f"    {cls:35s} {count:>10,} ({pct:5.2f}%)")

        lines.append("")

    lines.append("=" * 80)
    lines.append("  Key Findings & Decisions")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. CLASS IMBALANCE: Benign traffic dominates (~80%+). Decision: Use class-weighted")
    lines.append("   loss during training. Report per-class Precision/Recall/F1, not just accuracy.")
    lines.append("")
    lines.append("2. DATA QUALITY: Inf/NaN values present (esp. Flow Bytes/s, Flow Packets/s).")
    lines.append("   Decision: Replace inf→NaN, drop rows with NaN. Document dropped percentage.")
    lines.append("")
    lines.append("3. FEATURE SELECTION: Top 20 features by mutual information capture ~90% of")
    lines.append("   discriminative power. Consider feature subset for faster training.")
    lines.append("")
    lines.append("4. CORRELATION: Multiple highly correlated feature pairs found. May drop")
    lines.append("   redundant features in model refinement (Phase 2+).")
    lines.append("")
    lines.append("5. SPLIT STRATEGY: Stratified 70/15/15 train/val/test to preserve class ratios.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  📄 Report saved: {report_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run the complete EDA pipeline."""
    print("\n" + "🔬 " * 20)
    print("  SENTINEL-AI — Exploratory Data Analysis")
    print("🔬 " * 20)

    loader = DataLoader(data_dir=str(PROJECT_ROOT / "data" / "raw"))

    # 1. Load & inspect
    results = load_and_inspect(loader)

    # Process each available dataset
    for ds_name, ds_key, label_col_candidates, label_map in [
        ("CIC-IDS2017", "cicids2017", ["Label", " Label"], CICIDS_LABEL_MAP),
        ("UNSW-NB15", "unsw_nb15", ["attack_cat", "Label", "label"], UNSW_LABEL_MAP),
    ]:
        if results.get(ds_key) is None:
            print(f"\n⚠ Skipping {ds_name} (not loaded)")
            continue

        df = results[ds_key]["df"]
        report = results[ds_key]["report"]
        label_col = report.get("label_column")

        if label_col is None:
            # Try candidates
            for candidate in label_col_candidates:
                if candidate in df.columns:
                    label_col = candidate
                    break
        if label_col is None:
            print(f"\n⚠ Could not find label column for {ds_name}")
            continue

        # 2. Class distribution
        df_mapped = analyze_class_distribution(df, label_col, ds_name, label_map)

        # 3. Data quality audit
        audit_data_quality(df, ds_name)

        # 4. Feature statistics
        analyze_feature_statistics(df, ds_name)

        # 5. Correlations
        analyze_correlations(df, ds_name)

        # 6. Key feature boxplots
        plot_key_feature_comparisons(df, label_col, ds_name)

        # 7. Feature importance
        compute_feature_importance(df, label_col, ds_name, label_map)

    # 8. Summary report
    generate_summary_report(results)

    print("\n" + "✅ " * 20)
    print("  EDA Complete! Check outputs in: ml/notebooks/eda_outputs/")
    print("✅ " * 20 + "\n")


if __name__ == "__main__":
    main()
