from pathlib import Path
import pandas as pd

from yaeda import TabularEDA

# ==============================================================================
# Configuration: Dataset Paths and Target Settings
# ==============================================================================
TRAIN_DATA_PATH = "artifacts/data/train.csv"
TEST_DATA_PATH = "artifacts/data/test.csv"
TARGET_COLUMN = "Will_Buy_EV"
OUTPUT_DIR = "."

# Ensure output directory exists
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Step 1: Load Primary and Secondary Datasets
# ==============================================================================
# Primary dataset (Train): Contains the target column and features to profile.
train_df = pd.read_csv(TRAIN_DATA_PATH)

# Secondary dataset (Test): Evaluated against the primary dataset to track:
# - Missingness drift (shifts in NaN proportions)
# - Distribution discrepancies (via overlaid KDE and histogram charts)
# - Unseen categorical values (categories present in Test but absent in Train)
test_df = pd.read_csv(TEST_DATA_PATH)


# ==============================================================================
# Step 2: Initialize TabularEDA
# ==============================================================================
# TabularEDA serves as the unified orchestrator for profiling, feature intelligence,
# pairwise interactions, unsupervised space clustering, and multi-dataset drift.
eda = TabularEDA(
    # --- Primary Dataset ---
    # Pass a tuple (DataFrame, "Name") to label the dataset cleanly in reports.
    df=(train_df, "Train"),
    # --- Target Variable ---
    # The column to explain, model, and score feature associations against.
    # If working with an unlabelled dataset, set target=None to run in unsupervised mode.
    target=TARGET_COLUMN,
    # --- Task Type ---
    # "classification" or "regression".
    # Leave as None to let yaEDA infer the task automatically from target cardinality.
    target_type=None,
    # --- Secondary Datasets for Drift & Parity Analysis ---
    # Accepts a list of tuples: [(DataFrame, "Dataset_Name"), ...]
    # Compares feature distributions and detects data health drift.
    secondary_dfs=[(test_df, "Test")],
    # --- Feature Subset (Optional) ---
    # List of column names to analyze. If None, all columns except target are included.
    features=None,
    # --- Unsupervised Clustering Dimensions ---
    # List of k-values for KMeans partitioning in PCA space.
    # Measures cluster separation (silhouette) and cluster-to-target alignment.
    n_clusters=[2, 4],
    # --- Data Quality Thresholds ---
    collinear_threshold=0.80,  # Correlation threshold (|r| >= 0.80) to flag redundancy
    outlier_irq_factor=1.5,  # Tukey's IQR multiplier for detecting distribution tails
    n_frequent=5,  # Number of top frequent values to record per feature
    n_extremes=5,  # Number of smallest and greatest values to record
    # --- Reproducibility Seed ---
    seed=42,
)


# ==============================================================================
# Step 3: Export Interactive HTML Dashboard
# ==============================================================================
# Generates a completely self-contained, offline HTML dashboard with zero external
# dependencies (all Matplotlib/Seaborn visual charts are embedded as Base64 strings).
#
# Sections included:
# 1. Executive Summary KPIs: Dataset dimensions, memory footprints, and drift metrics.
# 2. Dataset Comparison Tab: Side-by-side missingness drift and novel category alerts.
# 3. Feature Deep-Dive Cards: Quantile breakdowns with overlaid distribution plots.
# 4. Golden Features Leaderboard: Multi-perspective ranking (MDI + Permutation + MI).
# 5. Partial Dependence (PDP) & ICE Curves: Marginal response curves for top features.
# 6. Feature Interactions: Pairwise arithmetic synergies (A * B, A / B, A +/- B).
# 7. Cluster Profiles: 2D PCA instance projections and cluster centroid defining attributes.
html_output_file = out_dir / "eda_report.html"
eda.to_html(
    output=html_output_file,
    top_n_features=6,  # Number of prominent features to render in PDP and deep dives
    top_n_interactions=10,  # Max candidate interaction formulas displayed in tables
)
print(f"✅ HTML Dashboard generated successfully: {html_output_file.resolve()}")


# ==============================================================================
# Step 4: Export Structured JSON Metadata
# ==============================================================================
# Serializes statistical profiles, target associations, collinearity pairs,
# golden feature rankings, and secondary dataset comparisons into a structured,
# lightweight JSON payload (< 100 KB) suitable for automated pipelines and logging.
json_output_file = out_dir / "eda_summary.json"
summary_payload = eda.to_json(output=json_output_file)
print(f"✅ JSON Metadata generated successfully: {json_output_file.resolve()}")
print(f"   Root summary keys: {list(summary_payload.keys())}")
