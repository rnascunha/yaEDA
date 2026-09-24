# yaEDA: Yet Another EDA 🚀

[![CI](https://github.com/your-username/yaEDA/actions/workflows/ci.yml/badge.svg)](https://github.com/rnascunha/yaEDA/actions)
[![PyPI version](https://img.shields.io/pypi/v/yaeda.svg)](https://pypi.org/project/yaeda/)
[![Python versions](https://img.shields.io/pypi/pyversions/yaeda.svg)](https://pypi.org/project/yaeda/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> **Active Development Status**: yaEDA is currently in active pre-1.0 development. APIs, configuration parameters, and export schemas are subject to change between minor releases. Feedback, bug reports, and pull requests are welcome!

**yaEDA** (Yet Another EDA) is an automated tabular feature intelligence, data profiling, and model error diagnostics library built specifically for competitive machine learning and tabular data workflows.

Unlike standard profiling tools that merely produce univariate histograms, yaEDA acts as an automated feature engineering assistant: it isolates predictive signals, discovers non-linear feature interactions, tracks multi-dataset distribution drift (e.g., Train vs. Test), identifies unseen categorical levels, segments cluster spaces, and dissects where and why models make mistakes.

---

## Pre-Computed Interactive Reports

Explore sample reports generated across various real-world execution modes:

| Execution Mode                     | Description                                                                                 |                   Interactive Dashboard                   |                 Machine-Readable Metadata                 |
| :--------------------------------- | :------------------------------------------------------------------------------------------ | :-------------------------------------------------------: | :-------------------------------------------------------: |
| **1. Supervised Single Dataset**   | Full supervised profiling with target column, Golden Features, PDP curves, and interactions |  [View HTML](artifacts/exports/1_supervised_single.html)  |  [View JSON](artifacts/exports/1_supervised_single.json)  |
| **2. Supervised Multi-Dataset**    | Primary (Train with target) vs. Secondary (Test unlabelled), drift tracking & KDE overlays  |  [View HTML](artifacts/exports/2_supervised_multi.html)   |  [View JSON](artifacts/exports/2_supervised_multi.json)   |
| **3. Unsupervised Single Dataset** | Unlabelled dataset exploration, feature health, and PCA cluster partitioning                | [View HTML](artifacts/exports/3_unsupervised_single.html) | [View JSON](artifacts/exports/3_unsupervised_single.json) |
| **4. Unsupervised Multi-Dataset**  | Cohort A vs. Cohort B comparative distribution, missingness shift, and novel levels         | [View HTML](artifacts/exports/4_unsupervised_multi.html)  | [View JSON](artifacts/exports/4_unsupervised_multi.json)  |

---

## Core Features & Visual Walkthrough

### 1. Multi-Dataset Drift & Parity Analysis

- **Missingness Drift Tracking**: Calculates directional delta ($\Delta$) missing percentages between Primary (Train) and Secondary (Test) splits.
- **Unseen Categorical Detection**: Automatically identifies and flags categories present in test sets that never appeared in training data.
- **Distribution Overlays**: Overlaid continuous KDE curves and grouped categorical bar charts with `[UNSEEN]` tags.

### 2. Feature Deep-Dive & Distribution Overlays

Detailed feature cards combining statistical metrics (quantiles, missingness, zero counts, outliers, skewness) with adaptive visual plots. When secondary datasets are present, cards display overlaid Train vs. Test density curves and side-by-side boxplots.

![Feature Deep Dive](artifacts/screenshots/features.png)

![Feature Table](artifacts/screenshots/features_table.png)

### 3. Golden Feature Discovery & Ranking

Combines multiple perspectives into a single composite rank score:

- **Tree Feature Importance (MDI)**
- **Out-of-Sample Permutation Drop** on validation splits
- **Non-Linear Mutual Information ($I(X; Y)$)**
- **Attribution Sensitivity** (Tree SHAP or PDP variance)

Features are partitioned into actionable tiers: **Tier 1 (Golden)**, **Tier 2 (Strong)**, **Tier 3 (Moderate)**, and **Tier 4 (Noise/Prune)**.

![Golden Features](artifacts/screenshots/golden_features.png)

### 4. Pairwise Feature Interactions & Arithmetic Synergy

Evaluates pairwise combinations ($A \times B$, $A / B$, $A + B$, $A - B$) against individual univariate baselines to surface engineered features that provide mathematical synergy gains.

![Feature Interactions](artifacts/screenshots/interactions.png)

### 5. Multicollinearity & Visual Diagnostics

Directly flags redundant collinear pairs ($\vert{}r\vert{} \ge 0.80$) across Pearson and Spearman correlations, visualizes target associations, and isolates data quality outliers.

![Visual Diagnostics](artifacts/screenshots/charts.png)

### 6. Unsupervised Cluster Profiling in PCA Space

Executes KMeans clustering across multiple candidate dimensions ($k$), evaluates silhouette separation, projects instances into 2D PCA space, profiles centroid deviations ($\sigma$ z-scores from global mean), and measures cluster-to-target mutual information.

![Cluster Analysis](artifacts/screenshots/cluster.png)

### 7. Model Error Forensics & SHAP Attribution

Pass model predictions to partition failure cohorts (False Positives, False Negatives, residual extremes). yaEDA ranks the worst errors, displays prediction confidence, and isolates distinguishing feature attributes alongside global Beeswarm and local Waterfall plots.

---

## Performance & Resource Optimization

> [!CAUTION]
> **Compute & Memory Resource Advisory**: Tabular feature intelligence tasks—especially pairwise arithmetic interactions ($\binom{D}{2} \times 5$ combinations), nearest-neighbor Mutual Information, and tree permutation loops—scale superlinearly with row count ($N$) and feature count ($D$). Running unconstrained deep diagnostics on large ($N > 100\text{k}$) or wide ($D > 50$) tables can result in high CPU utilization, memory pressure, and large HTML report files. Use the settings below to balance speed and diagnostic depth.

yaEDA is engineered with multi-tiered performance controls: basic statistical profiling (distributions, health, missingness) always runs across 100% of data via fast vectorized operations, while heavy diagnostic routines are governed by presets, subsampling limits, and parallel engines.

### 1. Execution Presets (`preset`)

Control pipeline depth with a single flag:

```python
eda = TabularEDA(df=train_df, target="target", preset="standard")
```

| Preset         | Profiling & Health | Correlation Matrix | Golden Features | Clustering (k) | Pairwise Interactions | PDP / ICE Curves | Recommended Use Case                             |
| -------------- | ------------------ | ------------------ | --------------- | -------------- | --------------------- | ---------------- | ------------------------------------------------ |
| """minimal"""  | ✅ Full            | ❌                 | ❌              | ❌             | ❌                    | ❌               | Millions of rows; sub-second health checks.      |
| """standard""" | ✅ Full            | ✅ Full            | ✅ Fast         | ✅             | ❌                    | ❌               | Default recommendation for large datasets.       |
| """deep"""     | ✅ Full            | ✅ Full            | ✅ Full         | ✅             | ✅ Full               | ✅ Full          | Deep exploration; competition feature discovery. |

> Any preset can be overridden with explicit toggles (`enable_interactions=False`, `enable_clustering=False`, `enable_pdp=False`, etc.).

### Statistical Subsampling Limits

Representative sample limits cap compute-heavy routines without sacrificing statistical significance. Set any limit to None to force 100% data usage:

- `fit_sample_limit` (default: `25_000`): Maximum rows used for training tree importance models.
- `permutation_sample_limit` (default: `10_000`): Maximum validation rows passed to permutation loss loops.
- `mi_sample_limit` (default: `25_000`): Limits sample size for $O(N \log N)$ nearest-neighbor Mutual Information queries.
- `clustering_sample_limit` (default: `30_000`): Limits rows partitioned by KMeans and projected in PCA space.
- `interaction_sample_limit` (default: `25_000`): Caps evaluation matrix when testing arithmetic synergies.
- `shap_sample_limit` (default: `500`): Samples passed to Tree SHAP matrix calculations.

### Model Engine (`model_engine`)

Select the model architecture used for feature importance and PDP curves:

- `"auto"` (default): Automatically uses `LightGBM` if installed; otherwise falls back to `ExtraTrees` for zero-dependency speed.

* `"lightgbm"`: Histogram-based gradient boosting. Delivers a $15\times - 30\times$ speedup and significantly lower RAM usage on large datasets. Install via `pip install "yaeda[fast]"`.
* `"extra_trees"`: Fast randomized decision trees via scikit-learn ($5\times - 10\times$ faster than standard Random Forest).
* `"random_forest"`: Standard scikit-learn Random Forest.

### Parallel Execution (`n_jobs`)

yaEDA parallelizes column-wise Mutual Information estimation, multi-$k$ KMeans clustering, secondary dataset profiling, and multi-model diagnostics across CPU threads:

```python
eda = TabularEDA(
    df=train_df,
    target="target",
    n_jobs=-1,  # Uses all available CPU cores (set to 1 for serial execution)
)
```

### Visual Card Limiting (`max_features_to_plot`)

For wide datasets with dozens or hundreds of columns, rendering every feature card generates hundreds of embedded visual canvases, resulting in heavy HTML files. Cap card rendering to the top predictors:

```python
eda.to_html("report.html", max_features_to_plot=20)
```

## Installation

Install using `pip`:

```bash
# Core package (lightweight, zero heavy binary dependencies)
$ pip install yaeda

# With high-speed LightGBM engine
$ pip install "yaeda[fast]"

# With SHAP model interpretability support
$ pip install "yaeda[shap]"

# Full installation (LightGBM + SHAP + WeasyPrint PDF export)
$ pip install "yaeda[all]"
```

Or add via `uv`:

```bash
$ uv add yaeda --extra allu
```

## Quickstart

### 1. Supervised Analysis with Multi-Dataset Comparison

```python
import pandas as pd
from yaeda import TabularEDA

train_df = pd.read_csv("artifacts/data/train.csv")
test_df = pd.read_csv("artifacts/data/test.csv")

eda = TabularEDA(
    # Primary dataset: Tuple of (DataFrame, "Name")
    df=(train_df, "Train"),
    target="Will_Buy_EV",
    # Secondary datasets: Automatically monitored for drift and unseen categories
    secondary_dfs=[(test_df, "Test")],
    # KMeans cluster dimensions to profile
    n_clusters=[2, 4],
)

# Export zero-dependency interactive HTML dashboard
eda.to_html("eda_report.html")

# Export compact, structured JSON metadata (< 100 KB)
eda.to_json("eda_summary.json")
```

### 2. Unsupervised / Unlabelled Dataset Profiling

```python
# target=None activates unsupervised mode
eda = TabularEDA(
    df=test_df,
    target=None,
    n_clusters=[3],
)

eda.to_html("unsupervised_report.html")
```

## Examples (`examples/`)

Runnable, standalone demonstration scripts are available in the examples/ folder:

- `examples/quickstart.py`: Minimal, commented walkthrough demonstrating data loading, initialization, clustering, and reporting.

To run an example script:

```bash
$ uv run examples/quickstart.py
```

## Development & Testing

yaEDA is managed using `uv` and tested with pytest.

```bash
# 1. Clone the repository
$ git clone [https://github.com/your-username/yaEDA.git](https://github.com/rnascunha/yaEDA.git)
$ cd yaEDA

# 2. Create virtual environment and install all dependencies
$ uv sync --extra dev --extra all

# 3. Run linting & formatting checks
$ uv run ruff check .

# 4. Execute the test suite with coverage
$ uv run pytest --cov=yaeda --cov-report=term-missing
```

## License

This project is licensed under the terms of the [MIT License](./LICENSE).
