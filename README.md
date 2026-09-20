# yaEDA: Yet Another EDA 🚀

[![CI](https://github.com/rnascunha/yaEDA/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/yaEDA/actions)
[![PyPI version](https://img.shields.io/pypi/v/yaeda.svg)](https://pypi.org/project/yaeda/)
[![Python versions](https://img.shields.io/pypi/pyversions/yaeda.svg)](https://pypi.org/project/yaeda/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**yaEDA** (Yet Another EDA) is an automated exploratory data analysis, feature intelligence, and model error diagnostics engine built for competitive tabular data science and machine learning.

---

## Features

- **Golden Feature Ranking**: Combines tree MDI, out-of-sample permutation drop, and Mutual Information into a composite rank score.
- **Partial Dependence & ICE Curves**: Evaluates individual non-linear response curves across primary predictors.
- **Arithmetic Synergy Engine**: Evaluates combinations ($A \times B$, $A / B$, $A \pm B$) to identify feature interactions exceeding baseline performance.
- **KMeans & PCA Clustering**: Identifies distinct clusters and measures target alignment using mutual information.
- **Model Error Diagnostics**: Partitions false positive/negative cohorts and residual tails with global beeswarms and local SHAP waterfall plots.
- **Self-Contained Dashboard**: Exports standalone HTML reports with zero runtime dependencies.

---

## Installation

```bash
# Core package
$ pip install yaeda

# With SHAP support
$ pip install "yaeda[shap]"

# With PDF export (WeasyPrint)
$ pip install "yaeda[pdf]"
```

Using `uv`:

```bash
$ uv add yaeda --extra shap
```

## Quickstart

```python
import pandas as pd
from yaeda import TabularEDA

df = pd.read_csv("train.csv")

eda = TabularEDA(
    df=df,
    target="target_col",
    target_type="classification",
)

# Generate offline HTML dashboard
eda.to_html("report.html")

# Export structured intelligence as JSON
eda.to_json("metadata.json")
```

## Development Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/yaEDA.git](https://github.com/your-username/yaEDA.git)
cd yaEDA

# Create virtual environment and sync dependencies using uv
uv sync --extra dev --extra all

# Run test suite
uv run pytest

# Run linting
uv run ruff check .
```
