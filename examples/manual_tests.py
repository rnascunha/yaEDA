"""Manual verification script for yaEDA exports across multiple execution modes.

Executes TabularEDA across 4 scenarios:
1. With Target, Without Secondary Dataset (Standard Supervised)
2. With Target, With Secondary Dataset    (Supervised + Multi-dataset comparison)
3. Without Target, Without Secondary     (Unsupervised Single Dataset)
4. Without Target, With Secondary        (Unsupervised Multi-dataset comparison)
"""

import json
from pathlib import Path
import time
import pandas as pd
from yaeda import TabularEDA

# ==============================================================================
# Configuration: Adjust paths and parameters here
# ==============================================================================
BASE_INPUT_DIR_PATH = Path("./artifacts/data/")
TRAIN_CSV_PATH = BASE_INPUT_DIR_PATH / "train.csv"
TEST_CSV_PATH = BASE_INPUT_DIR_PATH / "test.csv"
TARGET_COL = "Will_Buy_EV"
OUTPUT_DIR = Path("./artifacts/exports/")

# If True and the files above are not found, creates synthetic sample files to run
CREATE_SAMPLE_DATA_IF_MISSING = True
# ==============================================================================


def ensure_sample_data(train_path: Path, test_path: Path, target_col: str):
    """Generates synthetic classification CSVs if configured paths do not exist."""
    import numpy as np

    if train_path.exists() and test_path.exists():
        return

    print("⚠️  Configured CSVs not found. Generating synthetic demo datasets...")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_train = 300
    n_test = 150

    # Train Data
    t_num1 = np.random.normal(10, 2, n_train)
    t_num2 = t_num1 * 0.8 + np.random.normal(0, 0.5, n_train)
    t_cat = np.random.choice(["Low", "Medium", "High"], size=n_train, p=[0.4, 0.4, 0.2])
    logits = 0.8 * t_num1 - 0.5 * t_num2 + (t_cat == "High") * 1.5
    prob = 1 / (1 + np.exp(-logits + np.median(logits)))
    target = (prob > 0.5).astype(int)

    train_df = pd.DataFrame(
        {
            "feature_a": t_num1,
            "feature_b": t_num2,
            "category": t_cat,
            target_col: target,
        }
    )
    train_df.loc[:15, "feature_a"] = None  # Missingness

    # Test Data (with missingness drift and unseen categorical value)
    s_num1 = np.random.normal(11.5, 2.5, n_test)
    s_num2 = s_num1 * 0.75 + np.random.normal(0, 0.6, n_test)
    s_cat = np.random.choice(
        ["Low", "Medium", "High", "Critical_Novel"], size=n_test, p=[0.3, 0.3, 0.2, 0.2]
    )

    test_df = pd.DataFrame(
        {
            "feature_a": s_num1,
            "feature_b": s_num2,
            "category": s_cat,
        }
    )
    test_df.loc[:30, "feature_a"] = None  # Higher missing rate

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"   Created: {train_path} ({len(train_df)} rows)")
    print(f"   Created: {test_path} ({len(test_df)} rows)\n")


def format_size(path: Path) -> str:
    """Returns human-readable file size."""
    size_bytes = path.stat().st_size
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def run_setup(name: str, eda: TabularEDA, out_dir: Path, file_prefix: str) -> dict:
    """Executes export and validates HTML/JSON outputs."""
    print(f"\n[{name}]")
    start_time = time.time()

    html_file = out_dir / f"{file_prefix}.html"
    json_file = out_dir / f"{file_prefix}.json"

    # 1. Generate HTML Dashboard
    t0 = time.time()
    eda.to_html(output=html_file)
    html_duration = time.time() - t0

    # 2. Generate JSON Metadata
    t1 = time.time()
    _json_data = eda.to_json(output=json_file)
    json_duration = time.time() - t1

    total_duration = time.time() - start_time

    # Validate JSON integrity
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    print(
        f"  • HTML Dashboard : {html_file.name} ({format_size(html_file)}) in {html_duration:.2f}s"
    )
    print(
        f"  • JSON Metadata  : {json_file.name} ({format_size(json_file)}) in {json_duration:.2f}s"
    )
    print(f"  • Target Recorded: {loaded.get('metadata', {}).get('target_column')}")
    print(f"  • Secondary Exists: {bool(loaded.get('secondary_datasets'))}")

    return {
        "setup": name,
        "html_path": str(html_file),
        "html_size": format_size(html_file),
        "json_path": str(json_file),
        "json_size": format_size(json_file),
        "total_time": f"{total_duration:.2f}s",
    }


def main():
    train_path = Path(TRAIN_CSV_PATH)
    test_path = Path(TEST_CSV_PATH)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if CREATE_SAMPLE_DATA_IF_MISSING:
        ensure_sample_data(train_path, test_path, TARGET_COL)

    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found at '{train_path}'")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found at '{test_path}'")

    print("=" * 70)
    print("yaEDA Export Verification Runner")
    print(f"Train Source : {train_path}")
    print(f"Test Source  : {test_path}")
    print(f"Target Column: {TARGET_COL}")
    print(f"Output Path  : {out_dir.resolve()}")
    print("=" * 70)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Secondary dataset without target column
    test_df_clean = test_df.drop(columns=[TARGET_COL], errors="ignore")

    # Primary dataframe stripped of target for unsupervised runs
    train_df_no_target = train_df.drop(columns=[TARGET_COL], errors="ignore")

    results = []

    # -------------------------------------------------------------------------
    # Setup 1: With Target, Without Secondary Dataset
    # -------------------------------------------------------------------------
    eda_1 = TabularEDA(
        df=(train_df, "Train"),
        target=TARGET_COL,
        n_clusters=[2, 4],
    )
    results.append(
        run_setup("1. Supervised (Single Dataset)", eda_1, out_dir, "1_supervised_single")
    )

    # -------------------------------------------------------------------------
    # Setup 2: With Target, With Secondary Dataset
    # -------------------------------------------------------------------------
    eda_2 = TabularEDA(
        df=(train_df, "Train"),
        target=TARGET_COL,
        secondary_dfs=[(test_df_clean, "Test")],
        n_clusters=[2, 4],
    )
    results.append(
        run_setup("2. Supervised (Train vs Test Comparison)", eda_2, out_dir, "2_supervised_multi")
    )

    # -------------------------------------------------------------------------
    # Setup 3: Without Target, Without Secondary Dataset
    # -------------------------------------------------------------------------
    eda_3 = TabularEDA(
        df=(train_df_no_target, "Train_Unlabeled"),
        target=None,
        n_clusters=[2, 3],
    )
    results.append(
        run_setup(
            "3. Unsupervised (Single Unlabeled Dataset)", eda_3, out_dir, "3_unsupervised_single"
        )
    )

    # -------------------------------------------------------------------------
    # Setup 4: Without Target, With Secondary Dataset
    # -------------------------------------------------------------------------
    eda_4 = TabularEDA(
        df=(train_df_no_target, "Cohort_A"),
        target=None,
        secondary_dfs=[(test_df_clean, "Cohort_B")],
        n_clusters=[2, 3],
    )
    results.append(
        run_setup("4. Unsupervised (Cohort A vs Cohort B)", eda_4, out_dir, "4_unsupervised_multi")
    )

    # Summary
    print("\n" + "=" * 70)
    print("Verification Completed. Summary Deliverables:")
    print("=" * 70)
    summary_df = pd.DataFrame(results)[["setup", "html_size", "json_size", "total_time"]]
    print(summary_df.to_string(index=False))
    print("\nGenerated files are ready in:", out_dir.resolve())


if __name__ == "__main__":
    # print()
    main()
