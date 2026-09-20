import numpy as np
import pandas as pd
from yaeda.extract import TabularDataProfiler


def test_profiler_numerical_and_categorical(classification_df):
    profiler = TabularDataProfiler(classification_df, target="target")
    profile = profiler.run()

    assert profile.n_rows == 250
    assert profile.n_columns == 5
    assert profile.target_column == "target"

    # Check numeric stats
    feat_num1 = profile.features["feat_num1"]
    assert feat_num1.is_numeric is True
    assert feat_num1.missing_count == 11
    assert feat_num1.zero_count > 0
    assert feat_num1.min_value is not None
    assert feat_num1.median is not None
    assert feat_num1.iqr is not None

    # Check outlier detection on skewed feature
    feat_skewed = profile.features["feat_skewed"]
    assert feat_skewed.outliers is not None
    assert feat_skewed.outliers.count >= 1
    assert feat_skewed.skewness > 1.0

    # Check categorical stats
    feat_cat = profile.features["feat_cat"]
    assert feat_cat.is_numeric is False
    assert feat_cat.distinct_count == 3
    assert feat_cat.mode in ["TypeA", "TypeB", "TypeC"]
    assert len(feat_cat.most_frequent) == 3


def test_profiler_degenerate_cases():
    df = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan],
            "single_value": [1.0, 1.0, 1.0],
            "target": [0, 1, 0],
        }
    )
    profiler = TabularDataProfiler(df, target="target")
    profile = profiler.run()

    assert profile.features["all_nan"].missing_count == 3
    assert profile.features["single_value"].std_dev == 0.0
