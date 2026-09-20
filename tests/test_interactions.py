import numpy as np
from yaeda.interaction import FeatureInteractionAnalyzer


def test_arithmetic_interactions(classification_df):
    analyzer = FeatureInteractionAnalyzer(
        df=classification_df,
        target="target",
        features=["feat_num1", "feat_skewed"],
        target_type="classification",
    )
    report = analyzer.run()

    assert len(report.top_interactions) == 5  # 5 operations: *, A/B, B/A, +, -
    assert len(report.best_per_pair) == 1

    best_pair = report.best_per_pair[0]
    assert hasattr(best_pair, "synergy_gain")
    assert hasattr(best_pair, "formula")
    assert best_pair.operation in [
        "Multiplication",
        "Ratio (A/B)",
        "Ratio (B/A)",
        "Sum",
        "Difference",
    ]


def test_zero_denominator_safe(classification_df):
    # feat_num1 has zeros in rows 20:35
    analyzer = FeatureInteractionAnalyzer(
        df=classification_df,
        target="target",
        features=["feat_num1", "feat_skewed"],
    )
    report = analyzer.run()
    # Ensure no NaN/Inf caused a failure during synergy calculation
    for item in report.top_interactions:
        assert not np.isnan(item.combined_score)
        assert not np.isnan(item.synergy_gain)
