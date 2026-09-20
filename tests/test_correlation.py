from yaeda.correlation import FeatureTargetAnalyzer


def test_correlation_and_collinearity(classification_df):
    analyzer = FeatureTargetAnalyzer(
        df=classification_df,
        target="target",
        collinear_threshold=0.80,
    )
    report = analyzer.run()

    assert report.target == "target"
    assert report.target_type == "classification"

    # Verify collinearity detection
    assert len(report.collinear_pairs) >= 1
    pair = report.collinear_pairs[0]
    collinear_names = {pair.feature_a, pair.feature_b}
    assert "feat_num1" in collinear_names
    assert "feat_collinear" in collinear_names
    assert abs(pair.pearson_corr) >= 0.80

    # Verify Mutual Information calculation
    assoc = report.target_associations["feat_num1"]
    assert assoc.mutual_info is not None
    assert assoc.mutual_info >= 0.0


def test_correlation_regression(regression_df):
    analyzer = FeatureTargetAnalyzer(
        df=regression_df,
        target="target_price",
        target_type="regression",
    )
    report = analyzer.run()

    assert report.target_type == "regression"
    assoc1 = report.target_associations["feat_num1"]
    assert assoc1.pearson_corr > 0.5  # Strong positive slope in synthetic fixture