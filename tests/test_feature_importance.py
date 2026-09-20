from yaeda.feature_importance import FeatureImportanceAnalyzer


def test_feature_importance_ranking(classification_df):
    analyzer = FeatureImportanceAnalyzer(
        df=classification_df,
        target="target",
        random_state=42,
    )
    report = analyzer.run()

    assert len(report.importances) == 4  # 4 features excluding target
    top = report.importances[0]
    assert top.rank == 1
    assert top.golden_feature_score >= report.importances[-1].golden_feature_score
    assert top.permutation_mean >= 0.0

    # Check estimator state preservation for downstream PDP / SHAP
    assert report.fitted_model is not None
    assert report.preprocessed_X is not None
    assert len(report.feature_names) == 4


def test_feature_importance_regression(regression_df):
    analyzer = FeatureImportanceAnalyzer(
        df=regression_df,
        target="target_price",
        target_type="regression",
        random_state=42,
    )
    report = analyzer.run()

    assert report.target_type == "regression"
    assert report.model_type == "RandomForestRegressor"
    top_feature = report.importances[0].feature
    assert top_feature in ["feat_num1", "feat_num2"]
