import pytest
from yaeda import TabularEDA
from yaeda.feature_importance import HAS_LIGHTGBM, FeatureImportanceAnalyzer


def test_engine_extra_trees(classification_df):
    analyzer = FeatureImportanceAnalyzer(
        df=classification_df,
        target="target",
        model_engine="extra_trees",
    )
    rep = analyzer.run()
    assert "ExtraTrees" in rep.model_type
    assert len(rep.importances) > 0
    assert rep.fitted_model is not None


def test_engine_random_forest(classification_df):
    analyzer = FeatureImportanceAnalyzer(
        df=classification_df,
        target="target",
        model_engine="random_forest",
    )
    rep = analyzer.run()
    assert "RandomForest" in rep.model_type
    assert len(rep.importances) > 0


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
def test_engine_lightgbm(classification_df):
    analyzer = FeatureImportanceAnalyzer(
        df=classification_df,
        target="target",
        model_engine="lightgbm",
    )
    rep = analyzer.run()
    assert "LGBM" in rep.model_type
    assert len(rep.importances) > 0
    assert rep.importances[0].golden_feature_score is not None


def test_tabular_eda_auto_engine_end_to_end(classification_df, tmp_path):
    eda = TabularEDA(
        df=classification_df,
        target="target",
        model_engine="auto",
    )
    fi = eda.feature_importance
    assert len(fi.importances) > 0

    # PDP and HTML generation work with the selected engine
    html_out = tmp_path / "engine_test.html"
    eda.to_html(output=html_out)
    assert html_out.exists()
