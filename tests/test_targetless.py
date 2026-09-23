import pandas as pd
from yaeda import TabularEDA
from yaeda.correlation import FeatureTargetAnalyzer
from yaeda.clustering import TabularClusterAnalyzer
from yaeda.feature_importance import FeatureImportanceAnalyzer
from yaeda.interaction import FeatureInteractionAnalyzer


def test_targetless_correlation_analyzer(classification_df):
    df_no_target = classification_df.drop(columns=["target"])
    analyzer = FeatureTargetAnalyzer(df_no_target, target=None)
    report = analyzer.run()

    assert report.target is None
    assert report.target_type is None
    assert report.target_associations == {}
    assert len(report.pearson_matrix) > 0
    assert len(report.collinear_pairs) >= 1


def test_targetless_clustering(classification_df):
    df_no_target = classification_df.drop(columns=["target"])
    analyzer = TabularClusterAnalyzer(df_no_target, target=None, n_clusters=3)
    report = analyzer.run()

    assert report.target is None
    assert report.mutual_info_with_target is None
    assert report.n_clusters == 3
    assert len(report.clusters) == 3
    for c in report.clusters:
        assert c.target_mean is None
        assert c.target_distribution == {}
        assert len(c.defining_features) > 0


def test_targetless_feature_importance_and_interaction(classification_df):
    df_no_target = classification_df.drop(columns=["target"])

    fi_analyzer = FeatureImportanceAnalyzer(df_no_target, target=None)
    fi_rep = fi_analyzer.run()
    assert fi_rep.target is None
    assert fi_rep.importances == []

    inter_analyzer = FeatureInteractionAnalyzer(
        df_no_target, target=None, features=["feat_num1", "feat_skewed"]
    )
    inter_rep = inter_analyzer.run()
    assert inter_rep.target is None
    assert inter_rep.top_interactions == []


def test_targetless_tabular_eda_end_to_end(classification_df, tmp_path):
    df_no_target = classification_df.drop(columns=["target"])
    eda = TabularEDA(df=df_no_target, target=None, n_clusters=[2, 3])

    assert eda.has_target is False
    assert eda.diagnostics is None

    # Profiling works
    stats = eda.stats
    assert stats.target_column is None
    assert len(stats.features) == 4

    # Unsupervised feature selection works
    prominent = eda.select_prominent_features(top_n=3)
    assert len(prominent) == 3

    # Clustering without target works
    clusters = eda.clusters
    assert len(clusters) == 2

    # HTML and JSON output export without errors
    json_path = tmp_path / "targetless.json"
    data = eda.to_json(output=json_path)
    assert json_path.exists()
    assert data["metadata"]["dataset_name"] == "Primary"
    assert data["metadata"]["target_column"] is None
    assert data["metadata"]["target_type"] is None
