from yaeda.clustering import (
    ClusterReport,
    TabularClusterAnalyzer,
    TabularClustersCall,
)


def test_single_tabular_cluster_analyzer(classification_df):
    analyzer = TabularClusterAnalyzer(
        df=classification_df,
        target="target",
        features=["feat_num1", "feat_skewed"],
        target_type="classification",
        n_clusters=3,
        random_state=42,
    )
    report = analyzer.run()

    assert isinstance(report, ClusterReport)
    assert report.n_clusters == 3
    assert len(report.clusters) == 3
    assert len(report.cluster_labels) == len(classification_df)
    assert len(report.pca_coordinates) == len(classification_df)
    assert -1.0 <= report.silhouette_score <= 1.0

    cluster_0 = report.clusters[0]
    assert cluster_0.size > 0
    assert cluster_0.percentage > 0.0
    assert len(cluster_0.defining_features) > 0


def test_tabular_clusters_call_multi_dimensions(classification_df):
    call = TabularClustersCall(
        df=classification_df,
        target="target",
        features=["feat_num1", "feat_skewed"],
        target_type="classification",
        n_clusters=[2, 4],
        random_state=42,
    )

    analyzers = call.analyzers()
    assert set(analyzers.keys()) == {2, 4}

    reports = call.run()
    assert isinstance(reports, dict)
    assert set(reports.keys()) == {2, 4}
    assert reports[2].n_clusters == 2
    assert reports[4].n_clusters == 4
    assert len(reports[2].clusters) == 2
    assert len(reports[4].clusters) == 4


def test_tabular_clusters_call_deduplication_and_minimum(classification_df):
    # Cluster counts < 2 must be clamped to 2, and duplicates merged
    call = TabularClustersCall(
        df=classification_df,
        target="target",
        n_clusters=[1, 2, 2, 5],
        random_state=42,
    )

    analyzers = call.analyzers()
    assert list(analyzers.keys()) == [2, 5]


def test_clustering_regression(regression_df):
    call = TabularClustersCall(
        df=regression_df,
        target="target_price",
        features=["feat_num1", "feat_num2"],
        target_type="regression",
        n_clusters=[3],
        random_state=42,
    )
    reports = call.run()

    assert 3 in reports
    report = reports[3]
    assert report.target_type == "regression"
    for c in report.clusters:
        assert c.target_mean is not None
