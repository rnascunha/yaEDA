from yaeda import TabularEDA
from yaeda.clustering import TabularClusterAnalyzer
from yaeda.correlation import FeatureTargetAnalyzer
from yaeda.feature_importance import FeatureImportanceAnalyzer
from yaeda.interaction import FeatureInteractionAnalyzer


def test_subsampled_correlation_mutual_info(classification_df):
    analyzer = FeatureTargetAnalyzer(
        df=classification_df,
        target="target",
        mi_sample_limit=50,  # Force downsampling from 250 rows to 50
    )
    rep = analyzer.run()
    assert len(rep.target_associations) > 0
    assert rep.target_associations["feat_num1"].mutual_info is not None


def test_subsampled_feature_importance(classification_df):
    analyzer = FeatureImportanceAnalyzer(
        df=classification_df,
        target="target",
        fit_sample_limit=60,
        permutation_sample_limit=30,
        mi_sample_limit=50,
        shap_sample_limit=20,
    )
    rep = analyzer.run()
    assert len(rep.importances) > 0
    assert rep.importances[0].golden_feature_score is not None


def test_subsampled_clustering(classification_df):
    analyzer = TabularClusterAnalyzer(
        df=classification_df,
        target="target",
        n_clusters=3,
        sample_limit=80,  # Force downsampling from 250 rows to 80
    )
    rep = analyzer.run()
    assert rep.n_clusters == 3
    assert len(rep.clusters) == 3
    # Labels and PCA coordinates must match the subsampled rows (80)
    assert len(rep.cluster_labels) == 80
    assert len(rep.pca_coordinates) == 80


def test_subsampled_interactions(classification_df):
    analyzer = FeatureInteractionAnalyzer(
        df=classification_df,
        target="target",
        features=["feat_num1", "feat_skewed"],
        sample_limit=50,
    )
    rep = analyzer.run()
    assert len(rep.top_interactions) > 0
    assert rep.top_interactions[0].combined_score is not None


def test_tabular_eda_configured_subsampling_end_to_end(classification_df, tmp_path):
    eda = TabularEDA(
        df=classification_df,
        target="target",
        fit_sample_limit=80,
        permutation_sample_limit=40,
        mi_sample_limit=60,
        shap_sample_limit=30,
        interaction_sample_limit=70,
        clustering_sample_limit=90,
    )

    # 1. Full statistics are untouched (250 rows)
    assert eda.stats.n_rows == 250

    # 2. Clusters run on subsampled rows
    clusters = eda.clusters
    assert len(clusters[0].cluster_labels) == 90

    # 3. HTML and JSON generation succeed with subsampled engines
    html_out = tmp_path / "subsampled.html"
    json_out = tmp_path / "subsampled.json"
    eda.to_html(output=html_out)
    eda.to_json(output=json_out)

    assert html_out.exists()
    assert json_out.exists()
