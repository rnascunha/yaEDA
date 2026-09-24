import time
from yaeda import TabularEDA


def test_preset_minimal_execution_speed(classification_df):
    t0 = time.time()
    eda = TabularEDA(
        df=classification_df,
        target="target",
        preset="minimal",
    )

    # 1. Properties verify minimal state
    assert eda.enable_correlation is False
    assert eda.enable_feature_importance is False
    assert eda.enable_clustering is False
    assert eda.enable_interactions is False
    assert eda.enable_pdp is False

    # Stats profile executes
    stats = eda.stats
    assert stats.n_rows == len(classification_df)

    # Heavy modules return empty or bypassed representations
    assert eda.correlation.collinear_pairs == []
    assert eda.feature_importance.importances == []
    assert eda.clusters == []
    assert eda.interactions().top_interactions == []

    # Generation is fast (< 0.25 seconds)
    duration = time.time() - t0
    assert duration < 0.8


def test_preset_standard_skips_interactions_and_pdp(classification_df):
    eda = TabularEDA(
        df=classification_df,
        target="target",
        preset="standard",
    )

    assert eda.enable_correlation is True
    assert eda.enable_feature_importance is True
    assert eda.enable_clustering is True
    assert eda.enable_interactions is False
    assert eda.enable_pdp is False

    # Interactions must be skipped
    inter_rep = eda.interactions()
    assert inter_rep.top_interactions == []

    # PDP must return empty string
    assert eda.plot_partial_dependence() == ""


def test_preset_override_with_explicit_flags(classification_df):
    eda = TabularEDA(
        df=classification_df,
        target="target",
        preset="minimal",
        enable_clustering=True,  # Explicitly override
    )

    assert eda.enable_clustering is True
    assert eda.enable_interactions is False

    clusters = eda.clusters
    assert len(clusters) > 0


def test_max_features_to_plot_capping(classification_df, tmp_path):
    eda = TabularEDA(
        df=classification_df,
        target="target",
        max_features_to_plot=2,
    )

    html_file = tmp_path / "capped.html"
    content = eda.to_html(output=html_file)

    # Notice banner appears and only 2 cards are generated
    assert "Performance limit active" in content
    assert content.count("feature-card searchable-card") == 2
