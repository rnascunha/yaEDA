import json
from yaeda import TabularEDA


def test_analyze_end_to_end_classification_with_options(
    classification_df, fitted_binary_model, tmp_path
):
    model, y_pred, y_prob = fitted_binary_model

    # Pass diagnostics as list of dicts conforming to ModelDiagnosticsInputs
    diag_input_dicts = [
        {
            "predictions": y_pred,
            "probabilities": y_prob,
            "model": model,
            "name": "RF_Ensemble",
        },
        {
            "predictions": y_pred,
            "name": "Baseline_Rule",
        },
    ]

    eda = TabularEDA(
        df=classification_df,
        target="target",
        diagnostics=diag_input_dicts,
        n_clusters=[2, 4],
        target_type="classification",
        seed=42,
    )

    # 1. Verify clusters property returns list[ClusterReport]
    clusters = eda.clusters
    assert isinstance(clusters, list)
    assert len(clusters) == 2
    assert clusters[0].n_clusters == 2
    assert clusters[1].n_clusters == 4

    # 2. Verify diagnostics property returns list[ModelDiagnosticsReport]
    diagnostics = eda.diagnostics
    assert isinstance(diagnostics, list)
    assert len(diagnostics) == 2
    assert diagnostics[0].name == "RF_Ensemble"
    assert diagnostics[1].name == "Baseline_Rule"

    # 3. Verify prominent features selection
    prominent = eda.select_prominent_features(top_n=3, include_auxiliary=True)
    assert isinstance(prominent, list)
    assert len(prominent) <= 3

    # 4. Verify HTML generation
    html_file = tmp_path / "dashboard.html"
    html_content = eda.to_html(
        output=html_file,
        top_n_features=4,
        top_n_interactions=8,
    )

    assert html_file.exists()
    assert "<!DOCTYPE html>" in html_content

    # 5. Verify JSON export
    json_file = tmp_path / "metadata.json"
    _json_payload = eda.to_json(output=json_file)

    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert isinstance(loaded["cluster_analysis"], list)
    assert len(loaded["cluster_analysis"]) == 2
    assert isinstance(loaded["model_diagnostics"], list)
    assert len(loaded["model_diagnostics"]) == 2


def test_analyze_defaults_without_diagnostics(classification_df):
    eda = TabularEDA(
        df=classification_df,
        target="target",
        target_type="classification",
    )

    # When no diagnostics are provided, returns None
    assert eda.diagnostics is None

    # Default n_clusters is [4]
    clusters = eda.clusters
    assert isinstance(clusters, list)
    assert len(clusters) == 1
    assert clusters[0].n_clusters == 4


def test_analyze_regression(regression_df, fitted_regression_model, tmp_path):
    model, y_pred = fitted_regression_model

    eda = TabularEDA(
        df=regression_df,
        target="target_price",
        diagnostics=[{"predictions": y_pred, "model": model, "name": "Regressor_A"}],
        n_clusters=[3],
        target_type="regression",
    )

    assert len(eda.clusters) == 1
    assert len(eda.diagnostics) == 1
    assert eda.diagnostics[0].target_type == "regression"

    html_file = tmp_path / "regression.html"
    eda.to_html(output=html_file)
    assert html_file.exists()
