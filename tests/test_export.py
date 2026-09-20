from yaeda.clustering import TabularClustersCall
from yaeda.correlation import FeatureTargetAnalyzer
from yaeda.exports.export import StructuredDataExporter
from yaeda.extract import TabularDataProfiler
from yaeda.feature_importance import FeatureImportanceAnalyzer
from yaeda.model_diagnostics import ModelDiagnosticsCall, ModelDiagnosticsInputs


def test_structured_export_with_lists(classification_df, fitted_binary_model, tmp_path):
    model, y_pred, y_prob = fitted_binary_model

    # 1. Base analyzers on the full 250-row dataframe
    stats = TabularDataProfiler(classification_df, target="target").run()
    corr = FeatureTargetAnalyzer(classification_df, target="target").run()
    fi = FeatureImportanceAnalyzer(classification_df, target="target", random_state=42).run()

    # 2. Clusters report list
    cluster_dict = TabularClustersCall(classification_df, target="target", n_clusters=[2, 3]).run()
    cluster_reports = list(cluster_dict.values())

    # 3. Model diagnostics report list
    diag_inputs = [
        ModelDiagnosticsInputs(
            predictions=y_pred, probabilities=y_prob, model=model, name="ModelA"
        ),
        ModelDiagnosticsInputs(predictions=y_pred, name="ModelB"),
    ]
    diag_dict = ModelDiagnosticsCall(classification_df, target="target", inputs=diag_inputs).run()
    diag_reports = list(diag_dict.values())

    # 4. Instantiate exporter
    exporter = StructuredDataExporter(
        table_profile=stats,
        corr_report=corr,
        importance_report=fi,
        cluster_report=cluster_reports,
        diagnostic_report=diag_reports,
    )

    # Test JSON export
    json_path = tmp_path / "comprehensive_summary.json"
    payload = exporter.export_json(json_path)

    assert json_path.exists()
    assert isinstance(payload["cluster_analysis"], list)
    assert len(payload["cluster_analysis"]) == 2
    assert isinstance(payload["model_diagnostics"], list)
    assert len(payload["model_diagnostics"]) == 2
    assert payload["model_diagnostics"][0]["name"] == "ModelA"
    assert json_path.stat().st_size < 500 * 1024

    # Test CSV export
    csv_dir = tmp_path / "csv_out"
    csv_dict = exporter.export_csvs(csv_dir)
    assert (csv_dir / "cluster_profiles.csv").exists()
    assert "cluster_profiles" in csv_dict
