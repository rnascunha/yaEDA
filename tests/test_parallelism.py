from yaeda import TabularEDA
from yaeda.clustering import TabularClustersCall
from yaeda.correlation import FeatureTargetAnalyzer
from yaeda.model_diagnostics import ModelDiagnosticsCall, ModelDiagnosticsInputs


def test_parallel_correlation_analyzer(classification_df):
    # Single-thread run
    a_seq = FeatureTargetAnalyzer(classification_df, target="target", n_jobs=1, random_state=42)
    rep_seq = a_seq.run()

    # Multi-thread run
    a_par = FeatureTargetAnalyzer(classification_df, target="target", n_jobs=-1, random_state=42)
    rep_par = a_par.run()

    for feat in a_seq.features:
        mi_s = rep_seq.target_associations[feat].mutual_info
        mi_p = rep_par.target_associations[feat].mutual_info
        assert mi_s is not None and mi_p is not None
        assert abs(mi_s - mi_p) < 1e-4


def test_parallel_clusters_call(classification_df):
    call_par = TabularClustersCall(
        df=classification_df,
        target="target",
        n_clusters=[2, 3, 4],
        n_jobs=-1,
        random_state=42,
    )
    results = call_par.run()
    assert len(results) == 3
    assert set(results.keys()) == {2, 3, 4}
    for k, rep in results.items():
        assert rep.n_clusters == k
        assert len(rep.clusters) == k


def test_parallel_multi_model_diagnostics(classification_df, fitted_binary_model):
    model, y_pred, y_prob = fitted_binary_model
    inputs = [
        ModelDiagnosticsInputs(
            predictions=y_pred, probabilities=y_prob, model=model, name="Model1"
        ),
        ModelDiagnosticsInputs(
            predictions=y_pred, probabilities=y_prob, model=model, name="Model2"
        ),
    ]

    call = ModelDiagnosticsCall(
        df=classification_df,
        target="target",
        inputs=inputs,
        n_jobs=-1,
        random_state=42,
    )
    reports = call.run()
    assert len(reports) == 2
    assert "Model1" in reports
    assert "Model2" in reports


def test_tabular_eda_with_custom_n_jobs(classification_df, tmp_path):
    eda = TabularEDA(
        df=classification_df,
        target="target",
        n_clusters=[2, 3],
        n_jobs=2,
    )
    assert eda.n_jobs == 2

    # HTML and JSON execution with parallel engine
    html_file = tmp_path / "parallel.html"
    eda.to_html(output=html_file)
    assert html_file.exists()
