import numpy as np
from yaeda.model_diagnostics import (
    ModelDiagnosticsAnalyzer,
    ModelDiagnosticsCall,
    ModelDiagnosticsInputs,
    ModelDiagnosticsReport,
)


def test_model_diagnostics_inputs_initialization(fitted_binary_model):
    model, y_pred, y_prob = fitted_binary_model
    diag_input = ModelDiagnosticsInputs(
        predictions=y_pred,
        probabilities=y_prob,
        model=model,
        name="TestModel",
    )

    assert diag_input.name == "TestModel"
    assert len(diag_input.predictions) == len(y_pred)
    assert diag_input.probabilities is not None


def test_model_diagnostics_analyzer_classification(classification_df, fitted_binary_model):
    model, y_pred, y_prob = fitted_binary_model
    y_true = classification_df["target"].to_numpy()

    # Introduce exactly 12 misclassifications on the 250 rows
    y_pred_mod = y_true.copy()
    y_pred_mod[:12] = 1 - y_pred_mod[:12]

    inputs = ModelDiagnosticsInputs(
        predictions=y_pred_mod,
        probabilities=y_prob,
        model=model,
        name="RandomForest",
    )

    analyzer = ModelDiagnosticsAnalyzer(
        df=classification_df,  # Full 250 rows
        target="target",
        inputs=inputs,
        features=["feat_num1", "feat_skewed"],
        target_type="classification",
        max_worst_error=10,
    )
    rep = analyzer.run()

    assert isinstance(rep, ModelDiagnosticsReport)
    assert rep.name == "RandomForest"
    assert rep.error_count == 12
    assert rep.error_rate == round((12 / len(y_true)) * 100, 2)
    assert rep.confusion_data["total_errors"] == 12
    assert len(rep.worst_errors) <= 10
    assert "accuracy" in rep.metrics
    assert "f1" in rep.metrics


def test_model_diagnostics_analyzer_regression(regression_df, fitted_regression_model):
    model, y_pred = fitted_regression_model

    inputs = ModelDiagnosticsInputs(
        predictions=y_pred,  # 200 items
        model=model,
        name="RF_Regressor",
    )

    analyzer = ModelDiagnosticsAnalyzer(
        df=regression_df,  # Full 200 rows
        target="target_price",
        inputs=inputs,
        features=["feat_num1", "feat_num2"],
        target_type="regression",
        max_worst_error=8,
    )
    rep = analyzer.run()

    assert rep.target_type == "regression"
    assert "mae" in rep.metrics
    assert "rmse" in rep.metrics
    assert "r2" in rep.metrics
    assert len(rep.worst_errors) <= 8


def test_model_diagnostics_call_multi_model(classification_df, fitted_binary_model):
    model, y_pred, y_prob = fitted_binary_model

    inputs_list = [
        ModelDiagnosticsInputs(
            predictions=y_pred, probabilities=y_prob, model=model, name="Model-A"
        ),
        ModelDiagnosticsInputs(predictions=y_pred, name=None),  # Must auto-name to Model-1
    ]

    call = ModelDiagnosticsCall(
        df=classification_df,  # Full 250 rows
        target="target",
        inputs=inputs_list,
        features=["feat_num1", "feat_skewed"],
        target_type="classification",
    )

    analyzers = call.analyzers()
    assert "Model-A" in analyzers
    assert "Model-1" in analyzers

    results = call.run()
    assert isinstance(results, dict)
    assert "Model-A" in results
    assert "Model-1" in results
    assert results["Model-A"].name == "Model-A"
    assert results["Model-1"].name == "Model-1"


def test_model_diagnostics_to_dict_serialization(classification_df, fitted_binary_model):
    model, y_pred, y_prob = fitted_binary_model

    inputs = ModelDiagnosticsInputs(predictions=y_pred, probabilities=y_prob, name="ModelDict")
    analyzer = ModelDiagnosticsAnalyzer(
        df=classification_df,  # Full 250 rows
        target="target",
        inputs=inputs,
        features=["feat_num1"],
    )
    rep = analyzer.run()

    payload = rep.to_dict(max_errors_exported=5)
    assert isinstance(payload, dict)
    assert payload["name"] == "ModelDict"
    assert "metrics" in payload
    assert "confusion_data" in payload
    assert len(payload["worst_errors"]) <= 5
