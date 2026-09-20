from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class ErrorInstance:
    index: Any
    true_label: Any
    predicted_label: Any
    prediction_score: float | None
    error_type: str  # 'FP', 'FN', 'OVER_PREDICTION', 'UNDER_PREDICTION'
    feature_values: dict[str, Any]


@dataclass
class ModelDiagnosticsReport:
    target: str
    target_type: Literal["classification", "regression"]
    total_samples: int
    error_count: int
    error_rate: float
    metrics: dict[str, float]
    confusion_data: dict[str, Any]  # TP, FP, TN, FN counts and rates
    worst_errors: list[ErrorInstance]
    shap_available: bool = False
    shap_explanation: Any = field(default=None, repr=False)
    feature_names: list[str] = field(default_factory=list)
    aligned_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    name: str = field(default=None)

    def to_dict(self, max_errors_exported: int = 20) -> dict[str, Any]:
        payload = {
            "target": self.target,
            "target_type": self.target_type,
            "total_samples": self.total_samples,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "metrics": self.metrics,
            "confusion_data": self.confusion_data,
            "worst_errors": [
                {
                    "index": e.index,
                    "true": e.true_label,
                    "predicted": e.predicted_label,
                    "score": e.prediction_score,
                    "error_type": e.error_type,
                    # Store only the top 3 distinguishing feature values
                    "key_features": {k: v for k, v in list(e.feature_values.items())[:3]},
                }
                for e in self.worst_errors[:max_errors_exported]
            ],
            "feature_names": self.feature_names,
            "name": self.name,
        }

        if self.shap_explanation is not None:
            mean_abs_shap = np.abs(self.shap_explanation.values).mean(axis=0)
            shap_summary = {
                feat: round(float(val), 4) for feat, val in zip(self.feature_names, mean_abs_shap)
            }
            payload["global_shap_importance"] = shap_summary

        return payload


@dataclass
class ModelDiagnosticsInputs:
    predictions: pd.Series | np.ndarray | list[float] | str
    probabilities: pd.Series | np.ndarray | list[float] | str | None = field(default=None)
    model: Any | None = field(default=None)
    name: str | None = field(default=None)


class ModelDiagnosticsAnalyzer:
    """Dissects model predictions, partitions error cohorts, and generates SHAP explanations."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        inputs: ModelDiagnosticsInputs,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] = "classification",
        preprocessed_X: np.ndarray | None = None,
        max_worst_error: int = 20,
        random_state: int = 42,
    ):
        self.df = df.copy()
        self.target = target
        self.target_type = target_type
        self.model = inputs.model
        self.preprocessed_X = preprocessed_X
        self.max_worst_errors = max_worst_error
        self.random_state = random_state
        self.name = inputs.name

        # Resolve prediction array
        if isinstance(inputs.predictions, str) and inputs.predictions in self.df.columns:
            self.y_pred = self.df[inputs.predictions].to_numpy()
            self.features = [
                f for f in (features or self.df.columns) if f not in [target, inputs.predictions]
            ]
        else:
            self.y_pred = np.asarray(inputs.predictions)
            self.features = [f for f in (features or self.df.columns) if f != target]

        # Resolve probability array
        self.y_prob = None
        if inputs.probabilities is not None:
            if isinstance(inputs.probabilities, str) and inputs.probabilities in self.df.columns:
                self.y_prob = self.df[inputs.probabilities].to_numpy().astype(float)
            else:
                self.y_prob = np.asarray(inputs.probabilities, dtype=float)

        self.y_true = self.df[self.target].to_numpy()

        # Fail fast if sample sizes do not match
        if len(self.y_pred) != len(self.y_true):
            raise ValueError(
                f"Prediction length mismatch for model '{self.name}': "
                f"target has {len(self.y_true)} rows, but predictions has {len(self.y_pred)} items."
            )

        self.y_true = self.df[self.target].to_numpy()

    def _evaluate_classification(
        self,
    ) -> tuple[dict[str, float], dict[str, Any], list[ErrorInstance]]:
        # Map labels to binary 0 and 1
        classes = np.unique(self.y_true)
        is_binary = len(classes) == 2

        y_true_enc = pd.factorize(self.y_true)[0]
        if self.y_pred.dtype.kind in {"O", "U", "S"} or len(np.unique(self.y_pred)) <= 2:
            # Align prediction classes with y_true factorization
            pos_label = classes[-1]
            y_pred_enc = np.where(self.y_pred == pos_label, 1, 0)
        else:
            y_pred_enc = (self.y_pred >= 0.5).astype(int)

        metrics = {
            "accuracy": round(float(accuracy_score(y_true_enc, y_pred_enc)), 4),
            "precision": round(float(precision_score(y_true_enc, y_pred_enc, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_enc, y_pred_enc, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_enc, y_pred_enc, zero_division=0)), 4),
        }

        if self.y_prob is not None and is_binary:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true_enc, self.y_prob)), 4)
            except Exception:  # noqa: S110
                pass

        cm = confusion_matrix(y_true_enc, y_pred_enc)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        confusion_data = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "total_positives": int(tp + fn),
            "total_negatives": int(tn + fp),
            "total_errors": int(fp + fn),
        }

        # Vectorized identification of misclassified row indices
        error_indices = np.where(y_true_enc != y_pred_enc)[0]
        conf_scores = self.y_prob if self.y_prob is not None else y_pred_enc.astype(float)

        all_error_cases: list[ErrorInstance] = []
        for idx in error_indices:
            yt = y_true_enc[idx]
            yp = y_pred_enc[idx]
            err_type = "FP" if (yt == 0 and yp == 1) else "FN"
            feat_vals = {f: self.df.iloc[idx][f] for f in self.features[:6]}
            all_error_cases.append(
                ErrorInstance(
                    index=self.df.index[idx],
                    true_label=int(yt),
                    predicted_label=int(yp),
                    prediction_score=round(float(conf_scores[idx]), 4),
                    error_type=err_type,
                    feature_values=feat_vals,
                )
            )

        # Slice only the top representative mistakes for display/SHAP
        worst_fp = [e for e in all_error_cases if e.error_type == "FP"]
        worst_fn = [e for e in all_error_cases if e.error_type == "FN"]

        worst_fp.sort(key=lambda x: x.prediction_score or 0.0, reverse=True)
        worst_fn.sort(key=lambda x: x.prediction_score or 0.0, reverse=False)

        half_k = max(1, self.max_worst_errors // 2)
        pruned_worst_cases = worst_fp[:half_k] + worst_fn[:half_k]

        return metrics, confusion_data, pruned_worst_cases

    def _evaluate_regression(
        self,
    ) -> tuple[dict[str, float], dict[str, Any], list[ErrorInstance]]:
        y_true_f = self.y_true.astype(float)
        y_pred_f = self.y_pred.astype(float)
        residuals = y_true_f - y_pred_f

        mae = float(mean_absolute_error(y_true_f, y_pred_f))
        rmse = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))
        r2 = float(r2_score(y_true_f, y_pred_f))

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }

        # In regression, define significant errors as predictions exceeding 1 MAE threshold
        over_pred = np.sum(residuals < -mae)
        under_pred = np.sum(residuals > mae)
        accurate = np.sum(np.abs(residuals) <= mae)
        total_errors = int(over_pred + under_pred)

        confusion_data = {
            "over_predicted": int(over_pred),
            "under_predicted": int(under_pred),
            "accurate_within_mae": int(accurate),
            "total_errors": total_errors,
        }

        # Slice only the top worst residuals for display/SHAP
        worst_residual_indices = np.argsort(np.abs(residuals))[::-1][: self.max_worst_errors]
        pruned_worst_cases: list[ErrorInstance] = []
        for idx in worst_residual_indices:
            r = residuals[idx]
            err_type = "OVER_PREDICTION" if r < 0 else "UNDER_PREDICTION"
            feat_vals = {f: self.df.iloc[idx][f] for f in self.features[:6]}
            pruned_worst_cases.append(
                ErrorInstance(
                    index=self.df.index[idx],
                    true_label=round(float(y_true_f[idx]), 4),
                    predicted_label=round(float(y_pred_f[idx]), 4),
                    prediction_score=round(float(r), 4),
                    error_type=err_type,
                    feature_values=feat_vals,
                )
            )

        return metrics, confusion_data, pruned_worst_cases

    def _compute_shap_explanation(self, X_eval: np.ndarray) -> Any | None:
        if not HAS_SHAP or self.model is None:
            return None

        try:
            explainer = shap.Explainer(self.model, X_eval)
            sample_size = min(len(X_eval), 400)
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(len(X_eval), size=sample_size, replace=False)
            explanation = explainer(X_eval[sample_idx])

            # If multi-output classification explanation, isolate positive class
            if len(explanation.shape) == 3:
                explanation = explanation[:, :, 1]

            explanation.feature_names = self.features
            return explanation
        except Exception:
            return None

    def run(self) -> ModelDiagnosticsReport:
        if self.target_type == "classification":
            metrics, conf_data, error_cases = self._evaluate_classification()
        else:
            metrics, conf_data, error_cases = self._evaluate_regression()

        total = len(self.df)
        err_count = conf_data["total_errors"]
        err_rate = round((err_count / total) * 100, 2)

        # Prepare X for SHAP
        X_mat = self.preprocessed_X
        if X_mat is None:
            X_mat = self.df[self.features].select_dtypes(include=[np.number]).fillna(0).to_numpy()

        shap_exp = self._compute_shap_explanation(X_mat)

        return ModelDiagnosticsReport(
            target=self.target,
            target_type=self.target_type,
            total_samples=total,
            error_count=err_count,
            error_rate=err_rate,
            metrics=metrics,
            confusion_data=conf_data,
            worst_errors=error_cases,
            shap_available=(shap_exp is not None),
            shap_explanation=shap_exp,
            feature_names=self.features,
            aligned_df=self.df,
            name=self.name,
        )


class ModelDiagnosticsCall:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        inputs: list[ModelDiagnosticsInputs],
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] = "classification",
        preprocessed_X: np.ndarray | None = None,
        max_worst_error: int = 20,
        random_state: int = 42,
    ):
        for i, item in enumerate(inputs):
            if not item.name:
                item.name = f"Model-{i}"
        self._analyzers: dict[str, ModelDiagnosticsAnalyzer] = {
            item.name: ModelDiagnosticsAnalyzer(
                df=df,
                target=target,
                inputs=item,
                features=features,
                target_type=target_type,
                preprocessed_X=preprocessed_X,
                max_worst_error=max_worst_error,
                random_state=random_state,
            )
            for item in inputs
        }

    def analyzers(self):
        return self._analyzers

    def run(self) -> dict[str, ModelDiagnosticsReport]:
        return {name: analyzer.run() for name, analyzer in self._analyzers.items()}
