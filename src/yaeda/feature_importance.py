from dataclasses import asdict, dataclass, field
import importlib.util
from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
if HAS_LIGHTGBM:
    import lightgbm as lgb


def _subsample_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    is_classification: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return X, y

    if is_classification:
        counts = pd.Series(y).value_counts()
        if (counts >= 2).all() and len(counts) < max_samples:
            try:
                X_sub, _, y_sub, _ = train_test_split(
                    X, y, train_size=max_samples, random_state=random_state, stratify=y
                )
                return X_sub, y_sub
            except Exception:  # noqa: S110
                pass

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return X[idx], y[idx]


@dataclass
class FeatureImportanceMetric:
    feature: str
    tree_importance: float
    permutation_mean: float
    permutation_std: float
    attribution_score: float
    attribution_method: str
    mutual_info: float
    golden_feature_score: float
    tier: str
    recommendation: str
    rank: int


@dataclass
class FeatureImportanceReport:
    target: str | None
    target_type: Literal["classification", "regression"] | None
    model_type: str
    attribution_method: str = "SHAP (mean |value|)"
    importances: list[FeatureImportanceMetric] = field(default_factory=list)
    fitted_model: Any = None
    preprocessed_X: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.importances:
            return pd.DataFrame()
        data = [asdict(item) for item in self.importances]
        return pd.DataFrame(data).set_index("rank")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureImportanceAnalyzer:
    """Evaluates multi-perspective feature importance via Trees, Permutation, and SHAP."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] | None = None,
        model_engine: Literal["auto", "lightgbm", "extra_trees", "random_forest"] = "auto",
        mi_scores: dict[str, float] | None = None,
        test_size: float = 0.25,
        fit_sample_limit: int | None = 25000,
        permutation_sample_limit: int | None = 10000,
        mi_sample_limit: int | None = 25000,
        shap_sample_limit: int | None = 500,
        random_state: int = 42,
    ):
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame.")

        self.target = target
        self.model_engine = model_engine
        self.test_size = test_size
        self.fit_sample_limit = fit_sample_limit
        self.permutation_sample_limit = permutation_sample_limit
        self.mi_sample_limit = mi_sample_limit
        self.shap_sample_limit = shap_sample_limit or 500
        self.random_state = random_state
        self.mi_scores = mi_scores or {}

        if features is not None:
            self.features = [f for f in features if f != target and f in df.columns]
        else:
            self.features = [col for col in df.columns if col != target]

        if self.target is not None:
            self.df = df[self.features + [self.target]].dropna(subset=[self.target]).copy()
            self.target_type = target_type or self._infer_target_type(self.df[self.target])
        else:
            self.df = df[self.features].copy()
            self.target_type = None

    def _infer_target_type(
        self, target_series: pd.Series
    ) -> Literal["classification", "regression"]:
        is_num = pd.api.types.is_numeric_dtype(target_series) and not pd.api.types.is_bool_dtype(
            target_series
        )
        if not is_num or target_series.nunique() <= 10 or pd.api.types.is_bool_dtype(target_series):
            return "classification"
        return "regression"

    def _preprocess_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        X_df = self.df[self.features].copy()
        y_raw = self.df[self.target].copy()

        if self.target_type == "classification":
            y = pd.factorize(y_raw)[0]
        else:
            y = y_raw.astype(float).to_numpy()

        num_cols = [
            c
            for c in self.features
            if pd.api.types.is_numeric_dtype(X_df[c]) and not pd.api.types.is_bool_dtype(X_df[c])
        ]
        cat_cols = [c for c in self.features if c not in num_cols]

        processed_parts = []
        ordered_feature_names = []

        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            num_imputed = num_imputer.fit_transform(X_df[num_cols])
            processed_parts.append(num_imputed)
            ordered_feature_names.extend(num_cols)

        if cat_cols:
            cat_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__")
            cat_imputed = cat_imputer.fit_transform(X_df[cat_cols].astype(str))
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            cat_encoded = encoder.fit_transform(cat_imputed)
            processed_parts.append(cat_encoded)
            ordered_feature_names.extend(cat_cols)

        X = np.hstack(processed_parts)
        return X, y, ordered_feature_names

    def _build_model(self):
        engine = self.model_engine
        if engine == "auto":
            engine = "lightgbm" if HAS_LIGHTGBM else "extra_trees"

        is_classif = self.target_type == "classification"

        if engine == "lightgbm" and HAS_LIGHTGBM:
            if is_classif:
                return lgb.LGBMClassifier(
                    n_estimators=60,
                    max_depth=6,
                    learning_rate=0.08,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
                return lgb.LGBMRegressor(
                    n_estimators=60,
                    max_depth=6,
                    learning_rate=0.08,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
        elif engine == "extra_trees":
            if is_classif:
                return ExtraTreesClassifier(
                    n_estimators=60,
                    max_depth=7,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                return ExtraTreesRegressor(
                    n_estimators=60,
                    max_depth=7,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
        else:  # random_forest
            if is_classif:
                return RandomForestClassifier(
                    n_estimators=60,
                    max_depth=7,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                return RandomForestRegressor(
                    n_estimators=60,
                    max_depth=7,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray):
        model = self._build_model()
        model.fit(X_train, y_train)
        return model

    def _compute_attributions(self, model: Any, X: np.ndarray) -> tuple[np.ndarray, str]:
        if HAS_SHAP:
            try:
                explainer = shap.TreeExplainer(model)
                sample_size = min(len(X), self.shap_sample_limit)
                sample_idx = np.random.RandomState(self.random_state).choice(
                    len(X), size=sample_size, replace=False
                )
                shap_values = explainer.shap_values(X[sample_idx])

                if isinstance(shap_values, list):
                    attribution = np.mean([np.abs(c).mean(axis=0) for c in shap_values], axis=0)
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    attribution = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    vals = getattr(shap_values, "values", shap_values)
                    attribution = (
                        np.abs(vals).mean(axis=(0, 2))
                        if vals.ndim == 3
                        else np.abs(vals).mean(axis=0)
                    )

                return attribution, "SHAP (mean |value|)"
            except Exception:  # noqa: S110
                pass  # Fall through to PDP sensitivity variance

        # Fallback sensitivity variance
        variances = []
        sample_size = min(len(X), 300)
        X_sub = X[:sample_size]
        for feat_idx in range(X.shape[1]):
            try:
                pdp_res = partial_dependence(model, X_sub, features=[feat_idx], grid_resolution=15)
                variances.append(float(np.var(pdp_res["average"])))
            except Exception:
                variances.append(0.0)
        return np.array(variances), "PDP Sensitivity Variance"

    def run(self) -> FeatureImportanceReport:
        if self.target is None or self.target not in self.df.columns:
            return FeatureImportanceReport(
                target=None,
                target_type=None,
                model_type="None",
                attribution_method="None",
                importances=[],
                fitted_model=None,
                preprocessed_X=None,
                feature_names=self.features,
            )

        X, y, ordered_features = self._preprocess_data()
        is_classif = self.target_type == "classification"

        if len(y) < 20:
            X_train, X_val, y_train, y_val = X, X, y, y
        else:
            stratify = y if is_classif and pd.Series(y).value_counts().min() > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
            )

        # 1. Fit model on subsampled data
        X_train_fit, y_train_fit = _subsample_xy(
            X_train, y_train, self.fit_sample_limit, is_classif, self.random_state
        )
        model = self._fit_model(X_train_fit, y_train_fit)
        tree_mdi = getattr(model, "feature_importances_", np.ones(len(ordered_features)))

        # 2. Permutation importance on subsampled validation set
        X_val_perm, y_val_perm = _subsample_xy(
            X_val, y_val, self.permutation_sample_limit, is_classif, self.random_state
        )
        scoring = "roc_auc" if (is_classif and len(np.unique(y)) == 2) else None
        perm_res = permutation_importance(
            model,
            X_val_perm,
            y_val_perm,
            n_repeats=5,
            random_state=self.random_state,
            scoring=scoring,
            n_jobs=-1,
        )
        perm_mean = np.maximum(perm_res.importances_mean, 0.0)
        perm_std = perm_res.importances_std

        # 3. Attributions (Tree SHAP or PDP variance)
        attributions, attr_method = self._compute_attributions(model, X)

        def normalize(arr: np.ndarray) -> np.ndarray:
            denom = arr.max() - arr.min()
            return np.zeros_like(arr) if denom == 0 else (arr - arr.min()) / denom

        norm_tree = normalize(tree_mdi)
        norm_perm = normalize(perm_mean)
        norm_attr = normalize(attributions)

        # 4. Mutual Information
        mi_dict = getattr(self, "mi_scores", None)
        if not mi_dict:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

            X_train_mi, y_train_mi = _subsample_xy(
                X_train, y_train, self.mi_sample_limit, is_classif, self.random_state
            )
            if is_classif:
                mi_vals = mutual_info_classif(
                    X_train_mi, y_train_mi, random_state=self.random_state
                )
            else:
                mi_vals = mutual_info_regression(
                    X_train_mi, y_train_mi, random_state=self.random_state
                )
            mi_dict = {feat: float(score) for feat, score in zip(ordered_features, mi_vals)}
            self.mi_scores = mi_dict

        mi_array = np.array([mi_dict.get(f, 0.0) for f in ordered_features])
        norm_mi = normalize(mi_array)

        composite = (0.35 * norm_attr) + (0.35 * norm_perm) + (0.15 * norm_mi) + (0.15 * norm_tree)

        records: list[FeatureImportanceMetric] = []
        for i, feat in enumerate(ordered_features):
            score = round(float(composite[i]), 4)
            tier = (
                "Tier 1: Golden"
                if score >= 0.70
                else (
                    "Tier 2: Strong"
                    if score >= 0.40
                    else ("Tier 3: Moderate" if score >= 0.15 else "Tier 4: Noise / Low")
                )
            )
            rec = (
                "Primary Predictor"
                if score >= 0.70
                else (
                    "Secondary Predictor"
                    if score >= 0.40
                    else ("Auxiliary Predictor" if score >= 0.15 else "Candidate for pruning")
                )
            )

            records.append(
                FeatureImportanceMetric(
                    feature=feat,
                    tree_importance=round(float(tree_mdi[i]), 4),
                    permutation_mean=round(float(perm_mean[i]), 4),
                    permutation_std=round(float(perm_std[i]), 4),
                    attribution_score=round(float(attributions[i]), 4),
                    attribution_method=attr_method,
                    mutual_info=round(float(mi_array[i]), 4),
                    golden_feature_score=score,
                    tier=tier,
                    recommendation=rec,
                    rank=0,
                )
            )

        records.sort(key=lambda x: x.golden_feature_score, reverse=True)
        for idx, rec in enumerate(records, start=1):
            rec.rank = idx

        return FeatureImportanceReport(
            target=self.target,
            target_type=self.target_type,
            model_type=type(model).__name__,
            attribution_method=attr_method,
            importances=records,
            fitted_model=model,
            preprocessed_X=X,
            feature_names=ordered_features,
        )
