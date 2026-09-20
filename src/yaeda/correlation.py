from dataclasses import asdict, dataclass
from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class TargetAssociation:
    feature: str
    is_numeric: bool
    pearson_corr: float | None = None
    pearson_p_value: float | None = None
    spearman_corr: float | None = None
    spearman_p_value: float | None = None
    mutual_info: float | None = None


@dataclass
class CollinearPair:
    feature_a: str
    feature_b: str
    pearson_corr: float
    spearman_corr: float


@dataclass
class CorrelationReport:
    target: str
    target_type: Literal["classification", "regression"]
    target_associations: dict[str, TargetAssociation]
    collinear_pairs: list[CollinearPair]
    pearson_matrix: dict[str, dict[str, float]]
    spearman_matrix: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureTargetAnalyzer:
    """Computes bivariate associations and feature-target relationships."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] | None = None,
        collinear_threshold: float = 0.80,
        random_state: int = 42,
    ):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame.")

        self.target = target
        self.collinear_threshold = collinear_threshold
        self.random_state = random_state

        # Resolve features
        if features is not None:
            self.features = [f for f in features if f != target and f in df.columns]
        else:
            self.features = [col for col in df.columns if col != target]

        # Filter to rows where target is present
        self.df = df[self.features + [self.target]].dropna(subset=[self.target]).copy()

        # Determine target type if not provided
        self.target_type = target_type or self._infer_target_type(self.df[self.target])

    def _infer_target_type(
        self, target_series: pd.Series
    ) -> Literal["classification", "regression"]:
        is_num = pd.api.types.is_numeric_dtype(target_series) and not pd.api.types.is_bool_dtype(
            target_series
        )
        unique_count = target_series.nunique()

        # Categorical, boolean, or few unique integers imply classification
        if not is_num or unique_count <= 10 or pd.api.types.is_bool_dtype(target_series):
            return "classification"
        return "regression"

    def _compute_numeric_correlations(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
        numeric_features = [
            f
            for f in self.features
            if pd.api.types.is_numeric_dtype(self.df[f])
            and not pd.api.types.is_bool_dtype(self.df[f])
        ]

        if numeric_features:
            sub_df = self.df[numeric_features]
            pearson_mat = sub_df.corr(method="pearson").round(4)
            spearman_mat = sub_df.corr(method="spearman").round(4)
        else:
            pearson_mat = pd.DataFrame()
            spearman_mat = pd.DataFrame()

        target_stats: dict[str, dict[str, float]] = {}
        target_s = self.df[self.target]
        is_num_target = pd.api.types.is_numeric_dtype(target_s) and not pd.api.types.is_bool_dtype(
            target_s
        )
        is_binary = target_s.nunique() == 2

        clean_target = None
        if is_num_target and self.target_type == "regression":
            clean_target = target_s.astype(float)
        elif is_binary:
            # Point-biserial correlation: factorize binary target to 0.0 and 1.0
            clean_target = pd.Series(pd.factorize(target_s)[0], index=self.df.index, dtype=float)
        elif is_num_target:
            clean_target = target_s.astype(float)

        if clean_target is not None and numeric_features:
            for feat in numeric_features:
                pair_df = pd.concat([self.df[feat], clean_target], axis=1).dropna()
                if (
                    len(pair_df) < 3
                    or pair_df[feat].nunique() <= 1
                    or pair_df.iloc[:, 1].nunique() <= 1
                ):
                    continue
                try:
                    r_p, p_val_p = stats.pearsonr(pair_df[feat], pair_df.iloc[:, 1])
                    r_s, p_val_s = stats.spearmanr(pair_df[feat], pair_df.iloc[:, 1])
                    target_stats[feat] = {
                        "pearson_corr": round(float(r_p), 4),
                        "pearson_p_value": round(float(p_val_p), 6),
                        "spearman_corr": round(float(r_s), 4),
                        "spearman_p_value": round(float(p_val_s), 6),
                    }
                except Exception:  # noqa: S110
                    pass

        return pearson_mat, spearman_mat, target_stats

    def _compute_mutual_information(self) -> dict[str, float]:
        X = self.df[self.features].copy()
        y = self.df[self.target].copy()

        discrete_mask: list[bool] = []
        for col in self.features:
            is_discrete = not (
                pd.api.types.is_numeric_dtype(X[col])
                and not pd.api.types.is_bool_dtype(X[col])
                and X[col].nunique() > 20
            )
            discrete_mask.append(is_discrete)

            # Impute and encode for scikit-learn estimators
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(str).fillna("__MISSING__")
                X[col] = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ).fit_transform(X[[col]])
            else:
                median_val = X[col].median()
                X[col] = X[col].fillna(0.0 if np.isnan(median_val) else median_val)

        # Format target
        if self.target_type == "classification":
            y_encoded = pd.factorize(y)[0]
            mi_scores = mutual_info_classif(
                X,
                y_encoded,
                discrete_features=discrete_mask,
                random_state=self.random_state,
            )
        else:
            y_numeric = y.astype(float)
            mi_scores = mutual_info_regression(
                X,
                y_numeric,
                discrete_features=discrete_mask,
                random_state=self.random_state,
            )

        return {feat: round(float(score), 4) for feat, score in zip(self.features, mi_scores)}

    def _extract_collinear_pairs(
        self, pearson_mat: pd.DataFrame, spearman_mat: pd.DataFrame
    ) -> list[CollinearPair]:
        collinear_pairs: list[CollinearPair] = []
        if pearson_mat.empty:
            return collinear_pairs

        cols = pearson_mat.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                p_r = pearson_mat.iloc[i, j]
                s_r = spearman_mat.iloc[i, j]
                if abs(p_r) >= self.collinear_threshold or abs(s_r) >= self.collinear_threshold:
                    collinear_pairs.append(
                        CollinearPair(
                            feature_a=cols[i],
                            feature_b=cols[j],
                            pearson_corr=float(p_r),
                            spearman_corr=float(s_r),
                        )
                    )
        return sorted(collinear_pairs, key=lambda x: abs(x.pearson_corr), reverse=True)

    def run(self) -> CorrelationReport:
        p_matrix, s_matrix, target_numeric_stats = self._compute_numeric_correlations()
        mi_scores = self._compute_mutual_information()
        collinear_pairs = self._extract_collinear_pairs(p_matrix, s_matrix)

        associations: dict[str, TargetAssociation] = {}
        for feat in self.features:
            is_num = pd.api.types.is_numeric_dtype(
                self.df[feat]
            ) and not pd.api.types.is_bool_dtype(self.df[feat])
            num_stats = target_numeric_stats.get(feat, {})

            associations[feat] = TargetAssociation(
                feature=feat,
                is_numeric=is_num,
                pearson_corr=num_stats.get("pearson_corr"),
                pearson_p_value=num_stats.get("pearson_p_value"),
                spearman_corr=num_stats.get("spearman_corr"),
                spearman_p_value=num_stats.get("spearman_p_value"),
                mutual_info=mi_scores.get(feat),
            )

        return CorrelationReport(
            target=self.target,
            target_type=self.target_type,
            target_associations=associations,
            collinear_pairs=collinear_pairs,
            pearson_matrix=p_matrix.to_dict() if not p_matrix.empty else {},
            spearman_matrix=s_matrix.to_dict() if not s_matrix.empty else {},
        )
