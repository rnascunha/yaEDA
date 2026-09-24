from dataclasses import asdict, dataclass
from typing import Any, Literal
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection._mutual_info import _compute_mi, _iterate_columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, scale


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
    target: str | None
    target_type: Literal["classification", "regression"] | None
    target_associations: dict[str, TargetAssociation]
    collinear_pairs: list[CollinearPair]
    pearson_matrix: dict[str, dict[str, float]]
    spearman_matrix: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureTargetAnalyzer:
    """Computes bivariate associations, inter-feature collinearity, and target relationships."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] | None = None,
        collinear_threshold: float = 0.80,
        mi_sample_limit: int | None = 25000,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame.")

        self.target = target
        self.collinear_threshold = collinear_threshold
        self.mi_sample_limit = mi_sample_limit
        self.n_jobs = n_jobs
        self.random_state = random_state

        if features is not None:
            self.features = [f for f in features if f != target and f in df.columns]
        else:
            self.features = [col for col in df.columns if col != target]

        cols_to_keep = self.features + ([self.target] if self.target else [])
        if self.target:
            self.df = df[cols_to_keep].dropna(subset=[self.target]).copy()
            self.target_type = target_type or self._infer_target_type(self.df[self.target])
        else:
            self.df = df[cols_to_keep].copy()
            self.target_type = None

    def _infer_target_type(
        self, target_series: pd.Series
    ) -> Literal["classification", "regression"]:
        is_num = pd.api.types.is_numeric_dtype(target_series) and not pd.api.types.is_bool_dtype(
            target_series
        )
        unique_count = target_series.nunique()
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
        if self.target is None or self.target not in self.df.columns:
            return pearson_mat, spearman_mat, target_stats

        target_s = self.df[self.target]
        is_num_target = pd.api.types.is_numeric_dtype(target_s) and not pd.api.types.is_bool_dtype(
            target_s
        )
        is_binary = target_s.nunique() == 2

        clean_target = None
        if is_num_target and self.target_type == "regression":
            clean_target = target_s.astype(float)
        elif is_binary:
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
        if self.target is None or self.target not in self.df.columns:
            return {}

        X = self.df[self.features].copy()
        y = self.df[self.target].copy()

        # Stratified / random subsampling for Mutual Information k-NN scaling
        if self.mi_sample_limit is not None and len(X) > self.mi_sample_limit:
            stratify = None
            if self.target_type == "classification":
                val_counts = y.value_counts()
                if (val_counts >= 2).all() and len(val_counts) < self.mi_sample_limit:
                    stratify = y
            try:
                if stratify is not None:
                    X, _, y, _ = train_test_split(
                        X,
                        y,
                        train_size=self.mi_sample_limit,
                        random_state=self.random_state,
                        stratify=stratify,
                    )
                else:
                    rng = np.random.RandomState(self.random_state)
                    sample_idx = rng.choice(len(X), size=self.mi_sample_limit, replace=False)
                    X = X.iloc[sample_idx].copy()
                    y = y.iloc[sample_idx].copy()
            except Exception:  # noqa: S110
                pass

        discrete_mask: list[bool] = []
        for col in self.features:
            is_discrete = not (
                pd.api.types.is_numeric_dtype(X[col])
                and not pd.api.types.is_bool_dtype(X[col])
                and X[col].nunique() > 20
            )
            discrete_mask.append(is_discrete)

            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(str).fillna("__MISSING__")
                X[col] = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ).fit_transform(X[[col]])
            else:
                median_val = X[col].median()
                X[col] = X[col].fillna(0.0 if np.isnan(median_val) else median_val)

        # Prepare X and y arrays for parallel _compute_mi
        X_mat = X.to_numpy(dtype=np.float64, copy=True)
        discrete_target = self.target_type == "classification"

        if discrete_target:
            y_arr = pd.factorize(y)[0]
        else:
            y_arr = scale(y.astype(float).to_numpy(), with_mean=False)
            rng = np.random.RandomState(self.random_state)
            y_arr += (
                1e-10 * np.maximum(1, np.mean(np.abs(y_arr))) * rng.standard_normal(size=len(y_arr))
            )

        # Add jitter to continuous feature columns to break ties (standard Kraskov procedure)
        rng = np.random.RandomState(self.random_state)
        continuous_indices = [i for i, disc in enumerate(discrete_mask) if not disc]
        if continuous_indices:
            X_mat[:, continuous_indices] = scale(
                X_mat[:, continuous_indices], with_mean=False, copy=False
            )
            means = np.maximum(1, np.mean(np.abs(X_mat[:, continuous_indices]), axis=0))
            X_mat[:, continuous_indices] += (
                1e-10 * means * rng.standard_normal(size=(len(X_mat), len(continuous_indices)))
            )

        # Parallelize column-by-column mutual information across CPU threads
        n_jobs_eff = self.n_jobs if (len(self.features) > 1 and self.n_jobs != 1) else 1
        if n_jobs_eff == 1:
            mi_scores = [
                _compute_mi(x_col, y_arr, disc, discrete_target, 3)
                for x_col, disc in zip(_iterate_columns(X_mat), discrete_mask)
            ]
        else:
            mi_scores = Parallel(n_jobs=n_jobs_eff, prefer="threads")(
                delayed(_compute_mi)(x_col, y_arr, disc, discrete_target, 3)
                for x_col, disc in zip(_iterate_columns(X_mat), discrete_mask)
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
        if self.target is not None:
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
