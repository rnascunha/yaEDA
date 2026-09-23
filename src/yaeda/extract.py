from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class OutlierStats:
    lower_bound: float
    upper_bound: float
    count: int
    percentage: float


@dataclass
class FeatureProfile:
    name: str
    dtype: str
    is_numeric: bool
    is_target: bool
    total_count: int
    missing_count: int
    missing_percentage: float
    distinct_count: int
    distinct_percentage: float
    zero_count: int
    zero_percentage: float
    most_frequent: list[dict[str, Any]]
    mode: Any | None = None

    # Numeric-specific metrics
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    median: float | None = None
    variance: float | None = None
    std_dev: float | None = None
    q05: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    q95: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    smallest_values: list[dict[str, Any]] | None = None
    greatest_values: list[dict[str, Any]] | None = None
    outliers: OutlierStats | None = None


@dataclass
class TableProfile:
    dataset_name: str
    n_rows: int
    n_columns: int
    target_column: str | None
    memory_usage_bytes: int
    memory_usage_mb: float
    dtype_counts: dict[str, int]
    column_dtypes: dict[str, str]
    features: dict[str, FeatureProfile] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureComparison:
    feature: str
    in_primary: bool
    in_secondary: bool
    is_numeric: bool
    primary_missing_pct: float | None = None
    secondary_missing_pct: float | None = None
    delta_missing_pct: float | None = None
    primary_mean: float | None = None
    secondary_mean: float | None = None
    delta_mean: float | None = None
    primary_std: float | None = None
    secondary_std: float | None = None
    primary_distinct: int | None = None
    secondary_distinct: int | None = None
    unseen_categories: list[Any] = field(
        default_factory=list
    )  # In secondary, but absent in primary


@dataclass
class MultiTableProfile:
    primary_profile: TableProfile
    secondary_profiles: dict[str, TableProfile] = field(default_factory=dict)
    comparisons: dict[str, list[FeatureComparison]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TabularDataProfiler:
    """Extracts structural and statistical metrics from a Pandas DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        columns: list[str] | None = None,
        dataset_name: str = "Primary",
        n_frequent: int = 5,
        n_extremes: int = 5,
        outlier_iqr_factor: float = 1.5,
    ):
        self.raw_df = df
        self.target = target
        self.dataset_name = dataset_name
        self.n_frequent = n_frequent
        self.n_extremes = n_extremes
        self.outlier_factor = outlier_iqr_factor

        # Determine working columns
        if columns is not None:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self.columns = columns
        else:
            self.columns = list(df.columns)

        if self.target and self.target not in self.columns:
            self.columns.append(self.target)

        self.df = self.raw_df[self.columns]

    def _extract_table_stats(self) -> TableProfile:
        total_memory = int(self.df.memory_usage(deep=True).sum())
        dtype_counts = self.df.dtypes.value_counts().to_dict()
        dtype_counts_str = {str(k): int(v) for k, v in dtype_counts.items()}
        col_dtypes = {col: str(self.df[col].dtype) for col in self.df.columns}

        return TableProfile(
            dataset_name=self.dataset_name,
            n_rows=len(self.df),
            n_columns=len(self.columns),
            target_column=self.target,
            memory_usage_bytes=total_memory,
            memory_usage_mb=round(total_memory / (1024**2), 4),
            dtype_counts=dtype_counts_str,
            column_dtypes=col_dtypes,
        )

    def _get_top_frequencies(self, series: pd.Series, n: int) -> list[dict[str, Any]]:
        counts = series.value_counts(dropna=False).head(n)
        total = len(series)
        return [
            {
                "value": "NaN" if pd.isna(val) else val,
                "count": int(count),
                "percentage": round((count / total) * 100, 3),
            }
            for val, count in counts.items()
        ]

    def _get_extremes(
        self, series: pd.Series, n: int, smallest: bool = True
    ) -> list[dict[str, Any]]:
        clean_s = series.dropna()
        if clean_s.empty:
            return []

        counts = clean_s.value_counts()
        sorted_vals = (
            counts.sort_index(ascending=True) if smallest else counts.sort_index(ascending=False)
        )
        subset = sorted_vals.head(n)
        total = len(series)

        return [
            {
                # Safely cast Python/NumPy numeric types without triggering ExtensionDtype errors
                "value": float(val)
                if isinstance(val, (int, float, np.number)) and not isinstance(val, bool)
                else str(val),
                "count": int(count),
                "percentage": round((count / total) * 100, 3),
            }
            for val, count in subset.items()
        ]

    def _profile_feature(self, col_name: str) -> FeatureProfile:
        series = self.df[col_name]
        total_count = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_count) * 100, 3)
        distinct_count = int(series.nunique(dropna=False))
        distinct_pct = round((distinct_count / total_count) * 100, 3)

        mode_val = series.mode(dropna=True)
        mode = mode_val.iloc[0] if not mode_val.empty else None
        if isinstance(mode, (np.generic, pd.Timestamp)):
            mode = mode.item() if hasattr(mode, "item") else str(mode)

        # 1. Determine numeric status first using pandas API
        is_num = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        # 2. Safely count zeros
        zero_count = int((series == 0).sum()) if is_num else 0
        zero_pct = round((zero_count / total_count) * 100, 3)

        base_profile = FeatureProfile(
            name=col_name,
            dtype=str(series.dtype),
            is_numeric=is_num,
            is_target=(col_name == self.target),
            total_count=total_count,
            missing_count=missing_count,
            missing_percentage=missing_pct,
            distinct_count=distinct_count,
            distinct_percentage=distinct_pct,
            zero_count=zero_count,
            zero_percentage=zero_pct,
            most_frequent=self._get_top_frequencies(series, self.n_frequent),
            mode=mode,
        )

        if not is_num or series.dropna().empty:
            return base_profile

        # Compute numerical statistics
        clean_s = series.dropna().astype(float)

        q05, q25, q50, q75, q95 = np.percentile(clean_s, [5, 25, 50, 75, 95])
        iqr = q75 - q25

        # Outlier calculation via Tukey's IQR rule
        lower_bound = q25 - (self.outlier_factor * iqr)
        upper_bound = q75 + (self.outlier_factor * iqr)
        outlier_mask = (clean_s < lower_bound) | (clean_s > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / total_count) * 100, 3)

        base_profile.min_value = float(clean_s.min())
        base_profile.max_value = float(clean_s.max())
        base_profile.mean = round(float(clean_s.mean()), 4)
        base_profile.median = round(float(q50), 4)
        base_profile.variance = round(float(clean_s.var()), 4)
        base_profile.std_dev = round(float(clean_s.std()), 4)
        base_profile.q05 = round(float(q05), 4)
        base_profile.q25 = round(float(q25), 4)
        base_profile.q50 = round(float(q50), 4)
        base_profile.q75 = round(float(q75), 4)
        base_profile.q95 = round(float(q95), 4)
        base_profile.iqr = round(float(iqr), 4)
        base_profile.skewness = round(float(clean_s.skew()), 4)
        base_profile.kurtosis = round(float(clean_s.kurt()), 4)
        base_profile.smallest_values = self._get_extremes(series, self.n_extremes, smallest=True)
        base_profile.greatest_values = self._get_extremes(series, self.n_extremes, smallest=False)
        base_profile.outliers = OutlierStats(
            lower_bound=round(lower_bound, 4),
            upper_bound=round(upper_bound, 4),
            count=outlier_count,
            percentage=outlier_pct,
        )

        return base_profile

    def run(self) -> TableProfile:
        table_profile = self._extract_table_stats()
        for col in self.columns:
            table_profile.features[col] = self._profile_feature(col)
        return table_profile


class MultiDatasetProfiler:
    """Profiles primary and secondary datasets, calculating distribution shifts and unseen values."""

    def __init__(
        self,
        primary_df: pd.DataFrame,
        secondary_dfs: dict[str, pd.DataFrame],
        target: str | None = None,
        primary_name: str = "Primary",
        features: list[str] | None = None,
    ):
        self.primary_df = primary_df
        self.secondary_dfs = secondary_dfs
        self.target = target
        self.primary_name = primary_name
        self.features = features

    def run(self) -> MultiTableProfile:
        primary_profiler = TabularDataProfiler(
            df=self.primary_df,
            target=self.target,
            columns=self.features,
            dataset_name=self.primary_name,
        )
        primary_profile = primary_profiler.run()

        secondary_profiles: dict[str, TableProfile] = {}
        comparisons: dict[str, list[FeatureComparison]] = {}

        for sec_name, sec_df in self.secondary_dfs.items():
            sec_profiler = TabularDataProfiler(
                df=sec_df,
                target=self.target if (self.target and self.target in sec_df.columns) else None,
                dataset_name=sec_name,
            )
            sec_profile = sec_profiler.run()
            secondary_profiles[sec_name] = sec_profile

            # Compare features against primary
            feat_comps: list[FeatureComparison] = []
            all_cols = list(
                dict.fromkeys(
                    list(primary_profile.features.keys()) + list(sec_profile.features.keys())
                )
            )

            for col in all_cols:
                if self.target and col == self.target:
                    continue

                p_feat = primary_profile.features.get(col)
                s_feat = sec_profile.features.get(col)

                in_p = p_feat is not None
                in_s = s_feat is not None
                is_num = p_feat.is_numeric if p_feat else (s_feat.is_numeric if s_feat else False)

                unseen: list[Any] = []
                if in_p and in_s and not is_num:
                    p_cats = set(self.primary_df[col].dropna().unique())
                    s_cats = set(sec_df[col].dropna().unique())
                    unseen = list(s_cats - p_cats)[:10]

                delta_miss = None
                if in_p and in_s:
                    delta_miss = round(s_feat.missing_percentage - p_feat.missing_percentage, 3)

                delta_mean = None
                if in_p and in_s and is_num and p_feat.mean is not None and s_feat.mean is not None:
                    delta_mean = round(s_feat.mean - p_feat.mean, 4)

                feat_comps.append(
                    FeatureComparison(
                        feature=col,
                        in_primary=in_p,
                        in_secondary=in_s,
                        is_numeric=is_num,
                        primary_missing_pct=p_feat.missing_percentage if p_feat else None,
                        secondary_missing_pct=s_feat.missing_percentage if s_feat else None,
                        delta_missing_pct=delta_miss,
                        primary_mean=p_feat.mean if p_feat else None,
                        secondary_mean=s_feat.mean if s_feat else None,
                        delta_mean=delta_mean,
                        primary_std=p_feat.std_dev if p_feat else None,
                        secondary_std=s_feat.std_dev if s_feat else None,
                        primary_distinct=p_feat.distinct_count if p_feat else None,
                        secondary_distinct=s_feat.distinct_count if s_feat else None,
                        unseen_categories=unseen,
                    )
                )

            comparisons[sec_name] = feat_comps

        return MultiTableProfile(
            primary_profile=primary_profile,
            secondary_profiles=secondary_profiles,
            comparisons=comparisons,
        )
