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


class TabularDataProfiler:
    """Extracts structural and statistical metrics from a Pandas DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        columns: list[str] | None = None,
        n_frequent: int = 5,
        n_extremes: int = 5,
        outlier_iqr_factor: float = 1.5,
    ):
        self.raw_df = df
        self.target = target
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
            counts.sort_index(ascending=True)
            if smallest
            else counts.sort_index(ascending=False)
        )
        subset = sorted_vals.head(n)
        total = len(series)

        return [
            {
                # Safely cast Python/NumPy numeric types without triggering ExtensionDtype errors
                "value": float(val)
                if isinstance(val, (int, float, np.number))
                and not isinstance(val, bool)
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
        is_num = pd.api.types.is_numeric_dtype(
            series
        ) and not pd.api.types.is_bool_dtype(series)

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
        base_profile.smallest_values = self._get_extremes(
            series, self.n_extremes, smallest=True
        )
        base_profile.greatest_values = self._get_extremes(
            series, self.n_extremes, smallest=False
        )
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
