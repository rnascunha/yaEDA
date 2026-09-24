from dataclasses import asdict, dataclass
import itertools
from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split


@dataclass
class InteractionResult:
    feature_a: str
    feature_b: str
    operation: str  # 'Multiplication', 'Ratio (A/B)', 'Ratio (B/A)', 'Sum', 'Difference'
    formula: str
    combined_score: float
    baseline_score: float
    synergy_gain: float
    metric_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionReport:
    target: str | None
    target_type: Literal["classification", "regression"] | None
    top_interactions: list[InteractionResult]
    best_per_pair: list[InteractionResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureInteractionAnalyzer:
    """Evaluates candidate pairwise arithmetic interactions and quantifies synergy gains."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] | None = "regression",
        max_pairs: int | None = None,
        sample_limit: int | None = 25000,
        random_state: int = 42,
    ):
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame.")

        self.df = df
        self.target = target
        self.target_type = target_type
        self.max_pairs = max_pairs
        self.sample_limit = sample_limit
        self.random_state = random_state

        if features is not None:
            self.features = [f for f in features if f in df.columns and f != target]
        else:
            self.features = [col for col in df.columns if col != target]

    def _subsample_eval_df(self) -> pd.DataFrame:
        if self.sample_limit is None or len(self.df) <= self.sample_limit:
            return self.df

        if self.target and self.target in self.df.columns:
            y = self.df[self.target].dropna()
            is_classif = (self.target_type == "classification") or (y.nunique() <= 10)
            if is_classif:
                counts = y.value_counts()
                if (counts >= 2).all() and len(counts) < self.sample_limit:
                    try:
                        sampled, _ = train_test_split(
                            self.df,
                            train_size=self.sample_limit,
                            random_state=self.random_state,
                            stratify=y,
                        )
                        return sampled
                    except Exception:  # noqa: S110
                        pass

        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(self.df), size=self.sample_limit, replace=False)
        return self.df.iloc[idx].copy()

    def _evaluate_association(self, series: pd.Series, target_series: pd.Series) -> float:
        valid_mask = series.notna() & target_series.notna() & ~np.isinf(series)
        if valid_mask.sum() < 10:
            return 0.0

        x_clean = series[valid_mask].to_numpy()
        y_clean = target_series[valid_mask].to_numpy()

        if np.std(x_clean) == 0:
            return 0.0

        try:
            if self.target_type == "classification":
                f_val, _ = f_classif(x_clean.reshape(-1, 1), y_clean)
                score = float(np.log1p(max(0.0, f_val[0])))
            else:
                r, _ = stats.pearsonr(x_clean, y_clean.astype(float))
                score = abs(float(r))
            return score if not np.isnan(score) else 0.0
        except Exception:
            return 0.0

    def run(self) -> InteractionReport:
        if self.target is None or self.target not in self.df.columns:
            return InteractionReport(
                target=None,
                target_type=self.target_type,
                top_interactions=[],
                best_per_pair=[],
            )

        # Subsample evaluation dataframe if large
        eval_df = self._subsample_eval_df()
        y = eval_df[self.target]

        num_feats = [
            f
            for f in self.features
            if pd.api.types.is_numeric_dtype(eval_df[f])
            and not pd.api.types.is_bool_dtype(eval_df[f])
        ]
        if len(num_feats) < 2:
            return InteractionReport(
                target=self.target,
                target_type=self.target_type,
                top_interactions=[],
                best_per_pair=[],
            )

        # Baselines
        baselines = {f: self._evaluate_association(eval_df[f], y) for f in num_feats}

        pairs = list(itertools.combinations(num_feats, 2))
        if self.max_pairs is not None and len(pairs) > self.max_pairs:
            pairs.sort(key=lambda p: baselines[p[0]] + baselines[p[1]], reverse=True)
            pairs = pairs[: self.max_pairs]

        all_interactions: list[InteractionResult] = []
        best_per_pair: dict[tuple[str, str], InteractionResult] = {}
        metric_name = (
            "log(1 + F-statistic)" if self.target_type == "classification" else "Pearson |r|"
        )

        for fa, fb in pairs:
            sa = eval_df[fa].astype(float)
            sb = eval_df[fb].astype(float)
            base_score = max(baselines[fa], baselines[fb])

            eps = 1e-6
            ops = [
                ("Multiplication", f"{fa} * {fb}", sa * sb),
                ("Ratio (A/B)", f"{fa} / ({fb} + eps)", sa / (sb.replace(0, eps) + eps)),
                ("Ratio (B/A)", f"{fb} / ({fa} + eps)", sb / (sa.replace(0, eps) + eps)),
                ("Sum", f"{fa} + {fb}", sa + sb),
                ("Difference", f"{fa} - {fb}", sa - sb),
            ]

            for op_name, formula, syn_s in ops:
                comb_score = self._evaluate_association(syn_s, y)
                gain = round(comb_score - base_score, 4)

                res = InteractionResult(
                    feature_a=fa,
                    feature_b=fb,
                    operation=op_name,
                    formula=formula,
                    combined_score=round(comb_score, 4),
                    baseline_score=round(base_score, 4),
                    synergy_gain=gain,
                    metric_name=metric_name,
                )
                all_interactions.append(res)

                current_best = best_per_pair.get((fa, fb))
                if current_best is None or gain > current_best.synergy_gain:
                    best_per_pair[(fa, fb)] = res

        all_interactions.sort(key=lambda x: x.synergy_gain, reverse=True)
        top_pairs = sorted(best_per_pair.values(), key=lambda x: x.synergy_gain, reverse=True)

        return InteractionReport(
            target=self.target,
            target_type=self.target_type,
            top_interactions=all_interactions,
            best_per_pair=top_pairs,
        )
