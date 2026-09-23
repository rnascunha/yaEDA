from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


@dataclass
class ArithmeticInteraction:
    feature_a: str
    feature_b: str
    operation: str
    formula: str
    baseline_score: float
    combined_score: float
    synergy_gain: float
    metric_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionReport:
    target: str
    target_type: Literal["classification", "regression"]
    top_interactions: list[ArithmeticInteraction]
    best_per_pair: list[ArithmeticInteraction]

    def to_dataframe(self) -> pd.DataFrame:
        records = [item.to_dict() for item in self.top_interactions]
        return pd.DataFrame(records)


class FeatureInteractionAnalyzer:
    """Evaluates arithmetic synergies (Sum, Difference, Product, Ratio) across all feature pairs."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] = "regression",
        max_pairs: int | None = None,
        random_state: int = 42,
    ):
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame.")

        self.df = df
        self.target = target
        if features is not None:
            self.features = [f for f in features if f in df.columns and f != target]
        else:
            self.features = [col for col in df.columns if col != target]
        self.target_type = target_type
        self.max_pairs = max_pairs
        self.random_state = random_state

    def _eval_association(self, series: pd.Series, target_series: pd.Series) -> tuple[float, str]:
        valid = pd.concat([series, target_series], axis=1).dropna()
        if len(valid) < 5 or valid.iloc[:, 0].nunique() <= 1:
            return 0.0, "None"

        x = valid.iloc[:, 0]
        y = valid.iloc[:, 1]

        if self.target_type == "regression":
            try:
                r, _ = stats.pearsonr(x, y)
                return (0.0 if np.isnan(r) else abs(float(r))), "Pearson |r|"
            except Exception:
                return 0.0, "Pearson |r|"
        else:
            try:
                y_enc = pd.factorize(y)[0]
                mi = mutual_info_classif(
                    x.to_numpy().reshape(-1, 1),
                    y_enc,
                    random_state=self.random_state,
                )[0]
                return float(mi), "Mutual Information"
            except Exception:
                return 0.0, "Mutual Information"

    def run(self) -> InteractionReport:
        if self.target is None or self.target not in self.df.columns:
            return InteractionReport(
                target=None,
                target_type=self.target_type,
                top_interactions=[],
                best_per_pair=[],
            )

        num_feats = [
            f
            for f in self.features
            if pd.api.types.is_numeric_dtype(self.df[f])
            and not pd.api.types.is_bool_dtype(self.df[f])
        ]

        target_s = self.df[self.target]
        base_scores = {f: self._eval_association(self.df[f], target_s)[0] for f in num_feats}

        all_pairs = list(combinations(num_feats, 2))
        pairs = all_pairs[: self.max_pairs] if self.max_pairs else all_pairs

        all_interactions: list[ArithmeticInteraction] = []
        best_per_pair: list[ArithmeticInteraction] = []

        for fa, fb in pairs:
            sa = self.df[fa].astype(float)
            sb = self.df[fb].astype(float)
            baseline = max(base_scores.get(fa, 0.0), base_scores.get(fb, 0.0))

            iqr_b = sb.quantile(0.75) - sb.quantile(0.25)
            eps_b = (iqr_b * 1e-4) if iqr_b > 0 else 1e-4
            iqr_a = sa.quantile(0.75) - sa.quantile(0.25)
            eps_a = (iqr_a * 1e-4) if iqr_a > 0 else 1e-4

            candidates = {
                "Multiplication": (sa * sb, f"{fa} * {fb}"),
                "Ratio (A/B)": (
                    sa / (sb.replace(0, np.nan) + np.sign(sb) * eps_b),
                    f"{fa} / ({fb} + ε)",
                ),
                "Ratio (B/A)": (
                    sb / (sa.replace(0, np.nan) + np.sign(sa) * eps_a),
                    f"{fb} / ({fa} + ε)",
                ),
                "Sum": (sa + sb, f"{fa} + {fb}"),
                "Difference": (sa - sb, f"{fa} - {fb}"),
            }

            pair_candidates: list[ArithmeticInteraction] = []
            for op_name, (comb_series, formula) in candidates.items():
                comb_clean = comb_series.replace([np.inf, -np.inf], np.nan)
                score, metric_name = self._eval_association(comb_clean, target_s)
                synergy = score - baseline

                item = ArithmeticInteraction(
                    feature_a=fa,
                    feature_b=fb,
                    operation=op_name,
                    formula=formula,
                    baseline_score=round(baseline, 4),
                    combined_score=round(score, 4),
                    synergy_gain=round(synergy, 4),
                    metric_name=metric_name,
                )
                all_interactions.append(item)
                pair_candidates.append(item)

            if pair_candidates:
                pair_candidates.sort(key=lambda x: x.synergy_gain, reverse=True)
                best_per_pair.append(pair_candidates[0])

        all_interactions.sort(key=lambda x: x.synergy_gain, reverse=True)
        best_per_pair.sort(key=lambda x: x.synergy_gain, reverse=True)

        return InteractionReport(
            target=self.target,
            target_type=self.target_type,
            top_interactions=all_interactions,
            best_per_pair=best_per_pair,
        )
