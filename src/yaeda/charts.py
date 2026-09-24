import base64
import io
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EDAChartGenerator:
    """Generates self-contained Base64-encoded PNG charts for embedding in reports."""

    def __init__(self, theme: str = "whitegrid"):
        sns.set_theme(style=theme)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 9,
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "figure.autolayout": False,
            }
        )
        self.dataset_palette = ["#2563eb", "#d97706", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4"]

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def plot_golden_features(self, importances: list[Any], top_n: int = 10) -> str:
        records = importances[:top_n][::-1]
        if not records:
            return ""

        names = [r.feature for r in records]
        golden_scores = [r.golden_feature_score for r in records]
        perm_drops = [r.permutation_mean for r in records]

        fig, ax = plt.subplots(figsize=(8, max(3.5, len(names) * 0.42)))
        y_pos = np.arange(len(names))
        bar_height = 0.35

        bars_golden = ax.barh(
            y_pos + bar_height / 2,
            golden_scores,
            bar_height,
            label="Golden Feature Score",
            color="#d97706",
        )
        ax.barh(
            y_pos - bar_height / 2,
            perm_drops,
            bar_height,
            label="Permutation Drop (Val)",
            color="#2563eb",
            alpha=0.85,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontweight="bold")
        ax.set_xlabel("Impact Metric / Score")
        ax.set_title("Golden Features Leaderboard (Top Predictors)")
        ax.legend(loc="lower right")
        ax.set_xlim(0, max(max(golden_scores, default=1.0) * 1.15, 0.1))

        for bar in bars_golden:
            w = bar.get_width()
            ax.text(
                w + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center",
                fontsize=8,
                color="#92400e",
                fontweight="bold",
            )

        return self._fig_to_base64(fig)

    def plot_target_associations(self, associations: dict[str, Any], top_n: int = 10) -> str:
        if not associations:
            return ""

        def get_score(v):
            mi = (
                getattr(v, "mutual_info", None)
                if hasattr(v, "mutual_info")
                else v.get("mutual_info")
            )
            p_r = (
                getattr(v, "pearson_corr", None)
                if hasattr(v, "pearson_corr")
                else v.get("pearson_corr")
            )
            return (
                mi if mi is not None else -1.0,
                abs(p_r) if p_r is not None else -1.0,
            )

        sorted_items = sorted(associations.items(), key=lambda x: get_score(x[1]), reverse=True)[
            :top_n
        ]
        if not sorted_items:
            return ""

        feats = [item[0] for item in sorted_items]
        mi_vals = []
        pearson_vals = []
        is_num_flags = []

        for feat, v in sorted_items:
            mi = (
                getattr(v, "mutual_info", None)
                if hasattr(v, "mutual_info")
                else v.get("mutual_info")
            )
            p_r = (
                getattr(v, "pearson_corr", None)
                if hasattr(v, "pearson_corr")
                else v.get("pearson_corr")
            )
            is_num = (
                getattr(v, "is_numeric", True)
                if hasattr(v, "is_numeric")
                else v.get("is_numeric", True)
            )

            mi_vals.append(float(mi) if mi is not None else 0.0)
            pearson_vals.append(float(p_r) if p_r is not None else None)
            is_num_flags.append(is_num)

        has_correlations = any(p is not None for p in pearson_vals)
        y_pos = np.arange(len(feats))
        bar_colors = ["#059669" if num else "#8b5cf6" for num in is_num_flags]

        if has_correlations:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(9.5, max(3.5, len(feats) * 0.42)), sharey=True
            )

            bars1 = ax1.barh(y_pos, mi_vals, color=bar_colors, height=0.55)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(feats, fontweight="bold")
            ax1.invert_yaxis()
            ax1.set_title("Mutual Information $I(X; Y)$\n(Non-Linear / All Features)")
            ax1.set_xlabel("Dependency Score")
            ax1.grid(True, linestyle="--", alpha=0.5)

            for bar in bars1:
                w = bar.get_width()
                ax1.text(
                    w + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}",
                    va="center",
                    fontsize=8,
                    color="#065f46",
                )

            clean_pearson = [p if p is not None else 0.0 for p in pearson_vals]
            corr_colors = ["#2563eb" if p >= 0 else "#dc2626" for p in clean_pearson]
            bars2 = ax2.barh(y_pos, clean_pearson, color=corr_colors, height=0.55)
            ax2.set_title("Directional Correlation ($r$)\n(Linear / Point-Biserial)")
            ax2.set_xlabel("Pearson Correlation")
            ax2.set_xlim(-1.1, 1.1)
            ax2.axvline(0, color="gray", linestyle="-", linewidth=0.8)
            ax2.grid(True, linestyle="--", alpha=0.5)

            for i, bar in enumerate(bars2):
                val = pearson_vals[i]
                if val is not None:
                    offset = 0.04 if val >= 0 else -0.18
                    ax2.text(
                        val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.2f}",
                        va="center",
                        fontsize=8,
                        color="#1e293b",
                        fontweight="bold",
                    )
                else:
                    ax2.text(
                        0.04,
                        bar.get_y() + bar.get_height() / 2,
                        "[Categorical]",
                        va="center",
                        fontsize=8,
                        color="#64748b",
                        fontstyle="italic",
                    )

            fig.suptitle(
                "Feature vs. Target: Non-Linear (MI) vs. Directional Linear ($r$)",
                fontsize=11,
                fontweight="bold",
                y=1.02,
            )
        else:
            fig, ax = plt.subplots(figsize=(8, max(3.5, len(feats) * 0.42)))
            bars = ax.barh(y_pos, mi_vals, color=bar_colors, height=0.55)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feats, fontweight="bold")
            ax.invert_yaxis()
            ax.set_title(
                "Feature-Target Association Strength (Mutual Information)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Mutual Information Score $I(X; Y)$")
            ax.grid(True, linestyle="--", alpha=0.5)

            for bar in bars:
                w = bar.get_width()
                ax.text(
                    w + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}",
                    va="center",
                    fontsize=8,
                    color="#065f46",
                    fontweight="bold",
                )

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_correlation_heatmap(
        self, corr_matrix: dict[str, dict[str, float]], max_feats: int = 12
    ) -> str:
        df_corr = pd.DataFrame(corr_matrix)
        if df_corr.empty:
            return ""

        cols = df_corr.columns[:max_feats]
        sub_df = df_corr.loc[cols, cols]

        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.heatmap(
            sub_df,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title("Inter-Feature Multicollinearity Matrix", pad=12)
        plt.xticks(rotation=45, ha="right")
        return self._fig_to_base64(fig)

    def plot_data_health(self, features: dict[str, Any], target_col: str | None = None) -> str:
        flagged = []
        for feat, p in features.items():
            if feat == target_col:
                continue
            outlier_pct = p.outliers.percentage if p.outliers else 0.0
            if p.missing_percentage > 0 or p.zero_percentage > 5 or outlier_pct > 2:
                flagged.append((feat, p.missing_percentage, p.zero_percentage, outlier_pct))

        if not flagged:
            flagged = [
                (
                    f,
                    p.missing_percentage,
                    p.zero_percentage,
                    p.outliers.percentage if p.outliers else 0.0,
                )
                for f, p in list(features.items())[:8]
                if f != target_col
            ]

        flagged = flagged[:10]
        names = [x[0] for x in flagged]
        m_vals = [x[1] for x in flagged]
        z_vals = [x[2] for x in flagged]
        o_vals = [x[3] for x in flagged]

        fig, ax = plt.subplots(figsize=(8, max(3.5, len(names) * 0.42)))
        y_pos = np.arange(len(names))
        height = 0.25

        ax.barh(y_pos - height, m_vals, height, label="Missing %", color="#ef4444")
        ax.barh(y_pos, z_vals, height, label="Zeros %", color="#6b7280")
        ax.barh(y_pos + height, o_vals, height, label="Outliers %", color="#f59e0b")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontweight="bold")
        ax.set_xlabel("Percentage (%)")
        ax.set_title("Data Quality & Anomaly Distribution")
        ax.legend(loc="lower right")

        return self._fig_to_base64(fig)

    def plot_feature_distribution_comparison(
        self,
        datasets: dict[str, pd.DataFrame],
        feature: str,
        is_numeric: bool = True,
    ) -> str:
        """
        Renders a full-width comparative chart of a feature across multiple datasets:
          - Numeric: Overlaid KDE density curves and semi-transparent step histograms with mean/std markers.
          - Categorical: Grouped percentage bar chart with [UNSEEN] callouts for test-set novel levels.
        """
        valid_dfs = {
            name: df[feature].dropna()
            for name, df in datasets.items()
            if feature in df.columns and not df[feature].dropna().empty
        }
        if not valid_dfs:
            return ""

        colors = {
            name: self.dataset_palette[i % len(self.dataset_palette)]
            for i, name in enumerate(valid_dfs.keys())
        }
        fig, ax = plt.subplots(figsize=(8.8, 3.4))

        if is_numeric:
            all_unique = set()
            zero_vars = True
            for s in valid_dfs.values():
                all_unique.update(s.unique()[:25])
                if s.std() > 1e-6:
                    zero_vars = False
            is_discrete = (len(all_unique) <= 10) or zero_vars

            if is_discrete:
                prop_data = []
                for name, s in valid_dfs.items():
                    vc = s.value_counts(normalize=True) * 100
                    for val, pct in vc.items():
                        prop_data.append({"Dataset": name, "Value": str(val), "Percentage": pct})
                pdf = pd.DataFrame(prop_data)
                sns.barplot(
                    data=pdf, x="Value", y="Percentage", hue="Dataset", palette=colors, ax=ax
                )
                ax.set_title(
                    f"Comparative Distribution (Discrete): {feature}",
                    fontsize=10,
                    fontweight="bold",
                )
                ax.set_ylabel("Percentage (%)", fontsize=8.5)
                if len(all_unique) > 5:
                    ax.tick_params(axis="x", rotation=30)
            else:
                for name, s in valid_dfs.items():
                    m_val, s_val = float(s.mean()), float(s.std())
                    lbl = f"{name} (μ={m_val:.2g}, σ={s_val:.2g})"
                    try:
                        sns.kdeplot(s, ax=ax, label=lbl, color=colors[name], linewidth=2.0)
                        sns.histplot(
                            s,
                            ax=ax,
                            color=colors[name],
                            stat="density",
                            alpha=0.12,
                            element="step",
                            fill=True,
                        )
                    except Exception:
                        ax.axvline(m_val, color=colors[name], label=lbl, linestyle="--")
                ax.set_title(
                    f"Comparative Density (KDE): {feature}", fontsize=10, fontweight="bold"
                )
                ax.set_ylabel("Density", fontsize=8.5)
                ax.legend(fontsize=8, loc="upper right")

        else:
            prop_data = []
            primary_name = next(iter(valid_dfs.keys()))
            primary_cats = set(valid_dfs[primary_name].unique())

            for name, s in valid_dfs.items():
                vc = s.value_counts(normalize=True) * 100
                for val, pct in vc.items():
                    is_unseen = (name != primary_name) and (val not in primary_cats)
                    prop_data.append(
                        {
                            "Dataset": name,
                            "Category": f"{val} [UNSEEN]" if is_unseen else str(val),
                            "Percentage": pct,
                            "RawVal": str(val),
                            "Unseen": is_unseen,
                        }
                    )

            pdf = pd.DataFrame(prop_data)
            top_cats = valid_dfs[primary_name].value_counts().index.tolist()
            all_cats = list(
                dict.fromkeys(top_cats + [x for x in pdf["RawVal"].unique() if x not in top_cats])
            )[:10]
            pdf = pdf[pdf["RawVal"].isin(all_cats)]

            order = [f"{c} [UNSEEN]" if (c not in primary_cats) else str(c) for c in all_cats]
            sns.barplot(
                data=pdf,
                y="Category",
                x="Percentage",
                hue="Dataset",
                palette=colors,
                ax=ax,
                order=order,
            )
            ax.set_title(
                f"Category Proportion Comparison: {feature}", fontsize=10, fontweight="bold"
            )
            ax.set_xlabel("Proportion (%)", fontsize=8.5)
            ax.set_ylabel("Category", fontsize=8.5)
            ax.legend(fontsize=8, loc="lower right")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel(feature, fontsize=8.5)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_dataset_drift_overview(
        self,
        comparisons: dict[str, list[Any]],
        secondary_name: str | None = None,
        top_n: int = 10,
    ) -> str:
        """
        Renders a dual-panel overview summarizing missingness drift (Δ missing %)
        and unseen category counts between the Primary dataset and a Secondary dataset.
        """
        if not comparisons:
            return ""

        target_sec = secondary_name or next(iter(comparisons.keys()))
        comp_list = comparisons.get(target_sec, [])
        if not comp_list:
            return ""

        flagged = []
        for c in comp_list:
            delta_m = getattr(c, "delta_missing_pct", 0.0) or 0.0
            p_m = getattr(c, "primary_missing_pct", 0.0) or 0.0
            s_m = getattr(c, "secondary_missing_pct", 0.0) or 0.0
            unseen_count = len(getattr(c, "unseen_categories", []))
            flagged.append(
                {
                    "feature": c.feature,
                    "primary_missing": p_m,
                    "secondary_missing": s_m,
                    "delta_missing": delta_m,
                    "abs_delta": abs(delta_m),
                    "unseen_count": unseen_count,
                }
            )

        flagged.sort(key=lambda x: (x["unseen_count"] > 0, x["abs_delta"]), reverse=True)
        top_items = flagged[:top_n]
        if not top_items:
            return ""

        feats = [item["feature"] for item in top_items][::-1]
        p_vals = [item["primary_missing"] for item in top_items][::-1]
        s_vals = [item["secondary_missing"] for item in top_items][::-1]
        unseen_counts = [item["unseen_count"] for item in top_items][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, max(3.5, len(feats) * 0.40)))
        y_pos = np.arange(len(feats))
        height = 0.35

        # Panel 1: Missing Rate Comparison
        ax1.barh(y_pos - height / 2, p_vals, height, label="Primary", color="#2563eb", alpha=0.85)
        ax1.barh(y_pos + height / 2, s_vals, height, label=target_sec, color="#d97706", alpha=0.85)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feats, fontweight="bold", fontsize=8.5)
        ax1.set_xlabel("Missing Percentage (%)", fontsize=8.5)
        ax1.set_title(f"Missingness: Primary vs {target_sec}", fontsize=10, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Panel 2: Net Missing Delta & Unseen Level Flags
        deltas = [item["delta_missing"] for item in top_items][::-1]
        bar_colors = ["#dc2626" if d > 0 else ("#16a34a" if d < 0 else "#64748b") for d in deltas]
        bars = ax2.barh(y_pos, deltas, height=0.55, color=bar_colors, alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([""] * len(feats))
        ax2.axvline(0, color="gray", linestyle="-", linewidth=0.8)
        ax2.set_xlabel("Δ Missing % (Secondary - Primary)", fontsize=8.5)
        ax2.set_title("Shift & Novel Category Alerts", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)

        for i, bar in enumerate(bars):
            w = bar.get_width()
            offset = 0.5 if w >= 0 else -0.5
            ha = "left" if w >= 0 else "right"
            unseen_flag = f" [+{unseen_counts[i]} unseen!]" if unseen_counts[i] > 0 else ""
            ax2.text(
                w + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{w:+.1f}%{unseen_flag}",
                va="center",
                ha=ha,
                fontsize=7.5,
                fontweight="bold",
                color="#1e293b",
            )

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_feature_summary_card(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str | None = None,
        is_feat_numeric: bool = True,
        is_target_numeric: bool = False,
        secondary_dfs: dict[str, pd.DataFrame] | None = None,
        primary_name: str = "Primary",
    ) -> str:
        """
        Renders a dual-panel card visualization:
          - If secondary_dfs provided: Panel 1 overlays distributions across datasets.
          - If target provided: Panel 2 displays the feature's relationship to the target in the primary dataset.
          - If target is None: Panel 2 displays side-by-side quantile boxplots across datasets (or single panel if 1 dataset).
        """
        if feature not in df.columns or df[feature].dropna().empty:
            return ""

        has_target = (target is not None) and (target in df.columns) and (feature != target)
        has_secondary = bool(secondary_dfs)

        if not has_target and not has_secondary:
            fig, ax1 = plt.subplots(figsize=(6.2, 3.2))
            ax2 = None
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))

        # --- 1. Distribution Plot on ax1 ---
        all_dfs = {primary_name: df}
        if secondary_dfs:
            for s_name, s_df in secondary_dfs.items():
                if feature in s_df.columns:
                    all_dfs[s_name] = s_df

        if len(all_dfs) > 1:
            # Overlaid comparison on ax1
            colors = {
                name: self.dataset_palette[i % len(self.dataset_palette)]
                for i, name in enumerate(all_dfs.keys())
            }
            if is_feat_numeric:
                for name, d in all_dfs.items():
                    s = d[feature].dropna()
                    if s.empty:
                        continue
                    m_val, _us_val = float(s.mean()), float(s.std())
                    lbl = f"{name} (μ={m_val:.2g})"
                    try:
                        sns.kdeplot(s, ax=ax1, label=lbl, color=colors[name], linewidth=1.8)
                        sns.histplot(
                            s,
                            ax=ax1,
                            color=colors[name],
                            stat="density",
                            alpha=0.10,
                            element="step",
                        )
                    except Exception:
                        ax1.axvline(m_val, color=colors[name], label=lbl, linestyle="--")
                ax1.set_title(f"{feature} Density Comparison", fontsize=9.5, fontweight="bold")
                ax1.set_ylabel("Density")
                ax1.legend(fontsize=7.5, loc="upper right")
            else:
                prop_data = []
                p_cats = set(df[feature].dropna().unique())
                for name, d in all_dfs.items():
                    vc = d[feature].dropna().value_counts(normalize=True) * 100
                    for val, pct in vc.head(6).items():
                        is_unseen = (name != primary_name) and (val not in p_cats)
                        prop_data.append(
                            {
                                "Dataset": name,
                                "Category": f"{val}*" if is_unseen else str(val),
                                "Percentage": pct,
                            }
                        )
                pdf = pd.DataFrame(prop_data)
                sns.barplot(
                    data=pdf, x="Category", y="Percentage", hue="Dataset", palette=colors, ax=ax1
                )
                ax1.set_title(f"{feature} Proportions (*=Unseen)", fontsize=9.5, fontweight="bold")
                ax1.set_ylabel("Percentage (%)")
                ax1.tick_params(axis="x", rotation=30)
                ax1.legend(fontsize=7.5, loc="upper right")
        else:
            # Single dataset distribution on ax1
            s = df[feature].dropna()
            n_unique = s.nunique()
            if is_feat_numeric:
                if n_unique <= 10 or (s.std() == 0 if len(s) > 1 else True):
                    counts = s.value_counts().sort_index()
                    ax1.bar(
                        [str(x) for x in counts.index], counts.values, color="#3b82f6", alpha=0.85
                    )
                    ax1.set_title(
                        f"Distribution: {feature} (Discrete)", fontsize=9.5, fontweight="bold"
                    )
                    ax1.set_ylabel("Count")
                    if len(counts) > 5:
                        ax1.tick_params(axis="x", rotation=30)
                else:
                    bins = min(30, max(12, int(np.sqrt(len(s)))))
                    sns.histplot(
                        s, kde=True, ax=ax1, color="#2563eb", bins=bins, stat="density", alpha=0.55
                    )
                    mean_val = float(s.mean())
                    med_val = float(s.median())
                    ax1.axvline(
                        mean_val,
                        color="#dc2626",
                        linestyle="--",
                        linewidth=1.2,
                        label=f"Mean: {mean_val:.2g}",
                    )
                    ax1.axvline(
                        med_val,
                        color="#16a34a",
                        linestyle=":",
                        linewidth=1.2,
                        label=f"Median: {med_val:.2g}",
                    )
                    ax1.set_title(f"Distribution: {feature}", fontsize=9.5, fontweight="bold")
                    ax1.set_ylabel("Density")
                    ax1.legend(fontsize=7.5, loc="upper right")
            else:
                top_counts = s.value_counts().head(8)
                y_pos = np.arange(len(top_counts))
                bars = ax1.barh(y_pos, top_counts.values, color="#8b5cf6", height=0.6, alpha=0.85)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels([str(k)[:16] for k in top_counts.index], fontsize=8)
                ax1.invert_yaxis()
                ax1.set_title(f"Top Categories: {feature}", fontsize=9.5, fontweight="bold")
                ax1.set_xlabel("Count")
                total_s = len(s)
                max_w = max(top_counts.values) if not top_counts.empty else 1
                for bar in bars:
                    w = bar.get_width()
                    pct = (w / total_s) * 100
                    ax1.text(
                        w + (max_w * 0.02),
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(w):,} ({pct:.1f}%)",
                        va="center",
                        fontsize=7.5,
                        color="#4c1d95",
                    )
                ax1.set_xlim(0, max_w * 1.3)

        ax1.grid(True, linestyle="--", alpha=0.4)

        # --- 2. Relationship or Dataset Comparison on ax2 ---
        if ax2 is not None:
            if has_target:
                # Relationship to target
                sub = df[[feature, target]].dropna()
                if sub.empty:
                    ax2.text(
                        0.5,
                        0.5,
                        "No overlapping valid rows",
                        ha="center",
                        va="center",
                        color="#94a3b8",
                    )
                    ax2.set_title(f"{feature} vs. {target}", fontsize=9.5, fontweight="bold")
                else:
                    if len(sub) > 1500:
                        sub = sub.sample(1500, random_state=42)

                    if is_feat_numeric and is_target_numeric:
                        if sub[feature].nunique() <= 10:
                            sns.lineplot(
                                data=sub,
                                x=feature,
                                y=target,
                                ax=ax2,
                                marker="o",
                                color="#2563eb",
                                errorbar="sd",
                            )
                            ax2.set_title(
                                f"{feature} vs. {target} (Line & SD)",
                                fontsize=9.5,
                                fontweight="bold",
                            )
                        else:
                            sns.regplot(
                                data=sub,
                                x=feature,
                                y=target,
                                ax=ax2,
                                scatter_kws={"alpha": 0.35, "s": 15, "color": "#2563eb"},
                                line_kws={"color": "#dc2626", "linewidth": 1.4},
                            )
                            ax2.set_title(
                                f"{feature} vs. {target} (Trend)", fontsize=9.5, fontweight="bold"
                            )

                    elif is_feat_numeric and not is_target_numeric:
                        target_cardinality = sub[target].nunique()
                        if (
                            target_cardinality <= 4
                            and len(sub) >= 40
                            and sub[feature].nunique() > 10
                        ):
                            sns.violinplot(
                                data=sub,
                                x=target,
                                y=feature,
                                ax=ax2,
                                palette="crest",
                                inner="quartile",
                                cut=0,
                                hue=target,
                                legend=False,
                            )
                            ax2.set_title(
                                f"{feature} by {target} (Violin)", fontsize=9.5, fontweight="bold"
                            )
                        elif sub[feature].nunique() <= 5:
                            sns.stripplot(
                                data=sub,
                                x=target,
                                y=feature,
                                ax=ax2,
                                jitter=0.25,
                                alpha=0.6,
                                palette="crest",
                                hue=target,
                                legend=False,
                            )
                            ax2.set_title(
                                f"{feature} by {target} (Strip)", fontsize=9.5, fontweight="bold"
                            )
                        else:
                            sns.boxplot(
                                data=sub,
                                x=target,
                                y=feature,
                                ax=ax2,
                                palette="crest",
                                showmeans=True,
                                meanprops={
                                    "marker": "o",
                                    "markerfacecolor": "white",
                                    "markeredgecolor": "#0f172a",
                                },
                                hue=target,
                                legend=False,
                            )
                            ax2.set_title(
                                f"{feature} by {target} (Box)", fontsize=9.5, fontweight="bold"
                            )
                        if target_cardinality > 4:
                            ax2.tick_params(axis="x", rotation=30)

                    elif not is_feat_numeric and not is_target_numeric:
                        top_f = sub[feature].value_counts().head(7).index
                        top_t = sub[target].value_counts().head(5).index
                        sub_ct = sub[sub[feature].isin(top_f) & sub[target].isin(top_t)]
                        ct = pd.crosstab(sub_ct[feature], sub_ct[target])

                        if ct.shape[0] * ct.shape[1] <= 20 and not ct.empty:
                            ct_norm = (
                                pd.crosstab(sub_ct[feature], sub_ct[target], normalize="index")
                                * 100
                            )
                            sns.heatmap(
                                ct_norm,
                                annot=True,
                                fmt=".1f",
                                cmap="Blues",
                                cbar_kws={"label": "Row %", "shrink": 0.8},
                                ax=ax2,
                                annot_kws={"size": 8},
                            )
                            ax2.set_title(
                                f"{feature} vs. {target} (% Heatmap)",
                                fontsize=9.5,
                                fontweight="bold",
                            )
                            ax2.tick_params(axis="x", rotation=30)
                        else:
                            ct_norm = (
                                pd.crosstab(sub_ct[feature], sub_ct[target], normalize="index")
                                * 100
                            )
                            ct_norm.plot(
                                kind="barh", stacked=True, ax=ax2, colormap="tab10", alpha=0.85
                            )
                            ax2.set_title(
                                f"{feature} vs. {target} (Stacked %)",
                                fontsize=9.5,
                                fontweight="bold",
                            )
                            ax2.set_xlabel("Percentage (%)", fontsize=8)

                    else:
                        top_f = sub[feature].value_counts().head(6).index
                        sub_f = sub[sub[feature].isin(top_f)]
                        if len(top_f) <= 5:
                            sns.boxplot(
                                data=sub_f,
                                x=feature,
                                y=target,
                                ax=ax2,
                                palette="Blues_r",
                                showmeans=True,
                                meanprops={
                                    "marker": "o",
                                    "markerfacecolor": "white",
                                    "markeredgecolor": "#0f172a",
                                },
                                hue=feature,
                                legend=False,
                            )
                            ax2.set_title(
                                f"{target} by {feature} (Box)", fontsize=9.5, fontweight="bold"
                            )
                        else:
                            sns.barplot(
                                data=sub_f,
                                x=feature,
                                y=target,
                                ax=ax2,
                                palette="Blues_r",
                                errorbar="se",
                            )
                            ax2.set_title(
                                f"Mean {target} by {feature} (±SE)", fontsize=9.5, fontweight="bold"
                            )
                        if len(top_f) > 3:
                            ax2.tick_params(axis="x", rotation=30)

            elif has_secondary:
                # No target, but secondary datasets exist: show side-by-side quantile boxplot
                if is_feat_numeric:
                    box_records = []
                    for name, d in all_dfs.items():
                        vals = d[feature].dropna()
                        for v in vals:
                            box_records.append({"Dataset": name, "Value": float(v)})
                    if box_records:
                        b_df = pd.DataFrame(box_records)
                        colors = [
                            self.dataset_palette[i % len(self.dataset_palette)]
                            for i in range(len(all_dfs))
                        ]
                        sns.boxplot(
                            data=b_df,
                            x="Dataset",
                            y="Value",
                            palette=colors,
                            ax=ax2,
                            hue="Dataset",
                            legend=False,
                        )
                        ax2.set_title(
                            f"Quantile Comparison: {feature}", fontsize=9.5, fontweight="bold"
                        )
                        ax2.set_ylabel(feature)
                else:
                    # Categorical: Cardinality & Unseen counts
                    cats_summary = []
                    for name, d in all_dfs.items():
                        cats_summary.append(
                            {
                                "Dataset": name,
                                "Distinct": int(d[feature].nunique(dropna=True)),
                            }
                        )
                    c_df = pd.DataFrame(cats_summary)
                    sns.barplot(
                        data=c_df,
                        x="Dataset",
                        y="Distinct",
                        palette=self.dataset_palette[: len(all_dfs)],
                        ax=ax2,
                        hue="Dataset",
                        legend=False,
                    )
                    ax2.set_title(f"Distinct Levels: {feature}", fontsize=9.5, fontweight="bold")
                    ax2.set_ylabel("Cardinality")

            ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_partial_dependence(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        features_to_plot: Any,
        target_type: str = "regression",
        kind: str = "both",
        subsample_ice: int = 50,
        grid_resolution: int = 25,
        max_cols: int = 3,
    ) -> str:
        """
        Renders a multi-panel Partial Dependence Plot (PDP) with ICE curves
        directly using sklearn.inspection.partial_dependence, avoiding
        PartialDependenceDisplay axis-reshaping dimension conflicts.
        """
        from sklearn.inspection import partial_dependence

        if model is None or X is None or features_to_plot is None:
            return ""

        # 1. Flatten feature list defensively
        flat_feats: list[str] = []
        if isinstance(features_to_plot, (list, tuple, set)):
            for item in features_to_plot:
                if isinstance(item, (list, tuple, set)):
                    flat_feats.extend([str(x) for x in item])
                elif isinstance(item, str):
                    flat_feats.append(item)
        elif isinstance(features_to_plot, str):
            flat_feats = [features_to_plot]

        clean_feats = list(dict.fromkeys(flat_feats))
        valid_feats = [f for f in clean_feats if f in feature_names]
        if not valid_feats:
            return ""

        feat_indices = [feature_names.index(f) for f in valid_feats]
        n_feats = len(feat_indices)
        n_cols = min(max_cols, n_feats)
        n_rows = int(np.ceil(n_feats / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5 * n_cols, 3.3 * n_rows),
            squeeze=False,
        )
        ax_flat = axes.ravel()

        # 2. Subsample evaluation matrix for fast calculation
        if len(X) > 600:
            rng = np.random.RandomState(42)
            eval_idx = rng.choice(len(X), size=600, replace=False)
            X_eval = X[eval_idx]
        else:
            X_eval = X

        # 3. Compute and plot curves feature-by-feature
        for i, feat_idx in enumerate(feat_indices):
            ax = ax_flat[i]
            feat_name = valid_feats[i]

            try:
                pdp_res = partial_dependence(
                    estimator=model,
                    X=X_eval,
                    features=[feat_idx],
                    grid_resolution=grid_resolution,
                    kind=kind,
                )

                # Extract grid (x-axis values)
                grid_candidates = pdp_res.get("grid_values", pdp_res.get("values", None))
                if grid_candidates is None:
                    grid_candidates = getattr(
                        pdp_res, "grid_values", getattr(pdp_res, "values", None)
                    )
                x_vals = grid_candidates[0]

                # Extract average (mean PDP curve)
                avg_arr = pdp_res.get("average", getattr(pdp_res, "average", None))
                # For binary classification with shape (2, N), pick index 1 (positive class).
                # For regression with shape (1, N), index -1 cleanly selects row 0.
                y_mean = avg_arr[-1] if avg_arr.ndim == 2 else avg_arr

                # Plot Individual Conditional Expectation (ICE) curves if requested
                if kind in ("both", "individual"):
                    indiv_arr = pdp_res.get("individual", getattr(pdp_res, "individual", None))
                    if indiv_arr is not None:
                        ice_curves = indiv_arr[-1] if indiv_arr.ndim == 3 else indiv_arr
                        n_ice = min(len(ice_curves), subsample_ice)
                        for c_idx in range(n_ice):
                            ax.plot(
                                x_vals,
                                ice_curves[c_idx],
                                color="#93c5fd",
                                alpha=0.28,
                                linewidth=0.75,
                            )

                # Plot main PDP line on top
                ax.plot(x_vals, y_mean, color="#1d4ed8", linewidth=2.2, label="PDP (Mean)")

            except Exception as err:
                ax.text(
                    0.5,
                    0.5,
                    f"PDP Error: {err}",
                    ha="center",
                    va="center",
                    color="#dc2626",
                    fontsize=8,
                )

            ax.set_title(f"PDP: {feat_name}", fontsize=10, fontweight="bold")
            ax.set_xlabel(feat_name, fontsize=8.5)
            ylabel = "P(Target=1)" if target_type == "classification" else "Marginal Target"
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.tick_params(labelsize=8)

        # 4. Hide unused subplot panels
        for j in range(n_feats, n_rows * n_cols):
            ax_flat[j].set_visible(False)

        title_suffix = "(Mean Response & ICE Curves)" if kind == "both" else "(Marginal Response)"
        fig.suptitle(
            f"Partial Dependence of Prominent Features {title_suffix}",
            fontsize=11.5,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_pairwise_interaction_grid(
        self,
        df: pd.DataFrame,
        features: Any,
        target: str,
        is_target_numeric: bool = False,
        max_pairs: int = 6,
    ) -> str:
        """
        Renders a grid of Feature A vs Feature B scatter plots with points colored by the target.
        Safely flattens features if passed as a tuple/list of lists.
        """
        from itertools import combinations

        if df is None or target not in df.columns:
            return ""

        # Flatten nested collections (e.g., tuple of lists or 2D pair structures)
        flat_features: list[str] = []
        if isinstance(features, (list, tuple, set)):
            for item in features:
                if isinstance(item, (list, tuple, set)):
                    flat_features.extend([str(x) for x in item])
                elif isinstance(item, str):
                    flat_features.append(item)
        elif isinstance(features, str):
            flat_features = [features]

        # Deduplicate while preserving ordering
        clean_features = list(dict.fromkeys(flat_features))

        num_feats = [
            f
            for f in clean_features
            if f in df.columns
            and pd.api.types.is_numeric_dtype(df[f])
            and not pd.api.types.is_bool_dtype(df[f])
        ]

        pairs = list(combinations(num_feats, 2))[:max_pairs]
        if not pairs:
            return ""

        n_pairs = len(pairs)
        n_cols = min(3, n_pairs)
        n_rows = int(np.ceil(n_pairs / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), squeeze=False
        )
        ax_flat = axes.ravel()

        sub_df = df[num_feats + [target]].dropna()
        if len(sub_df) > 1500:
            sub_df = sub_df.sample(1500, random_state=42)

        for idx, (fa, fb) in enumerate(pairs):
            ax = ax_flat[idx]

            if is_target_numeric:
                scatter = ax.scatter(
                    sub_df[fa],
                    sub_df[fb],
                    c=sub_df[target],
                    cmap="viridis",
                    alpha=0.6,
                    s=20,
                    edgecolor="none",
                )
                cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label(target, fontsize=7.5)
            else:
                target_vals = sub_df[target].astype(str)
                palette = "tab10" if target_vals.nunique() > 2 else "coolwarm"
                sns.scatterplot(
                    data=sub_df,
                    x=fa,
                    y=fb,
                    hue=target,
                    palette=palette,
                    alpha=0.7,
                    s=22,
                    ax=ax,
                    legend="brief" if idx == 0 else False,
                )
                if idx == 0 and ax.get_legend():
                    ax.legend(title=target, fontsize=7.5, title_fontsize=8, loc="upper right")

            ax.set_title(f"{fa} vs. {fb}", fontsize=9.5, fontweight="bold")
            ax.set_xlabel(fa, fontsize=8.5)
            ax.set_ylabel(fb, fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=8)

        for j in range(n_pairs, n_rows * n_cols):
            ax_flat[j].set_visible(False)

        fig.suptitle(
            "Pairwise Bivariate Spaces",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_interaction_diagnostic_card(
        self,
        df: pd.DataFrame,
        feature_a: str,
        feature_b: str,
        operation: str,
        target: str,
        is_target_numeric: bool = False,
    ) -> str:
        """
        Renders a diagnostic card for an engineered interaction feature:
          - Panel 1 (Left): Distribution of the synthesized feature (Histogram/KDE + Mean/Median).
          - Panel 2 (Right): Synthesized feature vs. Target (Regression scatter or Categorical box/violin).
        """
        if feature_a not in df.columns or feature_b not in df.columns or target not in df.columns:
            return ""

        sub = df[[feature_a, feature_b, target]].dropna()
        if sub.empty:
            return ""

        if len(sub) > 1500:
            sub = sub.sample(1500, random_state=42)

        sa = sub[feature_a].astype(float)
        sb = sub[feature_b].astype(float)

        if "Multiplication" in operation:
            comb = sa * sb
            lbl = f"{feature_a} * {feature_b}"
        elif "Ratio (A/B)" in operation:
            comb = sa / (sb.replace(0, np.nan) + np.sign(sb) * 1e-4)
            lbl = f"{feature_a} / {feature_b}"
        elif "Ratio (B/A)" in operation:
            comb = sb / (sa.replace(0, np.nan) + np.sign(sa) * 1e-4)
            lbl = f"{feature_b} / {feature_a}"
        elif "Difference" in operation:
            comb = sa - sb
            lbl = f"{feature_a} - {feature_b}"
        else:
            comb = sa + sb
            lbl = f"{feature_a} + {feature_b}"

        sub["__engineered__"] = comb
        sub_clean = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["__engineered__"])
        if sub_clean.empty:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.2))

        # --- Panel 1: Synthesized Feature Distribution ---
        eng_s = sub_clean["__engineered__"]
        n_unique = eng_s.nunique()

        if n_unique <= 10:
            counts = eng_s.value_counts().sort_index()
            ax1.bar(
                [str(x) for x in counts.index],
                counts.values,
                color="#d97706",
                alpha=0.85,
            )
            ax1.set_title(f"Distribution: {lbl} (Discrete)", fontsize=9.5, fontweight="bold")
            ax1.set_ylabel("Count")
            if len(counts) > 5:
                ax1.tick_params(axis="x", rotation=30)
        else:
            # Filter extreme inf/outlier spikes for clean visual density
            q01, q99 = eng_s.quantile(0.01), eng_s.quantile(0.99)
            clipped_s = eng_s[(eng_s >= q01) & (eng_s <= q99)]
            if clipped_s.empty or clipped_s.std() == 0:
                clipped_s = eng_s

            bins = min(30, max(12, int(np.sqrt(len(clipped_s)))))
            sns.histplot(
                clipped_s,
                kde=True,
                ax=ax1,
                color="#d97706",
                bins=bins,
                stat="density",
                alpha=0.55,
            )
            mean_val = float(clipped_s.mean())
            med_val = float(clipped_s.median())
            ax1.axvline(
                mean_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                label=f"Mean: {mean_val:.2g}",
            )
            ax1.axvline(
                med_val,
                color="#16a34a",
                linestyle=":",
                linewidth=1.2,
                label=f"Median: {med_val:.2g}",
            )
            ax1.set_title(f"Distribution: {lbl}", fontsize=9.5, fontweight="bold")
            ax1.set_ylabel("Density")
            ax1.legend(fontsize=7.5, loc="upper right")

        ax1.set_xlabel(lbl, fontsize=8.5)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # --- Panel 2: Synthesized Feature vs Target ---
        if is_target_numeric:
            sns.regplot(
                data=sub_clean,
                x="__engineered__",
                y=target,
                ax=ax2,
                scatter_kws={"alpha": 0.35, "s": 15, "color": "#d97706"},
                line_kws={"color": "#1d4ed8", "linewidth": 1.5},
            )
            ax2.set_title(f"Signal: {lbl} vs. {target}", fontsize=9.5, fontweight="bold")
            ax2.set_ylabel(target, fontsize=8.5)
        else:
            target_cardinality = sub_clean[target].nunique()
            if target_cardinality <= 4 and len(sub_clean) >= 40:
                sns.violinplot(
                    data=sub_clean,
                    x=target,
                    y="__engineered__",
                    ax=ax2,
                    palette="Wistia",
                    inner="quartile",
                    cut=0,
                    hue=target,
                    legend=False,
                )
                ax2.set_title(
                    f"Distribution by {target}",
                    fontsize=9.5,
                    fontweight="bold",
                )
            else:
                sns.boxplot(
                    data=sub_clean,
                    x=target,
                    y="__engineered__",
                    ax=ax2,
                    palette="Wistia",
                    showmeans=True,
                    meanprops={
                        "marker": "o",
                        "markerfacecolor": "white",
                        "markeredgecolor": "#0f172a",
                    },
                )
                ax2.set_title(f"{lbl} by {target}", fontsize=9.5, fontweight="bold")
            if target_cardinality > 4:
                ax2.tick_params(axis="x", rotation=30)
            ax2.set_ylabel(lbl, fontsize=8.5)

        ax2.set_xlabel(lbl if is_target_numeric else target, fontsize=8.5)
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_cluster_projections(
        self,
        pca_coords: list[list[float]],
        cluster_labels: list[int],
        target_series: pd.Series | None = None,
        is_target_numeric: bool = False,
    ) -> str:
        """
        Renders clustering projections:
          - If Target exists (3 panels): PCA clusters, PCA target gradient, target distribution by cluster.
          - If Target is None (2 panels): PCA clusters and cluster instance counts & percentages.
        """
        if not pca_coords or not cluster_labels:
            return ""

        pca_arr = np.array(pca_coords)
        clusters_arr = np.array(cluster_labels)

        # Safely verify if target data is available and matches sample size
        has_target = (
            target_series is not None
            and len(target_series) == len(pca_arr)
            and not target_series.dropna().empty
        )

        # Downsample if dense to maintain clean vector rendering
        if len(pca_arr) > 1500:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pca_arr), size=1500, replace=False)
            pca_arr = pca_arr[idx]
            clusters_arr = clusters_arr[idx]
            if has_target:
                target_arr = target_series.to_numpy()[idx]
        else:
            if has_target:
                target_arr = target_series.to_numpy()

        if has_target:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 3.8))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))
            ax3 = None

        # --- Panel 1: PCA by Cluster ID (Always Rendered) ---
        unique_clusters = np.unique(clusters_arr)
        palette = sns.color_palette("tab10", len(unique_clusters))
        for i, c_id in enumerate(unique_clusters):
            mask = clusters_arr == c_id
            ax1.scatter(
                pca_arr[mask, 0],
                pca_arr[mask, 1],
                color=palette[i],
                label=f"Cluster {c_id}",
                alpha=0.65,
                s=20,
            )
            # Centroid notation
            c_center = pca_arr[mask].mean(axis=0)
            ax1.scatter(
                c_center[0],
                c_center[1],
                color="black",
                marker="X",
                s=70,
                edgecolor="white",
                linewidth=1,
            )

        ax1.set_title("Clusters in PCA Space", fontsize=10, fontweight="bold")
        ax1.set_xlabel("PCA Component 1", fontsize=8.5)
        ax1.set_ylabel("PCA Component 2", fontsize=8.5)
        ax1.legend(fontsize=7.5, loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.4)

        if has_target:
            # --- Panel 2: PCA by Target ---
            if is_target_numeric:
                sc = ax2.scatter(
                    pca_arr[:, 0],
                    pca_arr[:, 1],
                    c=target_arr.astype(float),
                    cmap="viridis",
                    alpha=0.65,
                    s=20,
                )
                cbar = fig.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02)
                cbar.ax.tick_params(labelsize=7.5)
            else:
                t_series = pd.Series(target_arr).astype(str)
                sns.scatterplot(
                    x=pca_arr[:, 0],
                    y=pca_arr[:, 1],
                    hue=t_series,
                    palette="tab10",
                    alpha=0.7,
                    s=22,
                    ax=ax2,
                )
                ax2.legend(fontsize=7.5, loc="upper right")

            ax2.set_title("Target Gradient in PCA Space", fontsize=10, fontweight="bold")
            ax2.set_xlabel("PCA Component 1", fontsize=8.5)
            ax2.set_ylabel("PCA Component 2", fontsize=8.5)
            ax2.grid(True, linestyle="--", alpha=0.4)

            # --- Panel 3: Target Distribution per Cluster ---
            cluster_labels_str = [f"C{c}" for c in clusters_arr]
            df_cluster_target = pd.DataFrame({"Cluster": cluster_labels_str, "Target": target_arr})

            if is_target_numeric:
                df_cluster_target["Target"] = df_cluster_target["Target"].astype(float)
                sns.boxplot(
                    data=df_cluster_target,
                    x="Cluster",
                    y="Target",
                    ax=ax3,
                    palette="Blues_r",
                    showmeans=True,
                    meanprops={
                        "marker": "o",
                        "markerfacecolor": "white",
                        "markeredgecolor": "#0f172a",
                    },
                    hue="Cluster",
                    legend=False,
                )
                global_mean = float(df_cluster_target["Target"].mean())
                ax3.axhline(
                    global_mean,
                    color="#dc2626",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Mean: {global_mean:.2g}",
                )
                ax3.set_title("Target Value by Cluster", fontsize=10, fontweight="bold")
                ax3.set_ylabel("Target", fontsize=8.5)
                ax3.legend(fontsize=7.5, loc="upper right")
            else:
                ct = (
                    pd.crosstab(
                        df_cluster_target["Cluster"], df_cluster_target["Target"], normalize="index"
                    )
                    * 100
                )
                ct.plot(kind="bar", stacked=True, ax=ax3, colormap="tab10", alpha=0.85)
                ax3.set_title("Class Proportion by Cluster (%)", fontsize=10, fontweight="bold")
                ax3.set_ylabel("Percentage", fontsize=8.5)
                ax3.tick_params(axis="x", rotation=0)
                ax3.legend(fontsize=7.5, bbox_to_anchor=(1.02, 1), loc="upper left")

            ax3.set_xlabel("Cluster ID", fontsize=8.5)
            ax3.grid(True, linestyle="--", alpha=0.4)

        else:
            # --- Panel 2 (Unsupervised / No Target): Cluster Sizes & Percentages ---
            unique, counts = np.unique(clusters_arr, return_counts=True)
            total_samples = len(clusters_arr)
            labels = [f"Cluster {u}" for u in unique]
            percentages = (counts / total_samples) * 100

            bars = ax2.bar(labels, counts, color=palette[: len(unique)], alpha=0.85)
            ax2.set_title("Cluster Instance Distribution", fontsize=10, fontweight="bold")
            ax2.set_xlabel("Cluster ID", fontsize=8.5)
            ax2.set_ylabel("Sample Count", fontsize=8.5)
            ax2.grid(True, linestyle="--", alpha=0.4)

            for bar, pct in zip(bars, percentages):
                yval = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + (max(counts) * 0.02),
                    f"{int(yval):,} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#1e293b",
                )
            ax2.set_ylim(0, max(counts) * 1.18)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_model_confusion_or_residuals(
        self,
        report: Any,
    ) -> str:
        """Renders Confusion Matrix for classification or Residual Plot for regression."""
        fig, ax = plt.subplots(figsize=(6.2, 3.8))

        if report.target_type == "classification":
            c = report.confusion_data
            cm_matrix = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
            sns.heatmap(
                cm_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                annot_kws={"size": 13, "weight": "bold"},
                xticklabels=["Pred Negative (0)", "Pred Positive (1)"],
                yticklabels=["Actual Negative (0)", "Actual Positive (1)"],
            )
            ax.set_title(
                f"Confusion Breakdown (Total Errors: {report.error_count:,})",
                fontsize=10.5,
                fontweight="bold",
                pad=10,
            )
        else:
            # Regression Residuals
            y_true = np.array([e.true_label for e in report.worst_errors])
            y_pred = np.array([e.predicted_label for e in report.worst_errors])
            res = y_true - y_pred

            ax.scatter(y_pred, res, color="#2563eb", alpha=0.65, s=24)
            ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.2)
            ax.set_title(
                "Top Residual Discrepancies ($y - \\hat{y}$)",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlabel("Predicted Value", fontsize=8.5)
            ax.set_ylabel("Residual", fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_shap_beeswarm(self, explanation: Any, max_display: int = 10) -> str:
        """Renders global SHAP Beeswarm summary plot."""
        if explanation is None:
            return ""

        try:
            import shap

            fig = plt.figure(figsize=(7.8, max(3.5, max_display * 0.35)))
            shap.plots.beeswarm(explanation, max_display=max_display, show=False)
            plt.title(
                "Global Feature Impact (SHAP Beeswarm)",
                fontsize=10.5,
                fontweight="bold",
                pad=12,
            )
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception:
            return ""

    def plot_shap_waterfall(self, explanation: Any, sample_idx: int, title: str) -> str:
        """Renders local SHAP Waterfall explaining an individual error instance."""
        if explanation is None or sample_idx >= len(explanation):
            return ""

        try:
            import shap

            fig = plt.figure(figsize=(7.5, 3.8))
            shap.plots.waterfall(explanation[sample_idx], max_display=8, show=False)
            plt.title(title, fontsize=10, fontweight="bold", pad=12)
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception:
            return ""
