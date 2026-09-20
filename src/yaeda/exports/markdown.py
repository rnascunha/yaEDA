import datetime
from pathlib import Path
from typing import Any


class EDAMarkdownReportBuilder:
    """Constructs a comprehensive, publication-ready GitHub-flavored Markdown report."""

    def __init__(self, table_profile: Any, corr_report: Any, importance_report: Any):
        self.tp = table_profile
        self.cr = corr_report
        self.ir = importance_report

    def generate(self, output_path: Path | str | None = None) -> str:
        lines = []
        gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005

        lines.append("# Exploratory Data Analysis & Golden Features Report")
        lines.append(
            f"**Generated:** `{gen_time}` | **Target:** `{self.tp.target_column}` | **Task:** `{self.cr.target_type.upper()}`\n"
        )
        lines.append(
            f"- **Dataset Shape:** {self.tp.n_rows:,} rows × {self.tp.n_columns} columns"
        )
        lines.append(f"- **Memory Usage:** {self.tp.memory_usage_mb:.3f} MB")
        attr_label = getattr(self.ir, "attribution_method", "Attribution Analysis")
        lines.append(f"- **Model Engine:** {self.ir.model_type} ({attr_label})\n")

        # Golden Features Leaderboard
        lines.append("## 1. Golden Features Ranking Leaderboard")
        lines.append(
            "Synthesized from attribution sensitivity, out-of-sample permutation drop, and Mutual Information.\n"
        )
        lines.append(
            "| Rank | Tier | Feature | Golden Score | Permutation Drop | Attribution | Mutual Info | ML Recommendation |"
        )
        lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
        for m in self.ir.importances:
            badge = (
                "🥇 Tier 1 (Golden)"
                if "Tier 1" in m.tier
                else (
                    "🥈 Tier 2 (Strong)"
                    if "Tier 2" in m.tier
                    else (
                        "🥉 Tier 3 (Moderate)" if "Tier 3" in m.tier else "⚠️ Low/Prune"
                    )
                )
            )
            lines.append(
                f"| {m.rank} | {badge} | **`{m.feature}`** | `{m.golden_feature_score:.4f}` | "
                f"`{m.permutation_mean:.4f}` | `{m.attribution_score:.4f}` | `{m.mutual_info:.4f}` | {m.recommendation} |"
            )

        # Multicollinearity
        lines.append("\n## 2. Multicollinearity & Redundancy Analysis")
        if self.cr.collinear_pairs:
            lines.append(
                f"Detected **{len(self.cr.collinear_pairs)}** feature pairs exceeding threshold ($|r| \\ge 0.80$):\n"
            )
            lines.append(
                "| Feature A | Feature B | Pearson $r$ | Spearman $\\rho$ | Preprocessing Recommendation |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :--- |")
            for p in self.cr.collinear_pairs:
                lines.append(
                    f"| `{p.feature_a}` | `{p.feature_b}` | **{p.pearson_corr:+.4f}** | {p.spearman_corr:+.4f} | Prune or aggregate one to avoid variance inflation & split dilution. |"
                )
        else:
            lines.append("✅ No critical collinear pairs detected above threshold.\n")

        # Health
        lines.append("\n## 3. Data Quality, Missing Values & Outliers")
        lines.append(
            "| Feature | Type | Missing (%) | Zeros (%) | Cardinality | Outliers (IQR) | Skewness | Health Status |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for feat, p in self.tp.features.items():
            if feat == self.tp.target_column:
                continue
            outlier_str = (
                f"{p.outliers.count} ({p.outliers.percentage:.1f}%)"
                if p.outliers
                else "N/A"
            )
            skew_str = f"{p.skewness:+.2f}" if p.skewness is not None else "N/A"

            flags = []
            if p.missing_percentage > 10.0:
                flags.append("Critical Missing")
            elif p.missing_percentage > 0:
                flags.append("Missing Present")
            if p.outliers and p.outliers.percentage > 5.0:
                flags.append("Heavy Outliers")
            if p.skewness and abs(p.skewness) > 1.5:
                flags.append("Highly Skewed")
            status = ", ".join(flags) if flags else "Healthy"

            lines.append(
                f"| `{feat}` | `{p.dtype}` | {p.missing_percentage:.2f}% | {p.zero_percentage:.2f}% | {p.distinct_count} | {outlier_str} | {skew_str} | {status} |"
            )

        content = "\n".join(lines)
        if output_path:
            p_out = Path(output_path)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            p_out.write_text(content, encoding="utf-8")
        return content
