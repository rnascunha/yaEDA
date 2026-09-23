import datetime
from pathlib import Path
from typing import Any


class EDAMarkdownReportBuilder:
    """Constructs a comprehensive, publication-ready GitHub-flavored Markdown report."""

    def __init__(
        self,
        table_profile: Any,
        corr_report: Any,
        importance_report: Any,
        multi_profile: Any | None = None,
    ):
        self.tp = table_profile
        self.cr = corr_report
        self.ir = importance_report
        self.mp = multi_profile

    def generate(self, output_path: Path | str | None = None) -> str:
        lines = []
        gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        target_label = self.tp.target_column or "None (Unsupervised / Test Exploration)"
        task_label = (
            self.cr.target_type.upper() if getattr(self.cr, "target_type", None) else "UNSUPERVISED"
        )
        dataset_name = getattr(self.tp, "dataset_name", "Primary")

        lines.append(f"# yaEDA Intelligence Report • {dataset_name}")
        lines.append(
            f"**Generated:** `{gen_time}` | **Target:** `{target_label}` | **Task:** `{task_label}`\n"
        )
        lines.append(
            f"- **Primary Dataset Shape:** {self.tp.n_rows:,} rows × {self.tp.n_columns} columns"
        )
        lines.append(f"- **Memory Usage:** {self.tp.memory_usage_mb:.3f} MB")
        attr_label = getattr(self.ir, "attribution_method", "Attribution Analysis")
        lines.append(f"- **Model Engine:** {self.ir.model_type} ({attr_label})\n")

        # Multi-dataset comparison section
        if self.mp and getattr(self.mp, "comparisons", None):
            lines.append("## Secondary Dataset Comparisons & Drift Analysis")
            for sec_name, comp_list in self.mp.comparisons.items():
                sec_prof = self.mp.secondary_profiles.get(sec_name)
                s_shape = (
                    f"{sec_prof.n_rows:,} rows × {sec_prof.n_columns} cols" if sec_prof else "N/A"
                )
                lines.append(f"### Comparison: {dataset_name} vs. `{sec_name}` ({s_shape})\n")
                lines.append(
                    "| Feature | Status | Type | Primary Miss % | Secondary Miss % | Δ Missing | Unseen Levels |"
                )
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
                for c in comp_list:
                    status = (
                        "Both"
                        if (c.in_primary and c.in_secondary)
                        else ("Only Primary" if c.in_primary else f"Only {sec_name}")
                    )
                    p_m = (
                        f"{c.primary_missing_pct:.1f}%"
                        if c.primary_missing_pct is not None
                        else "-"
                    )
                    s_m = (
                        f"{c.secondary_missing_pct:.1f}%"
                        if c.secondary_missing_pct is not None
                        else "-"
                    )
                    d_m = f"{c.delta_missing_pct:+.1f}%" if c.delta_missing_pct is not None else "-"
                    unseen = (
                        f"**{len(c.unseen_categories)} unseen** ({', '.join(map(str, c.unseen_categories[:2]))})"
                        if c.unseen_categories
                        else "None"
                    )
                    lines.append(
                        f"| `{c.feature}` | {status} | {'num' if c.is_numeric else 'cat'} | {p_m} | {s_m} | {d_m} | {unseen} |"
                    )
                lines.append("\n")

        # Golden features leaderboard (if supervised)
        if self.ir.importances:
            lines.append("## 1. Golden Features Ranking Leaderboard")
            lines.append(
                "| Rank | Tier | Feature | Golden Score | Permutation Drop | Attribution | Mutual Info | Recommendation |"
            )
            lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
            for m in self.ir.importances:
                badge = (
                    "🥇 Tier 1"
                    if "Tier 1" in m.tier
                    else (
                        "🥈 Tier 2"
                        if "Tier 2" in m.tier
                        else ("🥉 Tier 3" if "Tier 3" in m.tier else "⚠️ Low")
                    )
                )
                lines.append(
                    f"| {m.rank} | {badge} | **`{m.feature}`** | `{m.golden_feature_score:.4f}` | "
                    f"`{m.permutation_mean:.4f}` | `{m.attribution_score:.4f}` | `{m.mutual_info:.4f}` | {m.recommendation} |"
                )
            lines.append("\n")

        # Multicollinearity
        lines.append("## 2. Multicollinearity & Redundancy Analysis")
        if self.cr.collinear_pairs:
            lines.append(
                f"Detected **{len(self.cr.collinear_pairs)}** feature pairs exceeding threshold:\n"
            )
            lines.append(
                "| Feature A | Feature B | Pearson $r$ | Spearman $\\rho$ | Recommendation |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :--- |")
            for p in self.cr.collinear_pairs:
                lines.append(
                    f"| `{p.feature_a}` | `{p.feature_b}` | **{p.pearson_corr:+.4f}** | {p.spearman_corr:+.4f} | Redundant signal. |"
                )
        else:
            lines.append("✅ No critical collinear pairs detected above threshold.\n")

        content = "\n".join(lines)
        if output_path:
            p_out = Path(output_path)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            p_out.write_text(content, encoding="utf-8")
        return content
