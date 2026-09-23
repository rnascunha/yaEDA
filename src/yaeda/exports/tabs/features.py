from typing import Any

import pandas as pd

from .charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.extract import TableProfile
from yaeda.feature_importance import FeatureImportanceReport
from yaeda.correlation import CorrelationReport


class FeaturesTab(HTMLTab):
    def __init__(
        self,
        table_profile: TableProfile,
        corr_report: CorrelationReport,
        importance_report: FeatureImportanceReport,
        df: pd.DataFrame,
        cg: EDAChartGenerator,
        features: list[str],
        multi_profile: Any | None,
        secondary_dfs: dict[str, pd.DataFrame] | None,
        has_secondary: bool,
        has_target: bool,
        primary_name: str,
    ):
        super().__init__("features", "🔍 Feature & Target Deep Dive")
        self.tp = table_profile
        self.cr = corr_report
        self.ir = importance_report
        self.df = df
        self.cg = cg
        self.features = features
        self.multi_profile = multi_profile
        self.has_secondary = has_secondary
        self.secondary_dfs = secondary_dfs
        self.has_target = has_target
        self.primary_name = primary_name

    def has_report(self) -> str:
        return True

    def _build_feature_cards_html(self) -> str:
        cards_html = []
        target_name = self.tp.target_column
        target_profile = self.tp.features.get(target_name) if target_name else None
        is_target_num = (
            target_profile.is_numeric
            if target_profile
            else (self.cr.target_type == "regression" if self.cr.target_type else False)
        )

        ordered_feats = []
        if target_name and target_name in self.tp.features:
            ordered_feats.append(target_name)

        importance_order = [
            m.feature
            for m in self.ir.importances
            if m.feature in self.tp.features and m.feature != target_name
        ]
        remaining = [f for f in self.tp.features if f != target_name and f not in importance_order]
        ordered_feats.extend(importance_order + remaining)

        imp_lookup = {m.feature: m for m in self.ir.importances}

        # Build feature comparison lookup if secondary datasets exist
        comp_lookup: dict[str, list[Any]] = {}
        if self.multi_profile:
            for sec_name, comp_list in self.multi_profile.comparisons.items():
                for c in comp_list:
                    comp_lookup.setdefault(c.feature, []).append((sec_name, c))

        for feat in ordered_feats:
            p = self.tp.features[feat]
            assoc = self.cr.target_associations.get(feat)
            imp_meta = imp_lookup.get(feat)
            is_target = feat == target_name
            is_prominent = feat in self.features

            # Badges
            badges = [f'<span class="pill-type">{p.dtype}</span>']
            if is_target:
                badges.append('<span class="badge badge-golden">🎯 Target Variable</span>')
            elif self.has_target:
                if is_prominent:
                    badges.append('<span class="badge badge-golden">🌟 Prominent Predictor</span>')
                if imp_meta:
                    if "Tier 1" in imp_meta.tier:
                        badges.append(
                            f'<span class="badge badge-golden">Rank #{imp_meta.rank}</span>'
                        )
                    elif "Tier 2" in imp_meta.tier:
                        badges.append(
                            f'<span class="badge badge-strong">Rank #{imp_meta.rank}</span>'
                        )
                    else:
                        badges.append(
                            f'<span class="badge badge-moderate">Rank #{imp_meta.rank}</span>'
                        )
            elif is_prominent:
                badges.append('<span class="badge badge-strong">Selected Feature</span>')

            # Statistics table rows
            stat_rows = [
                f"<tr><td>Total Count</td><td>{p.total_count:,}</td></tr>",
                f"<tr><td>{self.primary_name} Missing</td><td>{p.missing_count:,} ({p.missing_percentage:.1f}%)</td></tr>",
                f"<tr><td>Distinct Count</td><td>{p.distinct_count:,} ({p.distinct_percentage:.1f}%)</td></tr>",
            ]

            if p.is_numeric:
                stat_rows.extend(
                    [
                        f"<tr><td>Zeros</td><td>{p.zero_count:,} ({p.zero_percentage:.1f}%)</td></tr>",
                        f"<tr><td>Mean &plusmn; Std</td><td>{p.mean} &plusmn; {p.std_dev}</td></tr>",
                        f"<tr><td>Median (IQR)</td><td>{p.median} ({p.iqr})</td></tr>",
                        f"<tr><td>Min &ndash; Max</td><td>[{p.min_value}, {p.max_value}]</td></tr>",
                        f"<tr><td>Outliers</td><td>{p.outliers.count if p.outliers else 0} ({p.outliers.percentage if p.outliers else 0.0:.1f}%)</td></tr>",
                        f"<tr><td>Skewness</td><td>{p.skewness if p.skewness is not None else '-'}</td></tr>",
                    ]
                )
            else:
                top_cats_str = ", ".join(
                    [f"<code>{c['value']}</code> ({c['percentage']}%)" for c in p.most_frequent[:3]]
                )
                stat_rows.extend(
                    [
                        f"<tr><td>Dominant Mode</td><td><code>{p.mode}</code></td></tr>",
                        f"<tr><td>Top Categories</td><td style='font-size:11px;'>{top_cats_str}</td></tr>",
                    ]
                )

            # Multi-dataset comparison rows
            if feat in comp_lookup:
                for sec_name, comp_obj in comp_lookup[feat]:
                    if comp_obj.secondary_missing_pct is not None:
                        stat_rows.append(
                            f"<tr><td>{sec_name} Missing</td><td>{comp_obj.secondary_missing_pct:.1f}% ({comp_obj.delta_missing_pct:+.1f}%)</td></tr>"
                        )
                    if comp_obj.unseen_categories:
                        stat_rows.append(
                            f'<tr><td style="color:#dc2626; font-weight:bold;">Unseen in {sec_name}</td><td><span class="badge badge-danger">{len(comp_obj.unseen_categories)} levels</span></td></tr>'
                        )

            # Target associations (only if target is provided)
            if self.has_target and not is_target:
                if assoc and assoc.mutual_info is not None:
                    stat_rows.append(
                        f"<tr><td>Mutual Information</td><td><code>{assoc.mutual_info:.4f}</code></td></tr>"
                    )
                if assoc and assoc.pearson_corr is not None:
                    stat_rows.append(
                        f"<tr><td>Pearson Correlation</td><td><code>{assoc.pearson_corr:+.4f}</code></td></tr>"
                    )
                if imp_meta:
                    stat_rows.append(
                        f"<tr><td>Permutation Drop</td><td><code>{imp_meta.permutation_mean:.4f}</code></td></tr>"
                    )

            stats_table_html = (
                f"<table class='feature-stat-table'><tbody>{''.join(stat_rows)}</tbody></table>"
            )

            # Chart generation
            chart_img_html = ""
            if self.df is not None:
                chart_b64 = self.cg.plot_feature_summary_card(
                    df=self.df,
                    feature=feat,
                    target=target_name if self.has_target else None,
                    is_feat_numeric=p.is_numeric,
                    is_target_numeric=is_target_num,
                    secondary_dfs=self.secondary_dfs if self.has_secondary else None,
                    primary_name=self.primary_name,
                )
                if chart_b64:
                    chart_img_html = f'<img class="feature-chart-img" src="data:image/png;base64,{chart_b64}" alt="Distribution for {feat}">'

            card = f"""
            <div class="feature-card searchable-card" data-feature="{feat}">
                <div class="feature-card-left">
                    <div class="feature-card-title">
                        <span class="feat-name">{feat}</span>
                        <div class="feat-badges">{" ".join(badges)}</div>
                    </div>
                    {stats_table_html}
                </div>
                <div class="feature-card-right">
                    {chart_img_html}
                </div>
            </div>
            """
            cards_html.append(card)

        return "\n".join(cards_html)

    def _generate(self) -> str:
        feature_cards_tab_content = self._build_feature_cards_html()
        return f"""
        <div class="card" style="margin-bottom: 16px; padding: 14px 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <span style="font-size: 13.5px; font-weight: 600; color: var(--text-muted);">
                    Feature distribution and empirical target association cards
                </span>
                <input type="text" class="search-box" placeholder="Filter by feature name..." onkeyup="filterCards('tab-features', this.value)">
            </div>
        </div>
        {feature_cards_tab_content}"""
