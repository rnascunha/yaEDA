from typing import Any

from yaeda.charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.extract import TableProfile


class ComparsionTab(HTMLTab):
    def __init__(
        self,
        table_profile: TableProfile,
        multi_profile: Any | None,
        has_secondary: bool,
        primary_name: str,
        cg: EDAChartGenerator,
    ):
        super().__init__("comparison", "🔄 Dataset Comparison")
        self.tp = table_profile
        self.multi_profile = multi_profile
        self.primary_name = primary_name
        self.has_secondary = has_secondary
        self.cg = cg

    def has_report(self):
        return self.has_secondary

    def _build_comparison_tab_html(self) -> str:
        if not self.has_secondary or not self.multi_profile:
            return '<div class="alert">No secondary datasets provided for comparison.</div>'

        sections = []
        for sec_name, comp_list in self.multi_profile.comparisons.items():
            sec_profile = self.multi_profile.secondary_profiles.get(sec_name)
            sec_rows_count = f"{sec_profile.n_rows:,}" if sec_profile else "N/A"
            sec_cols_count = f"{sec_profile.n_columns}" if sec_profile else "N/A"

            # Compute drift KPIs
            max_delta_miss = 0.0
            unseen_total_count = 0
            mismatched_features = 0

            table_rows = []
            for c in comp_list:
                d_m = c.delta_missing_pct or 0.0
                if abs(d_m) > abs(max_delta_miss):
                    max_delta_miss = d_m
                if c.unseen_categories:
                    unseen_total_count += len(c.unseen_categories)
                if not (c.in_primary and c.in_secondary):
                    mismatched_features += 1

                # Presence badge
                if c.in_primary and c.in_secondary:
                    presence_badge = '<span class="badge badge-success">Both</span>'
                elif c.in_primary:
                    presence_badge = (
                        f'<span class="badge badge-strong">Only {self.primary_name}</span>'
                    )
                else:
                    presence_badge = f'<span class="badge badge-warning">Only {sec_name}</span>'

                # Delta missing
                if c.delta_missing_pct is not None:
                    if c.delta_missing_pct > 2.0:
                        delta_miss_str = f'<span style="color:#dc2626; font-weight:bold;">+{c.delta_missing_pct:.1f}%</span>'
                    elif c.delta_missing_pct < -2.0:
                        delta_miss_str = f'<span style="color:#16a34a; font-weight:bold;">{c.delta_missing_pct:.1f}%</span>'
                    else:
                        delta_miss_str = f"{c.delta_missing_pct:+.1f}%"
                else:
                    delta_miss_str = "-"

                p_miss = (
                    f"{c.primary_missing_pct:.1f}%" if c.primary_missing_pct is not None else "-"
                )
                s_miss = (
                    f"{c.secondary_missing_pct:.1f}%"
                    if c.secondary_missing_pct is not None
                    else "-"
                )

                # Delta mean
                if c.is_numeric and c.primary_mean is not None and c.secondary_mean is not None:
                    d_mean_str = f"{c.delta_mean:+.2f}" if c.delta_mean is not None else "-"
                    means_str = f"{c.primary_mean:.2f} / {c.secondary_mean:.2f} ({d_mean_str})"
                else:
                    means_str = "-"

                # Cardinality
                p_dist = str(c.primary_distinct) if c.primary_distinct is not None else "-"
                s_dist = str(c.secondary_distinct) if c.secondary_distinct is not None else "-"

                # Unseen categories
                if c.unseen_categories:
                    unseen_display = ", ".join(map(str, c.unseen_categories[:3]))
                    if len(c.unseen_categories) > 3:
                        unseen_display += f" (+{len(c.unseen_categories) - 3} more)"
                    unseen_str = f'<span class="badge badge-danger">⚠️ {len(c.unseen_categories)} unseen: [{unseen_display}]</span>'
                else:
                    unseen_str = '<span style="color:#64748b;">None</span>'

                table_rows.append(f"""
                <tr class="searchable-row">
                    <td><strong>{c.feature}</strong></td>
                    <td>{presence_badge}</td>
                    <td><span class="pill-type">{"numeric" if c.is_numeric else "categorical"}</span></td>
                    <td>{p_miss}</td>
                    <td>{s_miss}</td>
                    <td>{delta_miss_str}</td>
                    <td>{means_str}</td>
                    <td>{p_dist} / {s_dist}</td>
                    <td>{unseen_str}</td>
                </tr>
                """)

            # Drift overview chart
            b64_drift_plot = self.cg.plot_dataset_drift_overview(
                self.multi_profile.comparisons,
                secondary_name=sec_name,
                top_n=10,
            )

            sections.append(f"""
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header">
                    <div class="card-title">Comparison: {self.primary_name} vs. {sec_name}</div>
                </div>

                <div class="kpi-grid" style="margin-bottom: 16px;">
                    <div class="kpi-card">
                        <div class="kpi-title">{self.primary_name} Shape</div>
                        <div class="kpi-value">{self.tp.n_rows:,} &times; {self.tp.n_columns}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">{sec_name} Shape</div>
                        <div class="kpi-value">{sec_rows_count} &times; {sec_cols_count}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Max Missing Shift (Δ)</div>
                        <div class="kpi-value" style="color: {
                "#dc2626" if abs(max_delta_miss) > 5 else "var(--primary)"
            };">{max_delta_miss:+.1f}%</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Unseen Category Alerts</div>
                        <div class="kpi-value" style="color: {
                "#dc2626" if unseen_total_count else "var(--success)"
            };">{unseen_total_count}</div>
                    </div>
                </div>

                {
                f'''
                <div style="text-align:center; background:#fafafa; border:1px solid var(--border); border-radius:8px; padding:12px; margin-bottom:16px;">
                    <img src="data:image/png;base64,{b64_drift_plot}" style="width:100%; height:auto;" alt="Dataset Drift Plot">
                </div>
                '''
                if b64_drift_plot
                else ""
            }

                <div class="card-header" style="margin-top: 16px; margin-bottom: 8px;">
                    <div class="card-title" style="font-size:14px;">Detailed Feature Comparison & Drift Table</div>
                    <input type="text" class="search-box" placeholder="Filter features..." onkeyup="filterRows('tab-comparison', this.value)">
                </div>

                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Coverage</th>
                                <th>Type</th>
                                <th>{self.primary_name} Missing</th>
                                <th>{sec_name} Missing</th>
                                <th>Δ Missing %</th>
                                <th>Mean ({self.primary_name} / {sec_name})</th>
                                <th>Distinct ({self.primary_name} / {sec_name})</th>
                                <th>Novel Unseen Levels</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(table_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
            """)

        return "\n".join(sections)

    def _generate(self, *args, **kwargs):
        return self._build_comparison_tab_html()
