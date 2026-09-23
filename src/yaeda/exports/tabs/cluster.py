import pandas as pd

from yaeda.extract import TableProfile
from yaeda.correlation import CorrelationReport
from .charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.clustering import ClusterReport


class ClusterTab(HTMLTab):
    def __init__(
        self,
        cluster_reports: list[ClusterReport] | None,
        table_profile: TableProfile,
        corr_report: CorrelationReport,
        df: pd.DataFrame,
        cg: EDAChartGenerator,
        has_target: bool,
    ):
        super().__init__("cluster", "🧩 Cluster Analysis")
        self._reports = cluster_reports
        self.tp = table_profile
        self.cr = corr_report
        self.has_target = has_target
        self.df = df
        self.cg = cg

    def has_report(self) -> str:
        return self._reports is not None

    def _build_cluster_tab_html(self) -> str:
        if not self._reports:
            return '<div class="alert">Cluster analysis was not executed.</div>'

        sections = []
        is_target_num = self.has_target and (self.cr.target_type == "regression")
        target_s = (
            self.df[self.tp.target_column]
            if (
                self.df is not None
                and getattr(self, "has_target", False)
                and self.tp.target_column in self.df.columns
            )
            else None
        )

        for rep in self._reports:
            b64_plot = ""
            if self.df is not None and rep.pca_coordinates and rep.cluster_labels:
                b64_plot = self.cg.plot_cluster_projections(
                    pca_coords=rep.pca_coordinates,
                    cluster_labels=rep.cluster_labels,
                    target_series=target_s,
                    is_target_numeric=is_target_num,
                )

            rows = []
            for c in rep.clusters:
                feat_tags = " ".join(
                    [
                        f'<span class="badge {"badge-golden" if f.z_difference > 0 else "badge-strong"}">{f.feature} ({f.z_difference:+.2f}&sigma;)</span>'
                        for f in c.defining_features
                    ]
                )
                t_val = (
                    f"<code>{c.target_mean}</code>"
                    if (rep.target_type == "regression" and c.target_mean is not None)
                    else (
                        ", ".join([f"{k}: {v:.1f}%" for k, v in c.target_distribution.items()])
                        if c.target_distribution
                        else "N/A"
                    )
                )
                rows.append(f"""
                <tr>
                    <td style="text-align:center; font-weight:bold;">Cluster {c.cluster_id}</td>
                    <td>{c.size:,} ({c.percentage:.1f}%)</td>
                    <td>{t_val}</td>
                    <td>{feat_tags}</td>
                </tr>
                """)

            mi_str = (
                f"{rep.mutual_info_with_target:.4f}"
                if rep.mutual_info_with_target is not None
                else "N/A (No Target)"
            )

            sections.append(f"""
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header">
                    <div class="card-title">Partition: {rep.n_clusters} Clusters</div>
                </div>
                    <div class="kpi-grid" style="margin-bottom: 20px;">
                    <div class="kpi-card">
                        <div class="kpi-title">Clusters Count</div>
                        <div class="kpi-value" style="color: var(--primary);">{rep.n_clusters}</div>
                        <div class="kpi-sub">K-Means Partitions</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Silhouette Score</div>
                        <div class="kpi-value">{rep.silhouette_score:.4f}</div>
                        <div class="kpi-sub">Separation Quality (-1 to 1)</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Target Dependency</div>
                        <div class="kpi-value" style="color: var(--golden);">{mi_str}</div>
                        <div class="kpi-sub">Mutual Information I(Cluster; Y)</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Inertia</div>
                        <div class="kpi-value">{rep.inertia:,.1f}</div>
                        <div class="kpi-sub">Sum of Squared Distances</div>
                    </div>
                </div>
                {
                f'''
                <div style="text-align:center; background:#fafafa; border:1px solid var(--border); border-radius:8px; padding:12px; margin-bottom:14px;">
                    <img src="data:image/png;base64,{b64_plot}" style="width:100%; height:auto;" alt="Cluster Projections">
                </div>
                '''
                if b64_plot
                else ""
            }
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr><th>Cluster ID</th><th>Sample Size</th><th>Target Summary</th><th>Defining Attributes (&sigma; from global mean)</th></tr>
                        </thead>
                        <tbody>{"".join(rows)}</tbody>
                    </table>
                </div>
            </div>
            """)

        return "\n".join(sections)

    def _generate(self) -> str:
        cluster_tab_content = self._build_cluster_tab_html()
        return cluster_tab_content
