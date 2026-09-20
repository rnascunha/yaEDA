from .charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.clustering import ClusterReport


class ClusterTab(HTMLTab):
    def __init__(
        self,
        cluster_report: list[ClusterReport] | None,
        df,
        cg: EDAChartGenerator,
    ):
        super().__init__("cluster", "🧩 Cluster Analysis")
        self._report = cluster_report
        self.df = df
        self.cg = cg

    def has_report(self) -> str:
        return self._report is not None

    def _build_cluster_tab_html(self, report: ClusterReport) -> str:
        b64_cluster_plot = self.cg.plot_cluster_projections(
            pca_coords=report.pca_coordinates,
            cluster_labels=report.cluster_labels,
            target_series=self.df[report.target],
            is_target_numeric=(report.target_type == "regression"),
        )

        if not report:
            return '<div class="alert">Cluster analysis was not executed.</div>'

        rows = []
        for c in report.clusters:
            feat_tags = " ".join(
                [
                    f'<span class="badge {"badge-golden" if f.z_difference > 0 else "badge-strong"}">{f.feature} ({f.z_difference:+.2f}&sigma;)</span>'
                    for f in c.defining_features
                ]
            )

            if report.target_type == "regression":
                t_val = f"<code>{c.target_mean}</code>"
            else:
                t_val = ", ".join([f"{k}: {v:.1f}%" for k, v in c.target_distribution.items()])

            rows.append(f"""
                <tr>
                    <td style="text-align:center; font-weight:bold;">Cluster {c.cluster_id}</td>
                    <td>{c.size:,} ({c.percentage:.1f}%)</td>
                    <td>{t_val}</td>
                    <td>{feat_tags}</td>
                </tr>
                """)

        table_body = "\n".join(rows)

        return f"""
            <div class="kpi-grid" style="margin-bottom: 20px;">
                <div class="kpi-card">
                    <div class="kpi-title">Clusters Count</div>
                    <div class="kpi-value" style="color: var(--primary);">{report.n_clusters}</div>
                    <div class="kpi-sub">K-Means Partitions</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Silhouette Score</div>
                    <div class="kpi-value">{report.silhouette_score:.4f}</div>
                    <div class="kpi-sub">Separation Quality (-1 to 1)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Target Dependency</div>
                    <div class="kpi-value" style="color: var(--golden);">{report.mutual_info_with_target:.4f}</div>
                    <div class="kpi-sub">Mutual Information I(Cluster; Y)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Inertia</div>
                    <div class="kpi-value">{report.inertia:,.1f}</div>
                    <div class="kpi-sub">Sum of Squared Distances</div>
                </div>
            </div>
    
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Cluster Geometry & Target Alignment</div>
                </div>
                <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 14px;">
                    Panel 1 & 2 display the first two principal components. Cross-reference Cluster boundaries with Target gradients to evaluate if cluster assignment segments target responses.
                </p>
                <div style="text-align:center; background:#fafafa; border: 1px solid var(--border); border-radius:8px; padding:12px;">
                    <img src="data:image/png;base64,{b64_cluster_plot}" style="width:100%; height:auto;" alt="Cluster Projections">
                </div>
            </div>
    
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Cluster Centroid Profiles & Distinguishing Attributes</div>
                </div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Cluster</th>
                                <th>Sample Size</th>
                                <th>Target Summary</th>
                                <th>Defining Feature Displacements (&sigma; from global mean)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_body}
                        </tbody>
                    </table>
                </div>
            </div>
            """

    def _build_all_clusters(self):
        clusters = [self._build_cluster_tab_html(report) for report in self._report]
        return f"""<div style="display:flex;flex-direction:column">
            {"<hr style='border: 1px solid lightgray; width: 50%; margin: 5px auto 20px'>".join(clusters)}
        </div>"""

    def _generate(self) -> str:
        # cluster_tab_content = self._build_cluster_tab_html(self._report)
        cluster_tab_content = self._build_all_clusters()
        return cluster_tab_content
