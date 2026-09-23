from .charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.model_diagnostics import ModelDiagnosticsReport


class ModelDiagnosticsTab(HTMLTab):
    def __init__(
        self,
        diagnostics_report: list[ModelDiagnosticsReport] | None,
        cg: EDAChartGenerator,
    ):
        super().__init__("model-diagnostics", "🎯 Model Error & SHAP")
        self._reports = diagnostics_report
        self.cg = cg

    def has_report(self) -> str:
        return self._reports and self._reports is not None

    def _build_diagnostics_tab_html(self) -> str:
        if not self._reports:
            return '<div class="alert">Model diagnostics were not provided. Pass <code>diagnostics=[...]</code> to activate this module.</div>'

        sections = []
        for rep in self._reports:
            b64_cm_res = self.cg.plot_model_confusion_or_residuals(rep)
            b64_beeswarm = self.cg.plot_shap_beeswarm(rep.shap_explanation)

            m = rep.metrics
            if rep.target_type == "classification":
                kpi_html = f"""
                <div class="kpi-card"><div class="kpi-title">Accuracy</div><div class="kpi-value">{m.get("accuracy", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">F1-Score</div><div class="kpi-value" style="color:var(--primary);">{m.get("f1", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Precision</div><div class="kpi-value">{m.get("precision", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Recall</div><div class="kpi-value">{m.get("recall", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Error Rate</div><div class="kpi-value" style="color:var(--danger);">{rep.error_rate}%</div></div>
                """
            else:
                kpi_html = f"""
                <div class="kpi-card"><div class="kpi-title">MAE</div><div class="kpi-value">{m.get("mae", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">RMSE</div><div class="kpi-value" style="color:var(--primary);">{m.get("rmse", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">R² Score</div><div class="kpi-value">{m.get("r2", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Error Samples</div><div class="kpi-value" style="color:var(--danger);">{rep.error_count:,}</div></div>
                """

            err_rows = []
            for e in rep.worst_errors[:10]:
                badge = (
                    '<span class="badge badge-danger">FP</span>'
                    if e.error_type == "FP"
                    else (
                        '<span class="badge badge-warning">FN</span>'
                        if e.error_type == "FN"
                        else f'<span class="badge badge-danger">{e.error_type}</span>'
                    )
                )
                feats_summary = ", ".join(
                    [f"<strong>{k}</strong>: {v}" for k, v in list(e.feature_values.items())[:3]]
                )
                err_rows.append(f"""
                <tr>
                    <td>#{e.index}</td>
                    <td>{badge}</td>
                    <td><code>{e.true_label}</code></td>
                    <td><code>{e.predicted_label}</code></td>
                    <td><code>{e.prediction_score if e.prediction_score is not None else "-"}</code></td>
                    <td style="font-size:11px;">{feats_summary}</td>
                </tr>
                """)

            sections.append(f"""
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header"><div class="card-title">Diagnostics: {rep.name}</div></div>
                <div class="kpi-grid" style="margin-bottom: 16px;">{kpi_html}</div>
                <div class="charts-grid" style="margin-bottom: 16px;">
                    <div class="chart-box"><img src="data:image/png;base64,{b64_cm_res}" alt="Confusion/Residuals"></div>
                    <div class="chart-box">{f'<img src="data:image/png;base64,{b64_beeswarm}" alt="SHAP Beeswarm">' if b64_beeswarm else '<div class="alert">SHAP Beeswarm unavailable</div>'}</div>
                </div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Sample</th><th>Error</th><th>Actual</th><th>Predicted</th><th>Score/Residual</th><th>Key Features</th></tr></thead>
                        <tbody>{"".join(err_rows)}</tbody>
                    </table>
                </div>
            </div>
            """)

        return "\n".join(sections)

    def _generate(self) -> str:
        diagnostics_tab_content = self._build_diagnostics_tab_html()
        return diagnostics_tab_content
