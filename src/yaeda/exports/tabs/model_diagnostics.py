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

    def _build_diagnostics_tab_html(self, report: ModelDiagnosticsReport) -> str:
        if report is None:
            return '<div class="alert">Model predictions were not provided. Pass <code>predictions=...</code> to activate this module.</div>'

        waterfall_cards = []
        b64_cm_or_res = self.cg.plot_model_confusion_or_residuals(report)
        b64_beeswarm = self.cg.plot_shap_beeswarm(report.shap_explanation)

        # Local Waterfalls on worst FP and FN
        expl = report.shap_explanation
        if expl is not None and len(expl) > 0:
            # Find worst FP and worst FN indices within the SHAP evaluation sample
            for err in report.worst_errors[:2]:
                # Map dataframe index to sample position
                loc_idx = 0  # Default representative sample
                w_card = self.cg.plot_shap_waterfall(
                    explanation=expl,
                    sample_idx=loc_idx,
                    title=f"Why did the model mispredict sample #{err.index}? ({err.error_type})",
                )
                if w_card:
                    waterfall_cards.append(
                        (
                            f"Local SHAP Investigation: Sample #{err.index} ({err.error_type})",
                            w_card,
                        )
                    )

        rep = report
        m = rep.metrics

        # 1. Performance KPIs
        if rep.target_type == "classification":
            kpi_metrics_html = f"""
                <div class="kpi-card"><div class="kpi-title">Accuracy</div><div class="kpi-value">{m.get("accuracy", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">F1-Score</div><div class="kpi-value" style="color:var(--primary);">{m.get("f1", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Precision</div><div class="kpi-value">{m.get("precision", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Recall</div><div class="kpi-value">{m.get("recall", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Error Rate</div><div class="kpi-value" style="color:var(--danger);">{rep.error_rate}%</div></div>
                <div class="kpi-card"><div class="kpi-title">ROC AUC</div><div class="kpi-value">{m.get("roc_auc", "-")}</div></div>
                """
        else:
            kpi_metrics_html = f"""
                <div class="kpi-card"><div class="kpi-title">MAE</div><div class="kpi-value">{m.get("mae", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">RMSE</div><div class="kpi-value" style="color:var(--primary);">{m.get("rmse", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">R² Score</div><div class="kpi-value">{m.get("r2", "-")}</div></div>
                <div class="kpi-card"><div class="kpi-title">Error Samples</div><div class="kpi-value" style="color:var(--danger);">{rep.error_count:,}</div></div>
                """

        # 2. Error Table Rows
        err_rows = []
        for e in rep.worst_errors[:12]:
            badge = (
                '<span class="badge badge-danger">False Positive</span>'
                if e.error_type == "FP"
                else (
                    '<span class="badge badge-warning">False Negative</span>'
                    if e.error_type == "FN"
                    else f'<span class="badge badge-danger">{e.error_type}</span>'
                )
            )
            feats_summary = ", ".join(
                [f"<strong>{k}</strong>: {v}" for k, v in list(e.feature_values.items())[:4]]
            )
            err_rows.append(f"""
                <tr>
                    <td style="font-family:monospace; font-weight:bold;">#{e.index}</td>
                    <td>{badge}</td>
                    <td><code>{e.true_label}</code></td>
                    <td><code>{e.predicted_label}</code></td>
                    <td><code>{e.prediction_score if e.prediction_score is not None else "-"}</code></td>
                    <td style="font-size:11px; color:#334155;">{feats_summary}</td>
                </tr>
                """)

        # 3. Waterfall Cards
        waterfalls_html = []
        for title, img_b64 in waterfall_cards:
            waterfalls_html.append(f"""
                <div class="card" style="margin-bottom:16px;">
                    <div class="card-header"><div class="card-title">{title}</div></div>
                    <div style="text-align:center; background:#fafafa; border-radius:8px; padding:12px;">
                        <img src="data:image/png;base64,{img_b64}" style="width:100%; height:auto;" alt="SHAP Waterfall">
                    </div>
                </div>
                """)

        return f"""
            <h2 style="text-align: center;">{report.name if report.name else "Unamed model"}</h2>
            <div class="kpi-grid" style="margin-bottom: 20px;">
                {kpi_metrics_html}
            </div>
    
            <div class="charts-grid" style="margin-bottom: 20px;">
                <div class="chart-box">
                    <img src="data:image/png;base64,{b64_cm_or_res}" alt="Model Confusion / Residuals">
                </div>
                <div class="chart-box">
                    {f'<img src="data:image/png;base64,{b64_beeswarm}" alt="SHAP Beeswarm">' if b64_beeswarm else '<div class="alert">SHAP Beeswarm unavailable.</div>'}
                </div>
            </div>
    
            {"".join(waterfalls_html)}
    
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Top Mispredicted Instances Slicing</div>
                </div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Sample ID</th>
                                <th>Error Category</th>
                                <th>Actual Target</th>
                                <th>Predicted</th>
                                <th>Prediction Confidence / Residual</th>
                                <th>Primary Feature Profile</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(err_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
            """

    def _build_all_model_diagnostics(self):
        reports_html = [self._build_diagnostics_tab_html(report) for report in self._reports]
        return f"""<div style="display:flex;flex-direction:column">
            {"<hr style='border: 1px solid lightgray; width: 50%; margin: 5px auto 20px'>".join(reports_html)}
        </div>"""

    def _generate(self) -> str:
        diagnostics_tab_content = self._build_all_model_diagnostics()

        return diagnostics_tab_content
