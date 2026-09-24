from .charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.extract import TableProfile
from yaeda.feature_importance import FeatureImportanceReport
from yaeda.correlation import CorrelationReport


class PDPTab(HTMLTab):
    def __init__(
        self,
        table_profile: TableProfile,
        corr_report: CorrelationReport,
        importance_report: FeatureImportanceReport,
        cg: EDAChartGenerator,
        features: list[str],
        has_target: bool,
        enable: bool,
    ):
        super().__init__("pdp", "📈 Partial Dependence")
        self.tp = table_profile
        self.cr = corr_report
        self.ir = importance_report
        self.cg = cg
        self.features = features
        self.has_target = has_target
        self.enable = enable

    def has_report(self) -> bool:
        return self.enable and self.has_target

    def _build_pdp_tab_html(self, b64_pdp: str) -> str:
        rows = []
        for rank_idx, feat in enumerate(self.features, start=1):
            # If a list/tuple of features was passed (e.g., for 2D interactions or nested input)
            if isinstance(feat, (list, tuple)):
                feat_name = " & ".join(feat)
                # Grab profile from the first feature in the group as reference
                p = self.tp.features.get(feat[0])
                assoc = self.cr.target_associations.get(feat[0])
                imp = next((m for m in self.ir.importances if m.feature in feat), None)
            else:
                feat_name = feat
                p = self.tp.features.get(feat)
                assoc = self.cr.target_associations.get(feat)
                imp = next((m for m in self.ir.importances if m.feature == feat), None)

            g_score = f"{imp.golden_feature_score:.4f}" if imp else "-"
            perm_val = f"{imp.permutation_mean:.4f}" if imp else "-"
            mi_val = f"{assoc.mutual_info:.4f}" if assoc and assoc.mutual_info is not None else "-"
            corr_val = (
                f"{assoc.pearson_corr:+.4f}"
                if assoc and assoc.pearson_corr is not None
                else "[Non-linear / Cat]"
            )

            rows.append(f"""
                <tr>
                    <td style="text-align:center; font-weight:bold;">#{rank_idx}</td>
                    <td><strong>{feat_name}</strong></td>
                    <td><span class="pill-type">{p.dtype if p else "unknown"}</span></td>
                    <td><code>{g_score}</code></td>
                    <td><code>{perm_val}</code></td>
                    <td><code>{mi_val}</code></td>
                    <td><code>{corr_val}</code></td>
                    <td><span class="badge badge-golden">Prominent Predictor</span></td>
                </tr>
                """)

        table_body = "\n".join(rows)

        return f"""
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Top Selected Prominent Features (De-duplicated & Screened)</div>
                </div>
                <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">
                    Selected by synthesizing Golden Feature scores, Permutation Importance, Mutual Information, and Pearson correlation while eliminating collinear redundancy.
                </p>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Selection Rank</th>
                                <th>Feature</th>
                                <th>Type</th>
                                <th>Golden Score</th>
                                <th>Permutation Drop</th>
                                <th>Mutual Info</th>
                                <th>Pearson r</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_body}
                        </tbody>
                    </table>
                </div>
            </div>
    
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Partial Dependence (PDP) & Individual Conditional Expectation (ICE)</div>
                </div>
                <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 14px;">
                    The <strong>blue line</strong> represents the average marginal response (PDP). The <strong>light blue lines</strong> display ICE samples; parallel lines indicate additive behavior, while divergent slopes signal feature interactions.
                </p>
                <div style="text-align:center; background:#fafafa; border: 1px solid var(--border); border-radius:8px; padding:16px;">
                    <img src="data:image/png;base64,{b64_pdp}" alt="Partial Dependence Plots" style="max-width:100%; height:auto; border-radius:6px;">
                </div>
            </div>
            """

    def _generate(self) -> str:
        b64_pdp = self.cg.plot_partial_dependence(
            model=getattr(self.ir, "fitted_model", None),
            X=getattr(self.ir, "preprocessed_X", None),
            feature_names=getattr(self.ir, "feature_names", []),
            features_to_plot=self.features,
            target_type=self.ir.target_type,
        )

        pdp_tab_content = self._build_pdp_tab_html(b64_pdp)
        return pdp_tab_content
