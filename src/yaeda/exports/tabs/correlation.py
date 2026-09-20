from .base import HTMLTab
from yaeda.correlation import CorrelationReport


class CollinearTab(HTMLTab):
    def __init__(
        self,
        corr_report: CorrelationReport | None,
    ):
        super().__init__("collinear", "🔗 Collinearity")
        self._report = corr_report

    def has_report(self) -> str:
        return self._report is not None

    def _generate(self) -> str:
        if self._report.collinear_pairs:
            c_rows = [
                f"<tr><td><strong>{p.feature_a}</strong></td><td><strong>{p.feature_b}</strong></td><td><span class='badge badge-warning'>{p.pearson_corr:+.4f}</span></td><td>{p.spearman_corr:+.4f}</td><td class='rec-text'>Redundant pair. Pruned one during prominent feature selection.</td></tr>"
                for p in self._report.collinear_pairs
            ]
            collinear_table_html = f"""
            <table class="data-table">
                <thead>
                    <tr><th>Feature A</th><th>Feature B</th><th>Pearson r</th><th>Spearman &rho;</th><th>Action</th></tr>
                </thead>
                <tbody>{"".join(c_rows)}</tbody>
            </table>"""
        else:
            collinear_table_html = '<div class="alert alert-success">✅ No collinear feature pairs detected above threshold (|r| &ge; 0.80).</div>'

        return f"""<div class="card">
            <div class="card-header"><div class="card-title">Multicollinear Feature Pairs (|r| &ge; 0.80)</div></div>
            {collinear_table_html}
        </div>"""
