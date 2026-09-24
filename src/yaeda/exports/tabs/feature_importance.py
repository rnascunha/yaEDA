from .base import HTMLTab
from yaeda.feature_importance import FeatureImportanceReport


class FeatureImporanceTab(HTMLTab):
    def __init__(self, importance_report: FeatureImportanceReport | None, enabled: bool):
        super().__init__("golden", "🌟 Golden Features")
        self._report = importance_report
        self.enabled = enabled

    def has_report(self) -> str:
        return self.enabled and self._report is not None

    def _generate(self) -> str:
        golden_rows = []
        for m in self._report.importances:
            pct_score = min(max(int(m.golden_feature_score * 100), 3), 100)
            badge = (
                '<span class="badge badge-golden">🥇 Tier 1</span>'
                if "Tier 1" in m.tier
                else (
                    '<span class="badge badge-strong">🥈 Tier 2</span>'
                    if "Tier 2" in m.tier
                    else (
                        '<span class="badge badge-moderate">🥉 Tier 3</span>'
                        if "Tier 3" in m.tier
                        else '<span class="badge badge-low">⚠️ Low</span>'
                    )
                )
            )
            bar_color = (
                "#d97706"
                if "Tier 1" in m.tier
                else ("#2563eb" if "Tier 2" in m.tier else "#4b5563")
            )

            golden_rows.append(f"""
            <tr class="searchable-row">
                <td style="text-align:center; font-weight:bold;">#{m.rank}</td>
                <td><strong>{m.feature}</strong></td>
                <td>{badge}</td>
                <td>
                    <div class="score-container">
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width: {pct_score}%; background-color: {bar_color};"></div>
                        </div>
                        <span class="score-val">{m.golden_feature_score:.4f}</span>
                    </div>
                </td>
                <td><code>{m.permutation_mean:.4f}</code> <small class="text-muted">(±{m.permutation_std:.4f})</small></td>
                <td><code>{m.attribution_score:.4f}</code></td>
                <td><code>{m.mutual_info:.4f}</code></td>
                <td class="rec-text">{m.recommendation}</td>
            </tr>
            """)
        golden_rows_html = "\n".join(golden_rows)

        return f"""<div class="card">
                    <div class="card-header">
                        <div class="card-title">Golden Features Ranking Leaderboard</div>
                        <input type="text" class="search-box" placeholder="Filter features..." onkeyup="filterRows('tab-{self._id}', this.value)">
                    </div>
                    <div class="table-responsive">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Rank</th><th>Feature</th><th>Quality Tier</th><th>Golden Score</th>
                                    <th>Permutation Drop</th><th>Attribution Score</th><th>Mutual Info</th><th>Recommendation</th>
                                </tr>
                            </thead>
                            <tbody>{golden_rows_html}</tbody>
                        </table>
                    </div>
                </div>"""
