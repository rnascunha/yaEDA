from yaeda.charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.extract import TableProfile
from yaeda.feature_importance import FeatureImportanceReport
from yaeda.correlation import CorrelationReport


class ChartsTab(HTMLTab):
    def __init__(
        self,
        table_profile: TableProfile,
        corr_report: CorrelationReport,
        importance_report: FeatureImportanceReport,
        cg: EDAChartGenerator,
    ):
        super().__init__("charts", "📊 Visual Diagnostics")
        self.tp = table_profile
        self.cr = corr_report
        self.ir = importance_report
        self.cg = cg

    def has_report(self) -> str:
        return True

    def _generate(self) -> str:
        b64_golden = self.cg.plot_golden_features(self.ir.importances)
        b64_assoc = self.cg.plot_target_associations(self.cr.target_associations)
        b64_heat = self.cg.plot_correlation_heatmap(self.cr.pearson_matrix)
        b64_health = self.cg.plot_data_health(self.tp.features, self.tp.target_column)

        return f"""<div class="charts-grid">
        <div class="chart-box"><img src="data:image/png;base64,{b64_golden}" alt="Golden Features"></div>
        <div class="chart-box"><img src="data:image/png;base64,{b64_assoc}" alt="Target Associations"></div>
        <div class="chart-box"><img src="data:image/png;base64,{b64_heat}" alt="Correlation Heatmap"></div>
        <div class="chart-box"><img src="data:image/png;base64,{b64_health}" alt="Data Health Distribution"></div>
    </div>"""
