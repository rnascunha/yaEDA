from pathlib import Path
import pandas as pd

from yaeda.charts import EDAChartGenerator

from yaeda.extract import TableProfile, MultiTableProfile
from yaeda.feature_importance import FeatureImportanceReport
from yaeda.interaction import InteractionReport
from yaeda.correlation import CorrelationReport
from yaeda.clustering import ClusterReport
from yaeda.model_diagnostics import ModelDiagnosticsReport

from .tabs.charts import ChartsTab
from .tabs.cluster import ClusterTab
from .tabs.correlation import CollinearTab
from .tabs.health import HealthTab
from .tabs.features import FeaturesTab
from .tabs.comparsion import ComparsionTab
from .tabs.feature_importance import FeatureImporanceTab
from .tabs.interactions import InteractionsTab
from .tabs.model_diagnostics import ModelDiagnosticsTab
from .tabs.pdp import PDPTab


class EDAHTMLDashboardBuilder:
    """Builds an interactive HTML dashboard featuring golden rankings, feature deep dives,

    and partial dependence curves for prominent predictors.
    """

    def __init__(
        self,
        table_profile: TableProfile,
        corr_report: CorrelationReport,
        importance_report: FeatureImportanceReport,
        chart_generator: EDAChartGenerator | None = None,
        df: pd.DataFrame | None = None,
        prominent_features: list[str] | None = None,
        interaction_report: InteractionReport = None,
        cluster_report: list[ClusterReport] | None = None,
        diagnostics_report: ModelDiagnosticsReport | None = None,
        multi_profile: MultiTableProfile | None = None,
        secondary_dfs: dict[str, pd.DataFrame] | None = None,
        primary_name: str = "Primary",
        max_features_to_plot: int | None = None,
        enable_pdp: bool = True,
        enable_feature_importance: bool = True,
    ):
        self.df = df
        self.cg = chart_generator or EDAChartGenerator()

        self.tp = table_profile
        self.ir = importance_report
        self.cr = corr_report

        self.multi_profile = multi_profile
        self.secondary_dfs = secondary_dfs or {}
        self.primary_name = primary_name

        self.max_features_to_plot = max_features_to_plot
        self.enable_pdp = enable_pdp

        self.has_target = self.tp.target_column is not None
        self.has_secondary = bool(self.secondary_dfs) and (self.multi_profile is not None)

        # Normalize prominent_features to list[str] regardless of nesting
        raw_feats = prominent_features or []
        flat_feats = []
        for item in raw_feats:
            if isinstance(item, (list, tuple, set)):
                flat_feats.extend([str(x) for x in item])
            elif isinstance(item, str):
                flat_feats.append(item)
        self.prominent_features = list(dict.fromkeys(flat_feats))

        # Tabs
        self.health_tab = HealthTab(table_profile)
        self.features_tab = FeaturesTab(
            table_profile,
            corr_report,
            importance_report,
            df,
            self.cg,
            self.prominent_features,
            self.multi_profile,
            self.secondary_dfs,
            self.has_secondary,
            self.has_target,
            self.primary_name,
            self.max_features_to_plot,
        )
        self.comp_tab = ComparsionTab(
            table_profile, multi_profile, self.has_secondary, self.primary_name, self.cg
        )
        self.cr_tab = CollinearTab(corr_report)
        self.ir_tab = FeatureImporanceTab(
            importance_report, self.has_target and enable_feature_importance
        )
        self.cluster_tab = ClusterTab(
            cluster_report, table_profile, corr_report, df, self.cg, self.has_target
        )
        self.chart_tab = ChartsTab(table_profile, corr_report, importance_report, self.cg)
        self.pdp_tab = PDPTab(
            table_profile,
            corr_report,
            importance_report,
            self.cg,
            self.prominent_features,
            self.has_target,
            self.enable_pdp,
        )
        self.interaction_tab = InteractionsTab(
            interaction_report,
            df,
            features=self.prominent_features,
            cg=self.cg,
            has_target=self.has_target,
        )
        self.diagnostics_tab = ModelDiagnosticsTab(diagnostics_report, self.cg)

    def generate(
        self,
        output_path: Path | str | None = None,
        max_interaction_cards: int = 6,
        max_table_rows: int = 6,
    ) -> str:
        golden_count = sum(1 for m in self.ir.importances if "Tier 1" in m.tier)
        strong_count = sum(1 for m in self.ir.importances if "Tier 2" in m.tier)
        collinear_count = len(self.cr.collinear_pairs)

        interaction_tab_content = self.interaction_tab.generate(
            max_table_rows, max_interaction_cards
        )
        cluster_tab_content = self.cluster_tab.generate()
        diagnostics_tab_content = self.diagnostics_tab.generate()
        feature_importance_cotennt = self.ir_tab.generate()
        collinear_table_html = self.cr_tab.generate()
        chart_tab_content = self.chart_tab.generate()
        feature_cards_tab_content = self.features_tab.generate()
        pdp_tab_content = self.pdp_tab.generate()
        health_tab_content = self.health_tab.generate()
        comparsion_tab_content = self.comp_tab.generate()

        target_badge = (
            f"Target: <strong>{self.tp.target_column}</strong> ({self.cr.target_type.upper() if self.cr.target_type else 'SUPERVISED'})"
            if self.has_target
            else "Target: <em>None (Unsupervised / Test Exploration)</em>"
        )

        attr_label = getattr(self.ir, "attribution_method", "Attribution Analysis")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Dashboard • {self.primary_name}</title>
    <style>
        :root {{
            --bg-body: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --primary: #2563eb;
            --golden: #d97706;
            --golden-light: #fef3c7;
            --radius: 10px;
            --shadow: 0 4px 6px -1px rgba(0,0,0,0.06);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            border-radius: var(--radius);
            padding: 22px 28px;
            margin-bottom: 22px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .engine-tag {{
            background: rgba(255,255,255,0.12);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
            margin-bottom: 22px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 15px 18px;
        }}
        .kpi-title {{ font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--text-muted); }}
        .kpi-value {{ font-size: 23px; font-weight: 700; margin: 3px 0; }}
        .tabs-nav {{
            display: flex;
            gap: 6px;
            border-bottom: 2px solid var(--border);
            margin-bottom: 20px;
            overflow-x: auto;
        }}
        .tab-btn {{
            background: transparent;
            border: none;
            padding: 10px 18px;
            font-size: 13.5px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }}
        .tab-btn.active {{
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 22px;
            margin-bottom: 24px;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }}
        .card-title {{ font-size: 16px; font-weight: 700; }}
        .search-box {{
            padding: 7px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 13px;
            outline: none;
            width: 260px;
        }}
        .table-responsive {{ overflow-x: auto; }}
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; text-align: left; }}
        .data-table th {{ background: #f8fafc; padding: 10px 12px; border-bottom: 2px solid var(--border); }}
        .data-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
        .score-container {{ display: flex; align-items: center; gap: 8px; min-width: 120px; }}
        .score-bar-bg {{ flex-grow: 1; background: #e2e8f0; height: 7px; border-radius: 4px; overflow: hidden; }}
        .score-bar-fill {{ height: 100%; }}
        .score-val {{ font-family: monospace; font-size: 11px; font-weight: 700; }}
        .badge {{ display: inline-block; padding: 3px 7px; border-radius: 5px; font-size: 10.5px; font-weight: 600; }}
        .badge-golden {{ background: var(--golden-light); color: var(--golden); border: 1px solid #fde68a; }}
        .badge-strong {{ background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }}
        .badge-moderate {{ background: #f3f4f6; color: #374151; }}
        .badge-low {{ background: #fee2e2; color: #b91c1c; }}
        .badge-warning {{ background: #fef3c7; color: #b45309; }}
        .badge-danger {{ background: #fee2e2; color: #dc2626; }}
        .pill-type {{ background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 10px; }}
        .rec-text {{ font-size: 11.5px; color: #475569; }}
        .alert {{ padding: 12px 16px; border-radius: 6px; font-size: 13px; }}
        .alert-success {{ background: #ecfdf5; color: #065f46; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }}
        .chart-box {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; text-align: center; }}
        .chart-box img {{ width: 100%; height: auto; border-radius: 6px; }}

        /* Feature Card Grid Layout */
        .feature-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px 20px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: 370px 1fr;
            gap: 22px;
        }}
        @media (max-width: 1100px) {{
            .feature-card {{ grid-template-columns: 1fr; }}
        }}
        .feature-card-title {{ display: flex; flex-direction: column; gap: 6px; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 8px; }}
        .feat-name {{ font-size: 16px; font-weight: 700; word-break: break-all; }}
        .feat-badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
        .feature-stat-table {{ width: 100%; border-collapse: collapse; font-size: 11.5px; }}
        .feature-stat-table td {{ padding: 4px 6px; border-bottom: 1px solid #f1f5f9; }}
        .feature-stat-table td:first-child {{ color: var(--text-muted); width: 48%; }}
        .feature-stat-table td:last-child {{ font-weight: 600; text-align: right; }}
        .feature-card-right {{ display: flex; justify-content: center; align-items: center; background: #fafafa; border-radius: 8px; padding: 8px; }}
        .feature-chart-img {{ width: 100%; height: auto; max-height: 280px; object-fit: contain; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1 style="font-size:21px; font-weight:700;">yaEDA Dashboard</h1>
            <p style="color:#94a3b8; font-size:13px;">Dataset: <strong>{
            self.primary_name
        }</strong> &bull; {target_badge}</p>
        </div>
        <div>
            <span class="engine-tag">Engine: {self.ir.model_type}</span>
            <span class="engine-tag">Attribution: {attr_label}</span>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">{self.primary_name} Dimensions</div>
            <div class="kpi-value">{self.tp.n_rows:,} &times; {self.tp.n_columns}</div>
        </div>
        {
            f'''
        <div class="kpi-card">
            <div class="kpi-title">Secondary Datasets</div>
            <div class="kpi-value" style="color: var(--primary);">{len(self.secondary_dfs)}</div>
        </div>
        '''
            if self.has_secondary
            else ""
        }
        {
            f'''
        <div class="kpi-card">
            <div class="kpi-title">Golden Features</div>
            <div class="kpi-value" style="color: var(--golden);">{golden_count}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Strong Predictors</div>
            <div class="kpi-value" style="color: var(--primary);">{strong_count}</div>
        </div>
        '''
            if self.has_target
            else ""
        }
        <div class="kpi-card">
            <div class="kpi-title">Collinear Warnings</div>
            <div class="kpi-value" style="color: {"#d97706" if collinear_count else "#10b981"};">{
            collinear_count
        }</div>
        </div>
    </div>

    <div class="tabs-nav">
        {self.health_tab.head}
        {self.features_tab.head}
        {self.comp_tab.head}
        {self.ir_tab.head}
        {self.cr_tab.head}
        {self.interaction_tab.head}    
        {self.chart_tab.head}
        {self.cluster_tab.head}
        {self.pdp_tab.head}
        {self.diagnostics_tab.head}
        
    </div>
    {health_tab_content}
    {feature_cards_tab_content}
    {comparsion_tab_content}
    {feature_importance_cotennt}
    {collinear_table_html}
    {interaction_tab_content}
    {chart_tab_content}
    {cluster_tab_content}
    {pdp_tab_content}
    {diagnostics_tab_content}
</div>

<script>
function switchTab(tabId, evt) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    document.getElementById(`button-${{tabId}}`).classList.add("active");
}}
function filterRows(tabId, query) {{
    const q = query.toLowerCase();
    document.querySelectorAll('#' + tabId + ' .searchable-row').forEach(row => {{
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
}}
function filterCards(tabId, query) {{
    const q = query.toLowerCase();
    document.querySelectorAll('#' + tabId + ' .searchable-card').forEach(card => {{
        const name = card.getAttribute('data-feature').toLowerCase();
        card.style.display = name.includes(q) ? '' : 'none';
    }});
}}

document.addEventListener('DOMContentLoaded', (evt) => {{
    switchTab('tab-health', evt)
}});
</script>
</body>
</html>"""

        if output_path:
            p_out = Path(output_path)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            p_out.write_text(html, encoding="utf-8")
        return html
