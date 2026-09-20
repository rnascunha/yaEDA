import datetime
from typing import Any
from pathlib import Path
import weasyprint


class EDAPDFReportBuilder:
    """Compiles a publication-grade, print-optimized PDF report with running headers,

    footers, and embedded high-resolution diagnostic charts.
    """

    def __init__(
        self,
        table_profile: Any,
        corr_report: Any,
        importance_report: Any,
        chart_generator: Any | None = None,
    ):
        self.tp = table_profile
        self.cr = corr_report
        self.ir = importance_report
        self.cg = chart_generator

    def _build_html_for_pdf(self) -> str:
        gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")  # noqa: DTZ005
        attr_label = getattr(self.ir, "attribution_method", "Attribution Analysis")

        # Generate charts
        b64_golden = self.cg.plot_golden_features(self.ir.importances) if self.cg else ""
        b64_assoc = self.cg.plot_target_associations(self.cr.target_associations) if self.cg else ""
        b64_heat = self.cg.plot_correlation_heatmap(self.cr.pearson_matrix) if self.cg else ""
        b64_health = (
            self.cg.plot_data_health(self.tp.features, self.tp.target_column) if self.cg else ""
        )

        golden_count = sum(1 for m in self.ir.importances if "Tier 1" in m.tier)
        strong_count = sum(1 for m in self.ir.importances if "Tier 2" in m.tier)

        # 1. Golden Leaderboard rows
        golden_rows = []
        for m in self.ir.importances:
            pct_score = min(max(int(m.golden_feature_score * 100), 2), 100)
            if "Tier 1" in m.tier:
                tier_badge = '<span class="badge badge-golden">Tier 1: Golden</span>'
                bar_color = "#d97706"
            elif "Tier 2" in m.tier:
                tier_badge = '<span class="badge badge-strong">Tier 2: Strong</span>'
                bar_color = "#2563eb"
            elif "Tier 3" in m.tier:
                tier_badge = '<span class="badge badge-moderate">Tier 3: Moderate</span>'
                bar_color = "#4b5563"
            else:
                tier_badge = '<span class="badge badge-low">Tier 4: Prune</span>'
                bar_color = "#9ca3af"

            row = f"""
            <tr>
                <td style="text-align:center; font-weight:bold;">#{m.rank}</td>
                <td><strong>{m.feature}</strong></td>
                <td>{tier_badge}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:4px;">
                        <div style="width:45px; background:#e2e8f0; height:6px; border-radius:3px; overflow:hidden;">
                            <div style="width:{pct_score}%; background:{bar_color}; height:100%;"></div>
                        </div>
                        <span style="font-family:monospace; font-size:7pt; font-weight:bold;">{m.golden_feature_score:.4f}</span>
                    </div>
                </td>
                <td><code>{m.permutation_mean:.4f}</code> <small style="color:#64748b;">(±{m.permutation_std:.4f})</small></td>
                <td><code>{m.attribution_score:.4f}</code></td>
                <td><code>{m.mutual_info:.4f}</code></td>
                <td style="font-size:7pt; color:#475569;">{m.recommendation}</td>
            </tr>
            """
            golden_rows.append(row)
        golden_table_body = "\n".join(golden_rows)

        # 2. Collinearity Section
        if self.cr.collinear_pairs:
            collinear_rows = []
            for p in self.cr.collinear_pairs:
                collinear_rows.append(f"""
                <tr>
                    <td><code>{p.feature_a}</code></td>
                    <td><code>{p.feature_b}</code></td>
                    <td style="font-weight:bold; color:#b45309;">{p.pearson_corr:+.4f}</td>
                    <td>{p.spearman_corr:+.4f}</td>
                    <td style="font-size:7pt;">High redundancy: Prune or aggregate one to avoid split dilution.</td>
                </tr>
                """)
            collinear_html = f"""
            <table>
                <thead>
                    <tr>
                        <th>Feature A</th>
                        <th>Feature B</th>
                        <th>Pearson r</th>
                        <th>Spearman rho</th>
                        <th>Recommended Action</th>
                    </tr>
                </thead>
                <tbody>{"".join(collinear_rows)}</tbody>
            </table>
            """
        else:
            collinear_html = '<div class="alert-box alert-success">✅ No collinear feature pairs detected above threshold (|r| &ge; 0.80).</div>'

        # 3. Data Quality Matrix
        quality_rows = []
        for feat, p in self.tp.features.items():
            if feat == self.tp.target_column:
                continue
            miss_str = (
                f'<span style="color:#dc2626; font-weight:bold;">{p.missing_percentage:.1f}%</span>'
                if p.missing_percentage > 5.0
                else f"{p.missing_percentage:.1f}%"
            )
            outlier_str = (
                f"{p.outliers.count} ({p.outliers.percentage:.1f}%)" if p.outliers else "-"
            )
            skew_str = f"{p.skewness:+.2f}" if p.skewness is not None else "-"
            iqr_str = f"{p.iqr:.2f}" if p.iqr is not None else "-"

            quality_rows.append(f"""
            <tr>
                <td><strong>{feat}</strong></td>
                <td><code>{p.dtype}</code></td>
                <td>{miss_str}</td>
                <td>{p.zero_percentage:.1f}%</td>
                <td>{p.distinct_count}</td>
                <td>{p.min_value if p.min_value is not None else "-"}</td>
                <td>{p.q25 if p.q25 is not None else "-"}</td>
                <td>{p.median if p.median is not None else "-"}</td>
                <td>{p.mean if p.mean is not None else "-"}</td>
                <td>{p.q75 if p.q75 is not None else "-"}</td>
                <td>{p.max_value if p.max_value is not None else "-"}</td>
                <td>{iqr_str}</td>
                <td>{outlier_str}</td>
                <td>{skew_str}</td>
            </tr>
            """)
        quality_table_body = "\n".join(quality_rows)

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4 portrait;
    margin: 18mm 12mm 18mm 12mm;
    @top-left {{
        content: "Automated EDA & Golden Features Report";
        font-size: 7.5pt;
        color: #64748b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    @top-right {{
        content: "Target: {self.tp.target_column} ({self.cr.target_type.upper()})";
        font-size: 7.5pt;
        color: #64748b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    @bottom-left {{
        content: "TabularAutoEDA Framework v0.1.0 • {gen_time}";
        font-size: 7.5pt;
        color: #94a3b8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-size: 7.5pt;
        color: #64748b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: bold;
    }}
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    font-size: 8pt;
    line-height: 1.35;
}}

.page-break {{ page-break-before: always; }}
.no-break {{ page-break-inside: avoid; }}

.banner {{
    background: #0f172a;
    color: white;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 14px;
}}
.banner h1 {{
    font-size: 16pt;
    margin: 0 0 3px 0;
    font-weight: 700;
    color: #ffffff;
}}
.banner p {{
    margin: 0;
    font-size: 8.5pt;
    color: #94a3b8;
}}
.tag {{
    display: inline-block;
    background: rgba(255, 255, 255, 0.15);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 7pt;
    font-weight: 500;
    color: #f1f5f9;
}}

.kpi-grid {{
    display: table;
    width: 100%;
    margin-bottom: 14px;
}}
.kpi-cell {{
    display: table-cell;
    width: 20%;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    text-align: center;
    vertical-align: middle;
}}
.kpi-cell + .kpi-cell {{ border-left: none; }}
.kpi-label {{
    font-size: 6.5pt;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748b;
}}
.kpi-val {{
    font-size: 13pt;
    font-weight: 800;
    margin: 2px 0;
}}
.kpi-desc {{ font-size: 6.5pt; color: #94a3b8; }}

h2 {{
    font-size: 10.5pt;
    color: #0f172a;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 6px;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 3px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7pt;
    margin-bottom: 10px;
}}
th {{
    background: #f1f5f9;
    color: #334155;
    font-weight: 700;
    padding: 4px 5px;
    border: 1px solid #cbd5e1;
    text-align: left;
    white-space: nowrap;
}}
td {{
    padding: 3.5px 5px;
    border: 1px solid #e2e8f0;
    vertical-align: middle;
}}
tr:nth-child(even) td {{ background: #f8fafc; }}

.badge {{
    display: inline-block;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 6.5pt;
    font-weight: 700;
    white-space: nowrap;
}}
.badge-golden {{ background: #fef3c7; color: #b45309; border: 0.5px solid #fde68a; }}
.badge-strong {{ background: #dbeafe; color: #1d4ed8; border: 0.5px solid #bfdbfe; }}
.badge-moderate {{ background: #f3f4f6; color: #374151; border: 0.5px solid #e5e7eb; }}
.badge-low {{ background: #fee2e2; color: #b91c1c; border: 0.5px solid #fecaca; }}

.alert-box {{
    padding: 7px 10px;
    border-radius: 5px;
    font-size: 7pt;
    margin-bottom: 8px;
}}
.alert-success {{
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}}

.chart-container {{
    text-align: center;
    margin-bottom: 8px;
    page-break-inside: avoid;
}}
.chart-container img {{
    width: 96%;
    max-height: 230px;
    object-fit: contain;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
}}
code {{
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 7pt;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
}}
</style>
</head>
<body>

<div class="banner">
    <h1>Exploratory Data Analysis & Golden Features Report</h1>
    <p>Target Column: <strong>{self.tp.target_column}</strong> &bull; Objective: <strong>{self.cr.target_type.upper()}</strong></p>
    <div style="margin-top: 6px;">
        <span class="tag">Engine: {self.ir.model_type}</span>
        <span class="tag">Attribution: {attr_label}</span>
        <span class="tag">Records: {self.tp.n_rows:,}</span>
        <span class="tag">Features: {self.tp.n_columns}</span>
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi-cell">
        <div class="kpi-label">Dataset Shape</div>
        <div class="kpi-val">{self.tp.n_rows:,} &times; {self.tp.n_columns}</div>
        <div class="kpi-desc">Rows &times; Columns</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Memory</div>
        <div class="kpi-val">{self.tp.memory_usage_mb:.3f} MB</div>
        <div class="kpi-desc">RAM footprint</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Golden Signals</div>
        <div class="kpi-val" style="color: #d97706;">{golden_count}</div>
        <div class="kpi-desc">Tier 1 priority</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Strong Signals</div>
        <div class="kpi-val" style="color: #2563eb;">{strong_count}</div>
        <div class="kpi-desc">Tier 2 candidates</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Collinearity</div>
        <div class="kpi-val" style="color: {"#d97706" if self.cr.collinear_pairs else "#10b981"};">{len(self.cr.collinear_pairs)}</div>
        <div class="kpi-desc">|r| &ge; 0.80 warnings</div>
    </div>
</div>

<h2>1. Golden Features Ranking Leaderboard</h2>
<table>
    <thead>
        <tr>
            <th>Rank</th>
            <th>Feature Name</th>
            <th>Tier</th>
            <th>Golden Score</th>
            <th>Permutation Drop</th>
            <th>Attribution</th>
            <th>Mutual Info</th>
            <th>Model Recommendation</th>
        </tr>
    </thead>
    <tbody>
        {golden_table_body}
    </tbody>
</table>

<div class="no-break">
    <h2>2. Multicollinearity & Redundancy Warnings</h2>
    {collinear_html}
</div>

<div class="page-break"></div>
<h2>3. Visual Diagnostics: Predictive Signal & Collinearity</h2>

<div class="chart-container">
    <img src="data:image/png;base64,{b64_golden}" alt="Golden Features Leaderboard Plot">
</div>

<div class="chart-container">
    <img src="data:image/png;base64,{b64_assoc}" alt="Target Associations Plot">
</div>

<div class="page-break"></div>
<h2>4. Visual Diagnostics: Inter-Feature Correlation & Health</h2>

<div class="chart-container">
    <img src="data:image/png;base64,{b64_heat}" alt="Correlation Heatmap Plot">
</div>

<div class="chart-container">
    <img src="data:image/png;base64,{b64_health}" alt="Data Health Distribution Plot">
</div>

<div class="page-break"></div>
<h2>5. Detailed Feature Statistics & Quality Health Matrix</h2>
<table>
    <thead>
        <tr>
            <th>Feature</th>
            <th>Dtype</th>
            <th>Missing</th>
            <th>Zeros</th>
            <th>Card.</th>
            <th>Min</th>
            <th>Q25</th>
            <th>Median</th>
            <th>Mean</th>
            <th>Q75</th>
            <th>Max</th>
            <th>IQR</th>
            <th>Outliers</th>
            <th>Skew</th>
        </tr>
    </thead>
    <tbody>
        {quality_table_body}
    </tbody>
</table>

</body>
</html>"""

    def generate(self, output_path: Path | str | None = None) -> weasyprint.HTML:
        html_str = self._build_html_for_pdf()
        doc = weasyprint.HTML(string=html_str)
        if output_path is not None:
            doc.write_pdf(output_path)
        return doc
