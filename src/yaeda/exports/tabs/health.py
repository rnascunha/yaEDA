from .base import HTMLTab
from yaeda.extract import TableProfile


class HealthTab(HTMLTab):
    def __init__(
        self,
        table_profile: TableProfile | None,
    ):
        super().__init__("health", "📋 Health Profiling")
        self.tp = table_profile

    def has_report(self) -> str:
        return True

    def _generate(self) -> str:
        health_rows = []
        for feat, p in self.tp.features.items():
            if feat == self.tp.target_column:
                continue
            miss_badge = (
                f'<span class="badge badge-danger">{p.missing_percentage:.1f}%</span>'
                if p.missing_percentage > 5.0
                else f"{p.missing_percentage:.1f}%"
            )
            outlier_val = (
                f"{p.outliers.count} ({p.outliers.percentage:.1f}%)" if p.outliers else "-"
            )
            skew_val = f"{p.skewness:+.2f}" if p.skewness is not None else "-"
            kurtosis_val = f"{p.kurtosis:+.2f}" if p.kurtosis is not None else "-"
            iqr_val = f"{p.iqr:.2f}" if p.iqr is not None else "-"

            health_rows.append(f"""
            <tr class="searchable-row">
                <td><strong>{feat}</strong></td>
                <td><span class="pill-type">{p.dtype}</span></td>
                <td>{miss_badge}</td>
                <td>{p.zero_percentage:.1f}%</td>
                <td>{p.distinct_count}</td>
                <td>{p.min_value if p.min_value is not None else "-"}</td>
                <td>{p.q25 if p.q25 is not None else "-"}</td>
                <td>{p.median if p.median is not None else "-"}</td>
                <td>{p.mean if p.mean is not None else "-"}</td>
                <td>{p.q75 if p.q75 is not None else "-"}</td>
                <td>{p.max_value if p.max_value is not None else "-"}</td>
                <td>{iqr_val}</td>
                <td>{outlier_val}</td>
                <td>{skew_val}</td>
                <td>{kurtosis_val}</td>
            </tr>
            """)
        health_rows_html = "\n".join(health_rows)

        return f"""<div class="card">
                    <div class="card-header">
                        <div class="card-title">Detailed Feature Statistics & Health Profiling</div>
                        <input type="text" class="search-box" placeholder="Filter profiles..." onkeyup="filterRows('tab-health', this.value)">
                    </div>
                    <div class="table-responsive">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Feature</th><th>Dtype</th><th>Missing</th><th>Zeros</th><th>Distinct</th>
                                    <th>Min</th><th>Q25</th><th>Median</th><th>Mean</th><th>Q75</th><th>Max</th>
                                    <th>IQR</th><th>Outliers</th><th>Skewness</th><th>Kurtosis</th>
                                </tr>
                            </thead>
                            <tbody>{health_rows_html}</tbody>
                        </table>
                    </div>
                </div>"""
