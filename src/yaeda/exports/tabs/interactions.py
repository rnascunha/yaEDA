from typing import Any
import pandas as pd

from .charts import EDAChartGenerator

from .base import HTMLTab
from yaeda.interaction import InteractionReport


class InteractionsTab(HTMLTab):
    def __init__(
        self,
        interaction_report: InteractionReport | None,
        df: pd.DataFrame,
        features: list[str],
        cg: EDAChartGenerator,
    ):
        super().__init__("interactions", "⚡ Feature Interactions")
        self._report = interaction_report
        self.df = df
        self.cg = cg
        self.features = features

    def has_report(self) -> str:
        return self._report is not None

    def _build_interactions_tab_html(
        self,
        b64_interaction_grid: str,
        top_interactions: list[Any],
        diagnostic_cards_b64: list[tuple[str, str]],
        max_table_rows: int = 25,
    ) -> str:
        table_rows = []
        for rank, item in enumerate(top_interactions[:max_table_rows], start=1):
            synergy_badge = (
                f'<span class="badge badge-golden">+{item.synergy_gain:.4f} Synergy</span>'
                if item.synergy_gain > 0.05
                else (
                    f'<span class="badge badge-strong">+{item.synergy_gain:.4f} Gain</span>'
                    if item.synergy_gain > 0
                    else f'<span class="badge badge-low">{item.synergy_gain:.4f}</span>'
                )
            )

            table_rows.append(f"""
                <tr>
                    <td style="text-align:center; font-weight:bold;">#{rank}</td>
                    <td><code>{item.formula}</code></td>
                    <td><strong>{item.operation}</strong></td>
                    <td><code>{item.baseline_score:.4f}</code></td>
                    <td><strong>{item.combined_score:.4f}</strong></td>
                    <td>{synergy_badge}</td>
                    <td><span class="pill-type">{item.metric_name}</span></td>
                </tr>
                """)

        table_body = "\n".join(table_rows)

        diagnostics_html = []
        for title, img_b64 in diagnostic_cards_b64:
            diagnostics_html.append(f"""
                <div class="card" style="margin-bottom:16px;">
                    <div class="card-header"><div class="card-title">Synergy Case Study: {title}</div></div>
                    <div style="text-align:center; background:#fafafa; border-radius:8px; padding:10px;">
                        <img src="data:image/png;base64,{img_b64}" style="width:100%; height:auto;" alt="Interaction Diagnostic">
                    </div>
                </div>
                """)

        return f"""
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Pairwise Interaction Spaces (Bivariate Distribution with Target Coloring)</div>
                </div>
                <p style="font-size: 12.5px; color: var(--text-muted); margin-bottom: 14px;">
                    Visualizes joint feature geometries. Diagonal separation lines suggest addition/subtraction ($A \\pm B$), hyperbolic contours suggest ratios ($A / B$), and quadrant clustering signals multiplicative interactions ($A \times B$).
                </p>
                <div style="text-align:center; background:#fafafa; border:1px solid var(--border); border-radius:8px; padding:14px;">
                    <img src="data:image/png;base64,{b64_interaction_grid}" style="width:100%; height:auto;" alt="Pairwise Scatters">
                </div>
            </div>
    
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Arithmetic Feature Engineering Synergy Leaderboard</div>
                </div>
                <p style="font-size: 12.5px; color: var(--text-muted); margin-bottom: 12px;">
                    Ranks candidate operations where the combined feature exceeds the individual predictive performance of either component ($\\Delta > 0$).
                </p>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Candidate Formula</th>
                                <th>Operation</th>
                                <th>Best Single Baseline</th>
                                <th>Combined Score</th>
                                <th>Synergy Gain (Δ)</th>
                                <th>Evaluated Metric</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_body}
                        </tbody>
                    </table>
                </div>
            </div>
    
            {"".join(diagnostics_html)}
            """

    def _build_interact_tab(
        self,
        max_table_rows: int,
        max_interaction_cards: int,
    ) -> str:
        interaction_grid = ""
        diagnostic_cards = []
        top_interactions_list = []

        top_interactions_list = self._report.top_interactions
        is_target_num = self._report.target_type == "regression"

        # Render scatter grid
        pairs_count = len(self._report.best_per_pair)
        interaction_grid = self.cg.plot_pairwise_interaction_grid(
            df=self.df,
            features=self.features,
            target=self._report.target,
            is_target_numeric=is_target_num,
            max_pairs=pairs_count,
        )

        # Render cards for all available pairs up to max_interaction_cards
        pairs_to_visualize = self._report.best_per_pair[:max_interaction_cards]
        for item in pairs_to_visualize:
            b64_card = self.cg.plot_interaction_diagnostic_card(
                df=self.df,
                feature_a=item.feature_a,
                feature_b=item.feature_b,
                operation=item.operation,
                target=self._report.target,
                is_target_numeric=is_target_num,
            )
            if b64_card:
                sign = "+" if item.synergy_gain >= 0 else ""
                title = f"{item.feature_a} & {item.feature_b} → {item.formula} ({item.operation}) [Synergy: {sign}{item.synergy_gain:.4f}]"
                diagnostic_cards.append((title, b64_card))

        interactions_tab_content = self._build_interactions_tab_html(
            b64_interaction_grid=interaction_grid,
            top_interactions=top_interactions_list,
            diagnostic_cards_b64=diagnostic_cards,
            max_table_rows=max_table_rows,
        )

        return interactions_tab_content

    def _generate(self, max_table_rows: int, max_interaction_cards: int) -> str:
        interaction_tab_content = self._build_interact_tab(
            max_interaction_cards=max_interaction_cards,
            max_table_rows=max_table_rows,
        )
        return interaction_tab_content
