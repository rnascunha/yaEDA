from typing import Literal, Any
from pathlib import Path

import pandas as pd

# import weasyprint

from .extract import TabularDataProfiler, TableProfile
from .correlation import FeatureTargetAnalyzer, CorrelationReport
from .feature_importance import FeatureImportanceAnalyzer, FeatureImportanceReport
from .interaction import FeatureInteractionAnalyzer, InteractionReport

from .clustering import TabularClustersCall, ClusterReport
from .model_diagnostics import (
    ModelDiagnosticsCall,
    ModelDiagnosticsReport,
    ModelDiagnosticsInputs,
)

from .charts import EDAChartGenerator
from .exports.html import EDAHTMLDashboardBuilder
from .exports.markdown import EDAMarkdownReportBuilder

# from .exports.pdf import EDAPDFReportBuilder
from .exports.export import StructuredDataExporter


class TabularEDA:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str] | None = None,
        *,
        diagnostics: dict[str, Any] | None = None,
        target_type: Literal["classification", "regression"] | None = None,
        n_clusters: list[int] | None = None,
        n_frequent: int = 3,
        n_extremes: int = 3,
        outlier_irq_factor: float = 1.5,
        collinear_threshold: float = 0.80,
        test_size: float = 0.25,
        shap_sample_limit: int = 500,
        seed: int = 42,
    ):
        self._df = df
        self._target = target
        self._features = features

        self._diagnostics = (
            [ModelDiagnosticsInputs(**input) for input in diagnostics]
            if diagnostics is not None
            else None
        )

        self._n_frequent = n_frequent
        self._n_extremes = n_extremes
        self._outlier_irq_factor = outlier_irq_factor

        self._target_type = target_type
        self._seed = seed

        self._collinear_threshold = collinear_threshold

        self._test_size = test_size
        self._shap_sample_limit = shap_sample_limit

        self._n_clusters = n_clusters if n_clusters is not None else [4]

        self._profile: TableProfile | None = None
        self._corr_report: CorrelationReport | None = None
        self._feature_importance: FeatureImportanceReport | None = None
        self._cluster_report: list[ClusterReport] | None = None
        self._diagnostics_report: ModelDiagnosticsReport = None

    @property
    def stats(self) -> TableProfile:
        if self._profile is not None:
            return self._profile

        profiler = TabularDataProfiler(
            self._df,
            target=self._target,
            columns=self._features,
            n_frequent=self._n_frequent,
            n_extremes=self._n_extremes,
            outlier_iqr_factor=self._outlier_irq_factor,
        )
        self._profile = profiler.run()
        return self._profile

    @property
    def correlation(self) -> CorrelationReport:
        if self._corr_report is not None:
            return self._corr_report

        analyzer = FeatureTargetAnalyzer(
            df=self._df,
            target=self._target,
            features=self._features,
            collinear_threshold=self._collinear_threshold,
            random_state=self._seed,
        )

        self._corr_report = analyzer.run()
        return self._corr_report

    @property
    def feature_importance(self) -> FeatureImportanceReport:
        if self._feature_importance is not None:
            return self._feature_importance

        analyzer = FeatureImportanceAnalyzer(
            df=self._df,
            features=self._features,
            target=self._target,
            target_type=self._target_type,
            test_size=self._test_size,
            shap_sample_limit=self._shap_sample_limit,
            random_state=self._seed,
        )

        self._feature_importance = analyzer.run()
        return self._feature_importance

    @property
    def clusters(self) -> list[ClusterReport]:
        if self._cluster_report is not None:
            return list(self._cluster_report.values())

        prominent = self.select_prominent_features(top_n=8)
        analyzer = TabularClustersCall(
            df=self._df,
            target=self._target,
            features=prominent,
            target_type=self._target_type
            or (
                "classification"
                if self.correlation.target_type == "classification"
                else "regression"
            ),
            n_clusters=self._n_clusters,
            random_state=self._seed,
        )
        self._cluster_report = analyzer.run()
        return list(self._cluster_report.values())

    @property
    def diagnostics(self) -> list[ModelDiagnosticsReport] | None:
        if self._diagnostics_report is not None:
            return list(self._diagnostics_report.values())

        if self._diagnostics is None:
            return None

        fi = self.feature_importance
        analyzer = ModelDiagnosticsCall(
            df=self._df,
            target=self._target,
            inputs=self._diagnostics,
            features=fi.feature_names,
            target_type=self._target_type or fi.target_type,
            preprocessed_X=fi.preprocessed_X,
            random_state=self._seed,
        )
        self._diagnostics_report = analyzer.run()
        return list(self._diagnostics_report.values())

    def plot_partial_dependence(
        self,
        top_n: int = 6,
        kind: str = "both",
        deduplicate_collinear: bool = True,
    ) -> str:
        """Returns base64 Partial Dependence chart for the top N prominent features."""
        fi = self.feature_importance
        prominent = self.select_prominent_features(
            top_n=top_n, deduplicate_collinear=deduplicate_collinear
        )
        print(prominent, type(prominent))
        chart_engine = EDAChartGenerator()
        return chart_engine.plot_partial_dependence(
            model=fi.fitted_model,
            X=fi.preprocessed_X,
            feature_names=fi.feature_names,
            features_to_plot=prominent,
            target_type=fi.target_type,
            kind=kind,
        )

    def analyze_all(
        self,
    ) -> tuple[TableProfile, CorrelationReport, FeatureImportanceReport]:
        return self.stats, self.correlation, self.feature_importance

    def select_prominent_features(
        self,
        top_n: int = 6,
        include_auxiliary: bool = True,
    ) -> list[str]:
        """
        Guarantees selection of all Tier 1 (Golden) and Tier 2 (Strong) features,
        followed by Tier 3 (Auxiliary) features up to top_n.
        """
        stats, _corr, fi = self.analyze_all()

        unhealthy = {
            feat
            for feat, p in stats.features.items()
            if p.missing_percentage > 80.0 or (p.is_numeric and p.std_dev == 0)
        }

        golden_feats = [
            m.feature for m in fi.importances if "Tier 1" in m.tier and m.feature not in unhealthy
        ]
        strong_feats = [
            m.feature for m in fi.importances if "Tier 2" in m.tier and m.feature not in unhealthy
        ]
        aux_feats = [
            m.feature for m in fi.importances if "Tier 3" in m.tier and m.feature not in unhealthy
        ]
        other_feats = [
            m.feature
            for m in fi.importances
            if m.feature not in golden_feats
            and m.feature not in strong_feats
            and m.feature not in aux_feats
            and m.feature not in unhealthy
        ]

        # 1. Always prioritize Golden and Strong features
        selected = list(golden_feats)
        for f in strong_feats:
            if f not in selected:
                selected.append(f)

        # 2. Add Auxiliary features
        if include_auxiliary:
            for f in aux_feats:
                if f not in selected:
                    selected.append(f)
                if len(selected) >= max(top_n, len(golden_feats) + len(strong_feats) + 1):
                    break

        # 3. Fill up to top_n if capacity remains
        if len(selected) < top_n:
            for f in other_feats:
                if f not in selected:
                    selected.append(f)
                if len(selected) == top_n:
                    break

        return selected

    def interactions(self, top_n: int = 6, max_pairs: int | None = None) -> InteractionReport:
        prominent = self.select_prominent_features(top_n=top_n, include_auxiliary=True)
        analyzer = FeatureInteractionAnalyzer(
            df=self._df,
            target=self._target,
            features=prominent,
            target_type=self._target_type
            or (
                "classification"
                if self.correlation.target_type == "classification"
                else "regression"
            ),
            max_pairs=max_pairs,
            random_state=self._seed,
        )
        return analyzer.run()

    def to_markdown(self, output: Path | str | None = None) -> str:
        stats, corr, fi = self.analyze_all()
        md_builder = EDAMarkdownReportBuilder(
            table_profile=stats,
            corr_report=corr,
            importance_report=fi,
        )
        md_data = md_builder.generate(output)
        return md_data

    def to_html(
        self,
        output: Path | str | None = None,
        top_n_features: int = 6,
        top_n_interactions: int = 20,
    ) -> str:
        stats, corr, fi = self.analyze_all()
        prominent = self.select_prominent_features(top_n=top_n_features, include_auxiliary=True)
        interaction_rep = self.interactions(top_n=top_n_features, max_pairs=None)
        cluster_rep = self.clusters
        diag_rep = self.diagnostics

        chart_engine = EDAChartGenerator()
        html_builder = EDAHTMLDashboardBuilder(
            table_profile=stats,
            corr_report=corr,
            importance_report=fi,
            chart_generator=chart_engine,
            df=self._df,
            prominent_features=prominent,
            interaction_report=interaction_rep,
            cluster_report=cluster_rep,
            diagnostics_report=diag_rep,
        )
        return html_builder.generate(
            output_path=output,
            max_interaction_cards=top_n_interactions,
            max_table_rows=top_n_interactions,
        )

    def to_json(self, output: Path | str | None = None) -> dict[str, Any]:
        stats, corr, fi = self.analyze_all()
        cluster_rep = self.clusters
        diagnostics = self.diagnostics
        json_builder = StructuredDataExporter(
            table_profile=stats,
            corr_report=corr,
            importance_report=fi,
            cluster_report=cluster_rep,
            diagnostic_report=diagnostics,
        )
        return json_builder.export_json(output)

    # def to_pdf(self, output: Path | str | None = None) -> weasyprint.HTML:
    #     stats, corr, fi = self.analyze_all()
    #     chart_engine = EDAChartGenerator()
    #     pdf_builder = EDAPDFReportBuilder(
    #         table_profile=stats,
    #         corr_report=corr,
    #         importance_report=fi,
    #         chart_generator=chart_engine,
    #     )
    #     pdf_data = pdf_builder.generate(output)
    #     return pdf_data

    def to_csv(self, output: Path | str | None = None) -> dict[str, Any]:
        stats, corr, fi = self.analyze_all()
        csv_builder = StructuredDataExporter(
            table_profile=stats,
            corr_report=corr,
            importance_report=fi,
        )
        csv_data = csv_builder.export_csvs(output)
        return csv_data

    def to_notebook(self, top_n_features: int = 6, top_n_interactions: int = 6):
        from IPython.display import display, HTML

        html = self.to_html(
            top_n_features=top_n_features,
            top_n_interactions=top_n_interactions,
        )
        return display(HTML(html))
