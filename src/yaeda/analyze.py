from pathlib import Path
from typing import Any, Literal
import pandas as pd

from .charts import EDAChartGenerator
from .clustering import ClusterReport, TabularClustersCall
from .correlation import CorrelationReport, FeatureTargetAnalyzer
from .exports.export import StructuredDataExporter
from .exports.html import EDAHTMLDashboardBuilder
from .exports.markdown import EDAMarkdownReportBuilder
from .extract import MultiDatasetProfiler, MultiTableProfile, TableProfile, TabularDataProfiler
from .feature_importance import FeatureImportanceAnalyzer, FeatureImportanceReport
from .interaction import FeatureInteractionAnalyzer, InteractionReport
from .model_diagnostics import (
    ModelDiagnosticsCall,
    ModelDiagnosticsInputs,
    ModelDiagnosticsReport,
)


class TabularEDA:
    def __init__(
        self,
        df: pd.DataFrame | tuple[pd.DataFrame, str],
        target: str | None = None,
        features: list[str] | None = None,
        *,
        secondary_dfs: list[pd.DataFrame | tuple[pd.DataFrame, str]]
        | dict[str, pd.DataFrame]
        | None = None,
        diagnostics: dict[str, Any] | list[dict[str, Any]] | None = None,
        target_type: Literal["classification", "regression"] | None = None,
        model_engine: Literal["auto", "lightgbm", "extra_trees", "random_forest"] = "auto",
        preset: Literal["minimal", "standard", "deep"] = "deep",
        enable_correlation: bool | None = None,
        enable_feature_importance: bool | None = None,
        enable_clustering: bool | None = None,
        enable_interactions: bool | None = None,
        enable_pdp: bool | None = None,
        max_features_to_plot: int | None = None,
        # --- Configurable Subsampling Limits ---
        fit_sample_limit: int | None = 25000,
        permutation_sample_limit: int | None = 10000,
        shap_sample_limit: int | None = 500,
        mi_sample_limit: int | None = 25000,
        interaction_sample_limit: int | None = 25000,
        clustering_sample_limit: int | None = 30000,
        # --------------------------------------
        n_clusters: list[int] | None = None,
        n_frequent: int = 3,
        n_extremes: int = 3,
        outlier_irq_factor: float = 1.5,
        collinear_threshold: float = 0.80,
        test_size: float = 0.25,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        # 1. Normalize Primary Dataset
        if isinstance(df, tuple):
            self._df, self._df_name = df[0], df[1]
        else:
            self._df, self._df_name = df, "Primary"

        # 2. Normalize Secondary Datasets
        self._secondary_dfs: dict[str, pd.DataFrame] = {}
        if secondary_dfs:
            if isinstance(secondary_dfs, dict):
                self._secondary_dfs = secondary_dfs
            elif isinstance(secondary_dfs, list):
                for idx, sec in enumerate(secondary_dfs):
                    if isinstance(sec, tuple):
                        self._secondary_dfs[sec[1]] = sec[0]
                    else:
                        self._secondary_dfs[f"Secondary_{idx + 1}"] = sec

        self._fit_sample_limit = fit_sample_limit
        self._permutation_sample_limit = permutation_sample_limit
        self._shap_sample_limit = shap_sample_limit
        self._mi_sample_limit = mi_sample_limit
        self._interaction_sample_limit = interaction_sample_limit
        self._clustering_sample_limit = clustering_sample_limit

        # 3. Target and Feature Space Validation
        self._target = target
        if self._target is not None and self._target not in self._df.columns:
            raise ValueError(
                f"Target '{self._target}' not found in primary dataset '{self._df_name}'."
            )

        self.model_engine = model_engine
        self.n_jobs = n_jobs
        self._features = features
        self._target_type = target_type
        self._seed = seed
        self._n_frequent = n_frequent
        self._n_extremes = n_extremes
        self._outlier_irq_factor = outlier_irq_factor
        self._collinear_threshold = collinear_threshold
        self._test_size = test_size
        self._shap_sample_limit = shap_sample_limit
        self._n_clusters = n_clusters if n_clusters is not None else [4]
        self._max_features_to_plot = max_features_to_plot

        # 4. Resolve Presets & Granular Execution Flags
        self.preset = preset
        preset_configs = {
            "minimal": {
                "correlation": False,
                "feature_importance": False,
                "clustering": False,
                "interactions": False,
                "pdp": False,
            },
            "standard": {
                "correlation": True,
                "feature_importance": True,
                "clustering": True,
                "interactions": False,
                "pdp": False,
            },
            "deep": {
                "correlation": True,
                "feature_importance": True,
                "clustering": True,
                "interactions": True,
                "pdp": True,
            },
        }

        cfg = preset_configs.get(preset, preset_configs["deep"])
        self.enable_correlation = (
            enable_correlation if enable_correlation is not None else cfg["correlation"]
        )
        self.enable_feature_importance = (
            enable_feature_importance
            if enable_feature_importance is not None
            else cfg["feature_importance"]
        )
        self.enable_clustering = (
            enable_clustering if enable_clustering is not None else cfg["clustering"]
        )
        self.enable_interactions = (
            enable_interactions if enable_interactions is not None else cfg["interactions"]
        )
        self.enable_pdp = enable_pdp if enable_pdp is not None else cfg["pdp"]

        # Diagnostics input
        self._diagnostics = (
            [ModelDiagnosticsInputs(**item) for item in diagnostics]
            if diagnostics is not None
            else None
        )

        # Lazy caches
        self._profile: TableProfile | None = None
        self._multi_profile: MultiTableProfile | None = None
        self._corr_report: CorrelationReport | None = None
        self._feature_importance: FeatureImportanceReport | None = None
        self._cluster_report: dict[int, ClusterReport] | None = None
        self._diagnostics_report: dict[str, ModelDiagnosticsReport] | None = None

    @property
    def has_target(self) -> bool:
        return self._target is not None and self._target in self._df.columns

    @property
    def has_secondary(self) -> bool:
        return len(self._secondary_dfs) > 0

    @property
    def stats(self) -> TableProfile:
        if self._profile is not None:
            return self._profile

        profiler = TabularDataProfiler(
            self._df,
            target=self._target,
            columns=self._features,
            dataset_name=self._df_name,
            n_frequent=self._n_frequent,
            n_extremes=self._n_extremes,
            outlier_iqr_factor=self._outlier_irq_factor,
        )
        self._profile = profiler.run()
        return self._profile

    @property
    def multi_stats(self) -> MultiTableProfile:
        if self._multi_profile is not None:
            return self._multi_profile

        profiler = MultiDatasetProfiler(
            primary_df=self._df,
            secondary_dfs=self._secondary_dfs,
            target=self._target,
            primary_name=self._df_name,
            features=self._features,
            n_jobs=self.n_jobs,
        )
        self._multi_profile = profiler.run()
        return self._multi_profile

    @property
    def correlation(self) -> CorrelationReport:
        if self._corr_report is not None:
            return self._corr_report

        if not self.enable_correlation:
            self._corr_report = CorrelationReport(
                target=self._target,
                target_type=self._target_type,
                target_associations={},
                collinear_pairs=[],
                pearson_matrix={},
                spearman_matrix={},
            )
            return self._corr_report

        analyzer = FeatureTargetAnalyzer(
            df=self._df,
            target=self._target,
            features=self._features,
            collinear_threshold=self._collinear_threshold,
            mi_sample_limit=self._mi_sample_limit,
            n_jobs=self.n_jobs,
            random_state=self._seed,
        )
        self._corr_report = analyzer.run()
        return self._corr_report

    @property
    def feature_importance(self) -> FeatureImportanceReport:
        if self._feature_importance is not None:
            return self._feature_importance

        if not self.enable_feature_importance or not self.has_target:
            self._feature_importance = FeatureImportanceReport(
                target=self._target,
                target_type=self._target_type,
                model_type="None",
                attribution_method="None",
                importances=[],
                fitted_model=None,
                preprocessed_X=None,
                feature_names=[
                    c for c in (self._features or self._df.columns) if c != self._target
                ],
            )
            return self._feature_importance

        analyzer = FeatureImportanceAnalyzer(
            df=self._df,
            features=self._features,
            target=self._target,
            target_type=self._target_type,
            model_engine=self.model_engine,
            test_size=self._test_size,
            fit_sample_limit=self._fit_sample_limit,
            permutation_sample_limit=self._permutation_sample_limit,
            mi_sample_limit=self._mi_sample_limit,
            shap_sample_limit=self._shap_sample_limit,
            random_state=self._seed,
        )
        self._feature_importance = analyzer.run()
        return self._feature_importance

    @property
    def clusters(self) -> list[ClusterReport]:
        if self._cluster_report is not None:
            return list(self._cluster_report.values())

        if not self.enable_clustering:
            return []

        prominent = self.select_prominent_features(top_n=8)
        analyzer = TabularClustersCall(
            df=self._df,
            target=self._target if self.has_target else None,
            features=prominent,
            target_type=self._target_type
            or (
                "classification"
                if self.has_target and self.correlation.target_type == "classification"
                else ("regression" if self.has_target else None)
            ),
            n_clusters=self._n_clusters,
            sample_limit=self._clustering_sample_limit,
            n_jobs=-1,
            random_state=self._seed,
        )
        self._cluster_report = analyzer.run()
        return list(self._cluster_report.values())

    @property
    def diagnostics(self) -> list[ModelDiagnosticsReport] | None:
        if self._diagnostics_report is not None:
            return list(self._diagnostics_report.values())

        if self._diagnostics is None or not self.has_target:
            return None

        fi = self.feature_importance
        analyzer = ModelDiagnosticsCall(
            df=self._df,
            target=self._target,
            inputs=self._diagnostics,
            features=fi.feature_names,
            target_type=self._target_type or fi.target_type,
            preprocessed_X=fi.preprocessed_X,
            n_jobs=self.n_jobs,
            random_state=self._seed,
        )
        self._diagnostics_report = analyzer.run()
        return list(self._diagnostics_report.values())

    def select_prominent_features(
        self,
        top_n: int = 6,
        include_auxiliary: bool = True,
    ) -> list[str]:
        stats = self.stats
        unhealthy = {
            feat
            for feat, p in stats.features.items()
            if p.missing_percentage > 80.0 or (p.is_numeric and p.std_dev == 0)
        }

        # 1. Supervised selection if feature importance is enabled and computed
        if (
            self.enable_feature_importance
            and self.has_target
            and self.feature_importance.importances
        ):
            fi = self.feature_importance
            golden_feats = [
                m.feature
                for m in fi.importances
                if "Tier 1" in m.tier and m.feature not in unhealthy
            ]
            strong_feats = [
                m.feature
                for m in fi.importances
                if "Tier 2" in m.tier and m.feature not in unhealthy
            ]
            aux_feats = [
                m.feature
                for m in fi.importances
                if "Tier 3" in m.tier and m.feature not in unhealthy
            ]
            other_feats = [
                m.feature
                for m in fi.importances
                if m.feature not in golden_feats
                and m.feature not in strong_feats
                and m.feature not in aux_feats
                and m.feature not in unhealthy
            ]

            selected = list(golden_feats)
            for f in strong_feats:
                if f not in selected:
                    selected.append(f)

            if include_auxiliary:
                for f in aux_feats:
                    if f not in selected:
                        selected.append(f)
                    if len(selected) >= max(top_n, len(golden_feats) + len(strong_feats) + 1):
                        break

            if len(selected) < top_n:
                for f in other_feats:
                    if f not in selected:
                        selected.append(f)
                    if len(selected) == top_n:
                        break

            return selected[:top_n]

        # 2. Unsupervised fallback: prioritize spread and data completeness
        candidates = [f for f in stats.features if f != self._target and f not in unhealthy]

        def feature_spread_score(col: str) -> float:
            p = stats.features[col]
            valid_factor = (100.0 - p.missing_percentage) / 100.0
            if p.is_numeric and p.std_dev is not None:
                return (p.std_dev or 1.0) * valid_factor
            return float(p.distinct_count) * valid_factor

        candidates.sort(key=feature_spread_score, reverse=True)
        return candidates[:top_n]

    def interactions(self, top_n: int = 6, max_pairs: int | None = None) -> InteractionReport:
        if not self.enable_interactions or not self.has_target:
            return InteractionReport(
                target=self._target,
                target_type=self._target_type,
                top_interactions=[],
                best_per_pair=[],
            )

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
            sample_limit=self._interaction_sample_limit,
            random_state=self._seed,
        )
        return analyzer.run()

    def plot_partial_dependence(
        self,
        top_n: int = 6,
        kind: str = "both",
        deduplicate_collinear: bool = True,
    ) -> str:
        if not self.enable_pdp or not self.has_target:
            return ""

        fi = self.feature_importance
        if fi.fitted_model is None or fi.preprocessed_X is None:
            return ""

        prominent = self.select_prominent_features(top_n=top_n)
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

    def to_markdown(self, output: Path | str | None = None) -> str:
        stats, corr, fi = self.analyze_all()
        multi_prof = self.multi_stats if self.has_secondary else None
        md_builder = EDAMarkdownReportBuilder(
            table_profile=stats,
            corr_report=corr,
            importance_report=fi,
            multi_profile=multi_prof,
        )
        return md_builder.generate(output)

    def to_html(
        self,
        output: Path | str | None = None,
        top_n_features: int = 6,
        top_n_interactions: int = 20,
        max_features_to_plot: int | None = None,
    ) -> str:
        stats = self.stats
        corr = self.correlation
        fi = self.feature_importance
        prominent = self.select_prominent_features(top_n=top_n_features, include_auxiliary=True)
        interaction_rep = (
            self.interactions(top_n=top_n_features, max_pairs=None)
            if self.enable_interactions
            else None
        )
        cluster_rep = self.clusters if self.enable_clustering else []
        diag_rep = self.diagnostics
        multi_prof = self.multi_stats if self.has_secondary else None

        card_limit = (
            max_features_to_plot if max_features_to_plot is not None else self._max_features_to_plot
        )

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
            multi_profile=multi_prof,
            secondary_dfs=self._secondary_dfs if self.has_secondary else None,
            primary_name=self._df_name,
            max_features_to_plot=card_limit,
            enable_pdp=self.enable_pdp,
        )
        return html_builder.generate(
            output_path=output,
            max_interaction_cards=top_n_interactions,
            max_table_rows=top_n_interactions,
        )

    def to_json(self, output: Path | str | None = None) -> dict[str, Any]:
        stats, corr, fi = self.analyze_all()
        cluster_rep = self.clusters if self.enable_clustering else []
        diagnostics = self.diagnostics
        multi_prof = self.multi_stats if self.has_secondary else None

        json_builder = StructuredDataExporter(
            table_profile=stats,
            corr_report=corr,
            importance_report=fi,
            cluster_report=cluster_rep,
            diagnostic_report=diagnostics,
            multi_profile=multi_prof,
        )
        return json_builder.export_json(output)

    def to_csv(self, output: Path | str | None = None) -> dict[str, Any]:
        stats, corr, fi = self.analyze_all()
        csv_builder = StructuredDataExporter(
            table_profile=stats,
            corr_report=corr,
            importance_report=fi,
            cluster_report=self.clusters if self.enable_clustering else [],
        )
        return csv_builder.export_csvs(output)

    def to_notebook(self, top_n_features: int = 6, top_n_interactions: int = 6):
        from IPython.display import display, HTML

        html = self.to_html(
            top_n_features=top_n_features,
            top_n_interactions=top_n_interactions,
        )
        return display(HTML(html))
