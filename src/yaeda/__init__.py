"""yaEDA - Automated Tabular EDA & Golden Feature Engineering Toolkit."""

from importlib.metadata import PackageNotFoundError, version

from .analyze import TabularEDA
from .clustering import ClusterReport, TabularClusterAnalyzer
from .correlation import CorrelationReport, FeatureTargetAnalyzer
from .extract import FeatureProfile, TableProfile, TabularDataProfiler
from .feature_importance import FeatureImportanceAnalyzer, FeatureImportanceReport
from .interaction import FeatureInteractionAnalyzer, InteractionReport
from .model_diagnostics import ModelDiagnosticsAnalyzer, ModelDiagnosticsReport

__all__ = [
    "ClusterReport",
    "CorrelationReport",
    "FeatureImportanceAnalyzer",
    "FeatureImportanceReport",
    "FeatureInteractionAnalyzer",
    "FeatureProfile",
    "FeatureTargetAnalyzer",
    "InteractionReport",
    "ModelDiagnosticsAnalyzer",
    "ModelDiagnosticsReport",
    "TableProfile",
    "TabularClusterAnalyzer",
    "TabularDataProfiler",
    "TabularEDA",
]

try:
    __version__ = version("yaeda")
except PackageNotFoundError:
    # Package is not installed in the current environment
    __version__ = "unknown"
