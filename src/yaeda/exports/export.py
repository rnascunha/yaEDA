from dataclasses import asdict
import datetime
import json
from pathlib import Path
from typing import Any
import pandas as pd


class StructuredDataExporter:
    """Exports profiling, correlation, feature importance, clustering, and model diagnostics."""

    def __init__(
        self,
        table_profile: Any,
        corr_report: Any,
        importance_report: Any,
        cluster_report: list[Any] | Any | None = None,
        diagnostic_report: list[Any] | Any | None = None,
    ):
        self.tp = table_profile
        self.cr = corr_report
        self.ir = importance_report
        self.clr = cluster_report
        self.dr = diagnostic_report

    def export_json(self, output_path: str | Path | None = None) -> dict[str, Any]:
        # Support list[ClusterReport] or single ClusterReport
        cluster_payload = None
        if self.clr is not None:
            if isinstance(self.clr, list):
                cluster_payload = [c.to_dict() for c in self.clr]
            elif hasattr(self.clr, "to_dict"):
                cluster_payload = self.clr.to_dict()

        # Support list[ModelDiagnosticsReport] or single ModelDiagnosticsReport
        diagnostic_payload = None
        if self.dr is not None:
            if isinstance(self.dr, list):
                diagnostic_payload = [d.to_dict() for d in self.dr]
            elif hasattr(self.dr, "to_dict"):
                diagnostic_payload = self.dr.to_dict()

        payload = {
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "n_rows": self.tp.n_rows,
                "n_columns": self.tp.n_columns,
                "target_column": self.tp.target_column,
                "target_type": self.cr.target_type,
                "memory_usage_mb": self.tp.memory_usage_mb,
                "model_used": self.ir.model_type,
            },
            "table_profile": {
                "dtype_counts": self.tp.dtype_counts,
                "column_dtypes": self.tp.column_dtypes,
                "features": {k: asdict(v) for k, v in self.tp.features.items()},
            },
            "correlation_analysis": {
                "target_associations": {
                    k: asdict(v) for k, v in self.cr.target_associations.items()
                },
                "collinear_pairs": [asdict(p) for p in self.cr.collinear_pairs],
                "pearson_matrix": self.cr.pearson_matrix,
                "spearman_matrix": self.cr.spearman_matrix,
            },
            "feature_importance_ranking": [asdict(m) for m in self.ir.importances],
            "cluster_analysis": cluster_payload,
            "model_diagnostics": diagnostic_payload,
        }

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        return payload

    def export_csvs(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        stat_rows = []
        for feat, p in self.tp.features.items():
            assoc = self.cr.target_associations.get(feat)
            stat_rows.append(
                {
                    "feature": feat,
                    "dtype": p.dtype,
                    "is_numeric": p.is_numeric,
                    "is_target": p.is_target,
                    "total_count": p.total_count,
                    "missing_count": p.missing_count,
                    "missing_pct": p.missing_percentage,
                    "distinct_count": p.distinct_count,
                    "distinct_pct": p.distinct_percentage,
                    "zero_count": p.zero_count,
                    "zero_pct": p.zero_percentage,
                    "min": p.min_value,
                    "median": p.median,
                    "mean": p.mean,
                    "max": p.max_value,
                    "iqr": p.iqr,
                    "std_dev": p.std_dev,
                    "skewness": p.skewness,
                    "outliers_count": p.outliers.count if p.outliers else 0,
                    "pearson_to_target": assoc.pearson_corr if assoc else None,
                    "mutual_info": assoc.mutual_info if assoc else None,
                }
            )

        csv_results = {
            "statistics": pd.DataFrame(stat_rows).to_csv(index=False),
            "golden_features": self.ir.to_dataframe().to_csv(index=True),
            "collinear_pairs": pd.DataFrame([asdict(p) for p in self.cr.collinear_pairs]).to_csv(
                index=False
            ),
            "correlation_matrix": pd.DataFrame(self.cr.pearson_matrix).to_csv(),
        }

        dir_path = Path(output_dir) if output_dir else None
        if dir_path:
            dir_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(stat_rows).to_csv(dir_path / "feature_statistics.csv", index=False)
            self.ir.to_dataframe().to_csv(dir_path / "golden_features.csv", index=True)
            pd.DataFrame([asdict(p) for p in self.cr.collinear_pairs]).to_csv(
                dir_path / "collinear_pairs.csv", index=False
            )
            pd.DataFrame(self.cr.pearson_matrix).to_csv(dir_path / "pearson_correlation_matrix.csv")

        # Cluster profiles export (iterating over lists safely)
        if self.clr is not None:
            cluster_list = self.clr if isinstance(self.clr, list) else [self.clr]
            cluster_rows = []
            for rep in cluster_list:
                for c in rep.clusters:
                    top_feats = ", ".join(
                        [f"{f.feature} (z={f.z_difference:+.2f})" for f in c.defining_features]
                    )
                    cluster_rows.append(
                        {
                            "n_clusters": rep.n_clusters,
                            "cluster_id": c.cluster_id,
                            "size": c.size,
                            "percentage": c.percentage,
                            "target_mean": c.target_mean,
                            "target_distribution": json.dumps(c.target_distribution),
                            "defining_features": top_feats,
                        }
                    )
            cluster_df = pd.DataFrame(cluster_rows)
            if dir_path:
                cluster_df.to_csv(dir_path / "cluster_profiles.csv", index=False)
            csv_results["cluster_profiles"] = cluster_df.to_csv(index=False)

        # Model diagnostics export
        if self.dr is not None:
            diag_list = self.dr if isinstance(self.dr, list) else [self.dr]
            diag_rows = []
            for rep in diag_list:
                row = {
                    "name": rep.name,
                    "target": rep.target,
                    "target_type": rep.target_type,
                    "total_samples": rep.total_samples,
                    "error_count": rep.error_count,
                    "error_rate": rep.error_rate,
                }
                row.update(rep.metrics)
                diag_rows.append(row)
            diag_df = pd.DataFrame(diag_rows)
            if dir_path:
                diag_df.to_csv(dir_path / "model_diagnostics.csv", index=False)
            csv_results["model_diagnostics"] = diag_df.to_csv(index=False)

        return csv_results
