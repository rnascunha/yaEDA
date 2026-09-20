from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


@dataclass
class ClusterCharacteristic:
    feature: str
    cluster_mean: float
    global_mean: float
    z_difference: float  # How many standard deviations this cluster deviates from the dataset mean


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    percentage: float
    target_mean: float | None = None
    target_distribution: dict[str, float] = field(default_factory=dict)
    defining_features: list[ClusterCharacteristic] = field(default_factory=list)


@dataclass
class ClusterReport:
    n_clusters: int
    target: str
    target_type: Literal["classification", "regression"]
    silhouette_score: float
    inertia: float
    mutual_info_with_target: float
    clusters: list[ClusterProfile]
    cluster_labels: list[int] = field(default_factory=list, repr=False)
    pca_coordinates: list[list[float]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("cluster_labels", None)
        d.pop("pca_coordinates", None)
        return d


class TabularClusterAnalyzer:
    """Segments instances via KMeans, measures target predictability, and extracts centroid profiles."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] = "regression",
        n_clusters: int = 4,
        random_state: int = 42,
    ):
        self.df = df
        self.target = target
        self.target_type = target_type
        self.n_clusters = max(2, n_clusters)
        self.random_state = random_state

        if features is not None:
            self.features = [f for f in features if f != target and f in df.columns]
        else:
            self.features = [col for col in df.columns if col != target]

    def _preprocess(self) -> tuple[np.ndarray, list[str], pd.Series]:
        valid_df = (
            self.df[self.features + [self.target]].dropna(subset=[self.target]).copy()
        )

        num_cols = [
            c
            for c in self.features
            if pd.api.types.is_numeric_dtype(valid_df[c])
            and not pd.api.types.is_bool_dtype(valid_df[c])
        ]
        cat_cols = [c for c in self.features if c not in num_cols]

        parts = []
        ordered_names = []

        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            num_data = num_imputer.fit_transform(valid_df[num_cols])
            parts.append(num_data)
            ordered_names.extend(num_cols)

        if cat_cols:
            cat_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__")
            cat_data = cat_imputer.fit_transform(valid_df[cat_cols].astype(str))
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            parts.append(enc.fit_transform(cat_data))
            ordered_names.extend(cat_cols)

        X_raw = np.hstack(parts)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        return X_scaled, ordered_names, valid_df[self.target]

    def run(self) -> ClusterReport:
        X_scaled, feature_names, target_s = self._preprocess()
        n_samples = len(X_scaled)
        k = min(self.n_clusters, max(2, n_samples - 1))

        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Silhouette score
        if n_samples > 2000:
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(n_samples, size=2000, replace=False)
            sil_score = float(
                silhouette_score(X_scaled[sample_idx], cluster_labels[sample_idx])
            )
        else:
            sil_score = float(silhouette_score(X_scaled, cluster_labels))

        # 2D PCA projection for space inspection
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_coords = pca.fit_transform(X_scaled)

        # Mutual Information between cluster label and target
        clusters_2d = cluster_labels.reshape(-1, 1)
        if self.target_type == "classification":
            y_enc = pd.factorize(target_s)[0]
            mi_score = float(
                mutual_info_classif(clusters_2d, y_enc, random_state=self.random_state)[
                    0
                ]
            )
        else:
            mi_score = float(
                mutual_info_regression(
                    clusters_2d, target_s.astype(float), random_state=self.random_state
                )[0]
            )

        # Feature characterization per cluster
        global_means = X_scaled.mean(axis=0)
        global_stds = np.where(X_scaled.std(axis=0) == 0, 1.0, X_scaled.std(axis=0))

        cluster_profiles: list[ClusterProfile] = []
        for c_id in range(k):
            mask = cluster_labels == c_id
            c_size = int(np.sum(mask))
            c_pct = round((c_size / n_samples) * 100, 2)
            c_target = target_s[mask]

            t_mean = None
            t_dist = {}
            if self.target_type == "regression":
                t_mean = round(float(c_target.astype(float).mean()), 4)
            else:
                counts = c_target.value_counts(normalize=True)
                t_dist = {
                    str(val): round(float(pct) * 100, 2) for val, pct in counts.items()
                }

            # Identify features with largest standardized displacement
            c_means = X_scaled[mask].mean(axis=0)
            z_diffs = (c_means - global_means) / global_stds

            top_feat_indices = np.argsort(np.abs(z_diffs))[::-1][:4]
            defining_feats = [
                ClusterCharacteristic(
                    feature=feature_names[i],
                    cluster_mean=round(float(c_means[i]), 3),
                    global_mean=round(float(global_means[i]), 3),
                    z_difference=round(float(z_diffs[i]), 3),
                )
                for i in top_feat_indices
            ]

            cluster_profiles.append(
                ClusterProfile(
                    cluster_id=c_id,
                    size=c_size,
                    percentage=c_pct,
                    target_mean=t_mean,
                    target_distribution=t_dist,
                    defining_features=defining_feats,
                )
            )

        return ClusterReport(
            n_clusters=k,
            target=self.target,
            target_type=self.target_type,
            silhouette_score=round(sil_score, 4),
            inertia=round(float(kmeans.inertia_), 2),
            mutual_info_with_target=round(mi_score, 4),
            clusters=cluster_profiles,
            cluster_labels=cluster_labels.tolist(),
            pca_coordinates=pca_coords.tolist(),
        )


class TabularClustersCall:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str] | None = None,
        target_type: Literal["classification", "regression"] = "regression",
        n_clusters: list[int] | None = None,
        random_state: int = 42,
    ):
        # keep unique and ordered list, and minimal value of 2
        n_clusters = n_clusters if n_clusters is not None else [4]
        n_clusters = list(dict.fromkeys([max(2, clu) for clu in n_clusters]))
        if not n_clusters:
            raise Exception("No cluster defined")  # noqa: TRY002
        self._analyzers = {
            clu: TabularClusterAnalyzer(
                df, target, features, target_type, clu, random_state
            )
            for clu in n_clusters
        }

    def analyzers(self):
        return self._analyzers

    def run(self) -> dict[int, ClusterReport]:
        return {clu: analyzer.run() for clu, analyzer in self._analyzers.items()}
