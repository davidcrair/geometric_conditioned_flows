"""transform pipeline for model spaces"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import logging
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import torch

from flatcfm._utils import dense_array

logger = logging.getLogger(__name__)


def _slug(value: str) -> str:
    """slug string"""

    return str(value).lower().replace(" ", "-").replace("/", "-").replace(":", "-")


@dataclass
class RawSpaceData:
    """raw space payload"""

    matrix: np.ndarray
    library_size: np.ndarray
    feature_names: list[str]


class BaseTransform(ABC):
    """base transform"""

    kind = "base"

    def __init__(
        self,
        feature_set: str = "all_genes",
        n_hvgs: int | None = None,
        target_sum: float = 1e4,
        hvg_batch_key: str | None = None,
        deg_control_column: str | None = None,
        deg_control_value: str | int | None = None,
        deg_perturbation_column: str | None = None,
        deg_cell_type_column: str | None = None,
        deg_n_top_genes: int = 25,
        precomputed_feature_names: list[str] | None = None,
    ):
        self.feature_set = str(feature_set)
        self.n_hvgs = None if n_hvgs is None else int(n_hvgs)
        self.target_sum = float(target_sum)
        self.hvg_batch_key = hvg_batch_key
        self.deg_control_column = deg_control_column
        self.deg_control_value = deg_control_value
        self.deg_perturbation_column = deg_perturbation_column
        self.deg_cell_type_column = deg_cell_type_column
        self.deg_n_top_genes = int(deg_n_top_genes)
        self.precomputed_feature_names = list(precomputed_feature_names) if precomputed_feature_names else None
        self.feature_names_in: list[str] = []
        self.is_fitted = False

    def fit(self, adata: sc.AnnData) -> BaseTransform:
        """fit transform"""

        if self.precomputed_feature_names is not None:
            self.feature_names_in = list(self.precomputed_feature_names)
        else:
            self.feature_names_in = self._select_feature_names(adata)
        self.is_fitted = True
        return self

    def _select_feature_names(self, adata: sc.AnnData) -> list[str]:
        """select feature names"""

        if self.feature_set == "all_genes":
            return list(adata.var_names)
        if self.feature_set not in {"hvg", "hvg_union_deg"}:
            raise ValueError(f"Unsupported feature_set: {self.feature_set}")
        hvg_names = self._select_hvg(adata)
        if self.feature_set == "hvg":
            return hvg_names
        # hvg_union_deg: union of HVG + DEG at highest dose per drug per cell type
        deg_names = self._select_deg(adata)
        union = list(dict.fromkeys(hvg_names + deg_names))  # preserve order, deduplicate
        logger.info(
            "hvg_union_deg: %d hvg + %d deg (%d new) = %d total features",
            len(hvg_names), len(deg_names), len(set(deg_names) - set(hvg_names)), len(union),
        )
        return union

    def _select_hvg(self, adata: sc.AnnData) -> list[str]:
        """select highly variable genes"""

        if self.n_hvgs is None:
            raise ValueError("n_hvgs must be set for hvg feature selection")
        if self.n_hvgs > adata.n_vars:
            raise ValueError(f"n_hvgs={self.n_hvgs} exceeds number of genes {adata.n_vars}")
        temp = adata.copy()
        hvg_kwargs = {"n_top_genes": self.n_hvgs, "flavor": "seurat_v3", "subset": False}
        if self.hvg_batch_key is not None and self.hvg_batch_key in temp.obs.columns:
            hvg_kwargs["batch_key"] = self.hvg_batch_key
        sc.pp.highly_variable_genes(temp, **hvg_kwargs)
        return adata.var_names[temp.var["highly_variable"]].tolist()

    def _select_deg(self, adata: sc.AnnData) -> list[str]:
        """select top DEGs per perturbation per cell type using scanpy rank_genes_groups

        for each cell type runs sc.tl.rank_genes_groups with the control group
        as reference and keeps the top deg_n_top_genes per perturbation group
        returns the union across all cell types and perturbations
        """

        if self.deg_control_column is None:
            raise ValueError("deg_control_column required for hvg_union_deg")

        pert_col = str(self.deg_perturbation_column or "product_name")
        ct_col = str(self.deg_cell_type_column or "cell_type")
        ctrl_col = str(self.deg_control_column)
        ctrl_val = self.deg_control_value
        n_top = self.deg_n_top_genes

        # normalize for DEG computation
        temp = adata.copy()
        sc.pp.normalize_total(temp, target_sum=self.target_sum)
        sc.pp.log1p(temp)

        # identify the control label in the perturbation column
        ctrl_mask = temp.obs[ctrl_col] == ctrl_val
        ctrl_pert_labels = temp.obs.loc[ctrl_mask, pert_col].unique()
        if len(ctrl_pert_labels) == 0:
            raise ValueError(f"no control cells found with {ctrl_col}=={ctrl_val}")
        reference_label = str(ctrl_pert_labels[0])

        all_deg: set[str] = set()
        cell_types = temp.obs[ct_col].unique().tolist()

        for ct in cell_types:
            subset = temp[temp.obs[ct_col].astype(str) == str(ct)].copy()
            # need at least the reference group plus one other
            groups = subset.obs[pert_col].unique()
            if reference_label not in groups.astype(str) or len(groups) < 2:
                continue
            sc.tl.rank_genes_groups(subset, groupby=pert_col, reference=reference_label)
            result_df = sc.get.rank_genes_groups_df(subset, group=None)
            # take top n_top genes per perturbation group
            for _, grp_df in result_df.groupby("group"):
                top_genes = grp_df.nlargest(n_top, "scores")["names"].tolist()
                all_deg.update(top_genes)

        # return in original gene order
        all_gene_names = list(temp.var_names)
        deg_ordered = [g for g in all_gene_names if g in all_deg]
        logger.info("DEG selection: %d genes from top %d per perturbation across %d cell types",
                     len(deg_ordered), n_top, len(cell_types))
        return deg_ordered

    def _require_fitted(self) -> None:
        """require fitted"""

        if not self.is_fitted:
            raise ValueError("Base transform must be fitted before transformation")

    def _align_features(
        self,
        matrix: np.ndarray,
        feature_names_in: list[str],
        desired_names: list[str],
    ) -> np.ndarray:
        """align features"""

        lookup = {name: idx for idx, name in enumerate(feature_names_in)}
        missing = [name for name in desired_names if name not in lookup]
        if missing:
            raise ValueError(f"Missing required features for space transform: {missing[:5]}")
        idx = [lookup[name] for name in desired_names]
        return np.asarray(matrix[:, idx], dtype=np.float32)

    @abstractmethod
    def transform(self, adata: sc.AnnData) -> RawSpaceData:
        """transform adata"""

    @abstractmethod
    def transform_raw(
        self,
        raw_matrix: np.ndarray,
        library_size: np.ndarray,
        feature_names_in: list[str],
    ) -> RawSpaceData:
        """transform raw matrix"""

    @abstractmethod
    def inverse_to_raw(
        self,
        base_matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse to raw"""

    def feature_names_out(self) -> list[str]:
        """feature names out"""

        self._require_fitted()
        return list(self.feature_names_in)

    def export_spec(self) -> dict:
        """export spec"""

        spec = {
            "kind": self.kind,
            "feature_set": self.feature_set,
            "n_hvgs": self.n_hvgs,
            "target_sum": self.target_sum,
            "feature_names_in": list(self.feature_names_in),
        }
        if self.feature_set == "hvg_union_deg":
            spec["deg_control_column"] = self.deg_control_column
            spec["deg_control_value"] = self.deg_control_value
            spec["deg_perturbation_column"] = self.deg_perturbation_column
            spec["deg_cell_type_column"] = self.deg_cell_type_column
            spec["deg_n_top_genes"] = self.deg_n_top_genes
        return spec


class RawCountsBaseTransform(BaseTransform):
    """raw identity base transform"""

    kind = "raw_counts"

    def transform(self, adata: sc.AnnData) -> RawSpaceData:
        """transform adata"""

        self._require_fitted()
        raw_all = dense_array(adata.X)
        library_size = raw_all.sum(axis=1).astype(np.float32)
        raw_subset = dense_array(adata[:, self.feature_names_in].X)
        return RawSpaceData(
            matrix=np.asarray(raw_subset, dtype=np.float32),
            library_size=library_size,
            feature_names=list(self.feature_names_in),
        )

    def transform_raw(
        self,
        raw_matrix: np.ndarray,
        library_size: np.ndarray,
        feature_names_in: list[str],
    ) -> RawSpaceData:
        """transform raw matrix"""

        self._require_fitted()
        del library_size
        aligned = self._align_features(raw_matrix, feature_names_in, self.feature_names_in)
        aligned_library = np.asarray(aligned.sum(axis=1), dtype=np.float32)
        return RawSpaceData(
            matrix=np.asarray(aligned, dtype=np.float32),
            library_size=aligned_library,
            feature_names=list(self.feature_names_in),
        )

    def inverse_to_raw(
        self,
        base_matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse to raw"""

        del library_size, sample
        return np.asarray(base_matrix, dtype=np.float32)


def _check_library_size_vs_subset(
    library_size: np.ndarray,
    subset_matrix: np.ndarray,
) -> None:
    """check that library_size >= subset column sums

    the library size passed to transform_raw should be the total counts across ALL
    genes not just the HVG subset - full-gene library size must be at least as large
    as the HVG subset sum
    """

    if subset_matrix.shape[0] == 0:
        return
    subset_sums = np.asarray(subset_matrix, dtype=np.float32).sum(axis=1)
    lib = np.asarray(library_size, dtype=np.float32)
    # allow small floating point tolerance
    violations = np.sum(lib < subset_sums - 1.0)
    if violations > 0:
        median_lib = float(np.median(lib))
        median_subset = float(np.median(subset_sums))
        logger.warning(
            "library_size < subset column sums for %d / %d cells "
            "(median lib=%.1f median subset=%.1f) - "
            "this likely means library_size was computed from HVG-only counts "
            "instead of full-gene counts which will inflate normalized values",
            violations,
            len(lib),
            median_lib,
            median_subset,
        )


class NormalizedLog1pBaseTransform(BaseTransform):
    """normalized log1p base transform"""

    kind = "normalized_log1p"

    def _normalize(self, raw_subset: np.ndarray, library_size: np.ndarray) -> np.ndarray:
        """normalize raw subset"""

        denom = np.clip(np.asarray(library_size, dtype=np.float32)[:, None], a_min=1.0, a_max=None)
        return np.log1p(np.asarray(raw_subset, dtype=np.float32) / denom * self.target_sum).astype(np.float32)

    def transform(self, adata: sc.AnnData) -> RawSpaceData:
        """transform adata"""

        self._require_fitted()
        raw_all = dense_array(adata.X)
        library_size = raw_all.sum(axis=1).astype(np.float32)
        raw_subset = dense_array(adata[:, self.feature_names_in].X)
        return RawSpaceData(
            matrix=self._normalize(raw_subset, library_size),
            library_size=library_size,
            feature_names=list(self.feature_names_in),
        )

    def transform_raw(
        self,
        raw_matrix: np.ndarray,
        library_size: np.ndarray,
        feature_names_in: list[str],
    ) -> RawSpaceData:
        """transform raw matrix"""

        self._require_fitted()
        aligned = self._align_features(raw_matrix, feature_names_in, self.feature_names_in)
        _check_library_size_vs_subset(library_size, aligned)
        return RawSpaceData(
            matrix=self._normalize(aligned, library_size),
            library_size=np.asarray(library_size, dtype=np.float32),
            feature_names=list(self.feature_names_in),
        )

    def inverse_to_raw(
        self,
        base_matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse to raw"""

        del sample
        raw = np.expm1(np.asarray(base_matrix, dtype=np.float32)).clip(min=0.0)
        denom = np.asarray(library_size, dtype=np.float32)[:, None] / self.target_sum
        return np.asarray(raw * denom, dtype=np.float32)


class CompositionBaseTransform(BaseTransform):
    """composition (probability simplex) base transform

    maps raw counts to proportions: p_i = count_i / library_size
    adds a small pseudocount and renormalizes so no entry is exactly zero
    the resulting vector lies on the probability simplex (sums to 1 all >= 0)
    """

    kind = "composition"

    def _normalize(self, raw_subset: np.ndarray, library_size: np.ndarray) -> np.ndarray:
        """normalize raw subset to probability simplex"""

        denom = np.clip(np.asarray(library_size, dtype=np.float32)[:, None], a_min=1.0, a_max=None)
        p = np.asarray(raw_subset, dtype=np.float32) / denom
        p = p + 1e-8
        p = p / p.sum(axis=1, keepdims=True)
        return p.astype(np.float32)

    def transform(self, adata: sc.AnnData) -> RawSpaceData:
        """transform adata"""

        self._require_fitted()
        raw_all = dense_array(adata.X)
        library_size = raw_all.sum(axis=1).astype(np.float32)
        raw_subset = dense_array(adata[:, self.feature_names_in].X)
        return RawSpaceData(
            matrix=self._normalize(raw_subset, library_size),
            library_size=library_size,
            feature_names=list(self.feature_names_in),
        )

    def transform_raw(
        self,
        raw_matrix: np.ndarray,
        library_size: np.ndarray,
        feature_names_in: list[str],
    ) -> RawSpaceData:
        """transform raw matrix"""

        self._require_fitted()
        aligned = self._align_features(raw_matrix, feature_names_in, self.feature_names_in)
        _check_library_size_vs_subset(library_size, aligned)
        return RawSpaceData(
            matrix=self._normalize(aligned, library_size),
            library_size=np.asarray(library_size, dtype=np.float32),
            feature_names=list(self.feature_names_in),
        )

    def inverse_to_raw(
        self,
        base_matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse to raw: proportions * library_size"""

        del sample
        p = np.clip(np.asarray(base_matrix, dtype=np.float32), a_min=0.0, a_max=None)
        row_sums = p.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, a_min=1e-10, a_max=None)
        p = p / row_sums
        raw = p * np.asarray(library_size, dtype=np.float32)[:, None]
        return np.asarray(raw, dtype=np.float32)


class ProjectionTransform(ABC):
    """projection transform"""

    kind = "projection"

    def __init__(self):
        self.input_feature_names: list[str] = []
        self.output_feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, matrix: np.ndarray, feature_names: list[str]) -> ProjectionTransform:
        """fit projection"""

        self.input_feature_names = list(feature_names)
        self._fit(np.asarray(matrix, dtype=np.float32))
        self.output_feature_names = self._build_feature_names_out()
        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        """require fitted"""

        if not self.is_fitted:
            raise ValueError("Projection transform must be fitted before transformation")

    def feature_names_out(self) -> list[str]:
        """feature names out"""

        self._require_fitted()
        return list(self.output_feature_names)

    @abstractmethod
    def _fit(self, matrix: np.ndarray) -> None:
        """fit projection"""

    @abstractmethod
    def _build_feature_names_out(self) -> list[str]:
        """build output feature names"""

    @abstractmethod
    def transform(self, matrix: np.ndarray, device: str = "cpu") -> np.ndarray:
        """transform matrix"""

    @abstractmethod
    def inverse_transform(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse transform"""

    @abstractmethod
    def export_spec(self) -> dict:
        """export spec"""


class IdentityProjection(ProjectionTransform):
    """identity projection"""

    kind = "identity"

    def _fit(self, matrix: np.ndarray) -> None:
        """fit projection"""

        del matrix

    def _build_feature_names_out(self) -> list[str]:
        """build output feature names"""

        return list(self.input_feature_names)

    def transform(self, matrix: np.ndarray, device: str = "cpu") -> np.ndarray:
        """transform matrix"""

        del device
        self._require_fitted()
        return np.asarray(matrix, dtype=np.float32)

    def inverse_transform(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse transform"""

        del library_size, sample
        self._require_fitted()
        return np.asarray(matrix, dtype=np.float32)

    def export_spec(self) -> dict:
        """export spec"""

        return {
            "kind": self.kind,
            "input_feature_names": list(self.input_feature_names),
            "output_feature_names": list(self.output_feature_names),
        }


class PCAProjection(ProjectionTransform):
    """pca projection"""

    kind = "pca"

    def __init__(self, n_components: int = 50):
        super().__init__()
        self.requested_components = int(n_components)
        self.n_components = int(n_components)
        self.model: Optional[PCA] = None

    def _fit(self, matrix: np.ndarray) -> None:
        """fit projection"""

        n_components = min(self.requested_components, matrix.shape[1], matrix.shape[0])
        self.n_components = int(n_components)
        self.model = PCA(n_components=n_components, svd_solver="auto", random_state=0)
        self.model.fit(matrix)

    def _build_feature_names_out(self) -> list[str]:
        """build output feature names"""

        return [f"pc_{idx}" for idx in range(self.n_components)]

    def transform(self, matrix: np.ndarray, device: str = "cpu") -> np.ndarray:
        """transform matrix"""

        del device
        self._require_fitted()
        return np.asarray(self.model.transform(np.asarray(matrix, dtype=np.float32)), dtype=np.float32)

    def inverse_transform(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse transform"""

        del library_size, sample
        self._require_fitted()
        return np.asarray(self.model.inverse_transform(np.asarray(matrix, dtype=np.float32)), dtype=np.float32)

    def export_spec(self) -> dict:
        """export spec"""

        return {
            "kind": self.kind,
            "n_components": self.n_components,
            "input_feature_names": list(self.input_feature_names),
            "output_feature_names": list(self.output_feature_names),
        }


class OrthogonalLiftProjection(ProjectionTransform):
    """orthogonal lift projection"""

    kind = "orthogonal_lift"

    def __init__(self, ambient_dim: int, seed: int = 0):
        super().__init__()
        self.ambient_dim = int(ambient_dim)
        self.seed = int(seed)
        self.basis: Optional[np.ndarray] = None

    def _fit(self, matrix: np.ndarray) -> None:
        """fit projection"""

        input_dim = int(matrix.shape[1])
        if self.ambient_dim < input_dim:
            raise ValueError(f"ambient_dim={self.ambient_dim} must be >= input_dim={input_dim}")
        rng = np.random.default_rng(self.seed)
        random_matrix = rng.standard_normal((self.ambient_dim, input_dim)).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)
        self.basis = np.asarray(q[:, :input_dim], dtype=np.float32)

    def _build_feature_names_out(self) -> list[str]:
        """build output feature names"""

        return [f"lift_{idx}" for idx in range(self.ambient_dim)]

    def transform(self, matrix: np.ndarray, device: str = "cpu") -> np.ndarray:
        """transform matrix"""

        del device
        self._require_fitted()
        return np.asarray(np.asarray(matrix, dtype=np.float32) @ self.basis.T, dtype=np.float32)

    def inverse_transform(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse transform"""

        del library_size, sample
        self._require_fitted()
        return np.asarray(np.asarray(matrix, dtype=np.float32) @ self.basis, dtype=np.float32)

    def export_spec(self) -> dict:
        """export spec"""

        return {
            "kind": self.kind,
            "ambient_dim": self.ambient_dim,
            "seed": self.seed,
            "input_feature_names": list(self.input_feature_names),
            "output_feature_names": list(self.output_feature_names),
        }


class NonlinearRFFLiftProjection(ProjectionTransform):
    """nonlinear rff lift projection"""

    kind = "nonlinear_rff_lift"

    def __init__(self, ambient_dim: int, seed: int = 0, feature_scale: float = 1.0):
        super().__init__()
        self.ambient_dim = int(ambient_dim)
        self.seed = int(seed)
        self.feature_scale = float(feature_scale)
        self.input_dim = 0
        self.frequency_matrix: Optional[np.ndarray] = None
        self.phase: Optional[np.ndarray] = None

    def _fit(self, matrix: np.ndarray) -> None:
        """fit projection"""

        self.input_dim = int(matrix.shape[1])
        if self.ambient_dim < self.input_dim:
            raise ValueError(f"ambient_dim={self.ambient_dim} must be >= input_dim={self.input_dim}")
        extra_dim = self.ambient_dim - self.input_dim
        if extra_dim == 0:
            self.frequency_matrix = np.zeros((0, self.input_dim), dtype=np.float32)
            self.phase = np.zeros((0,), dtype=np.float32)
            return
        rng = np.random.default_rng(self.seed)
        scale = max(self.feature_scale, 1e-6)
        self.frequency_matrix = np.asarray(
            rng.standard_normal((extra_dim, self.input_dim)) / scale,
            dtype=np.float32,
        )
        self.phase = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=extra_dim), dtype=np.float32)

    def _build_feature_names_out(self) -> list[str]:
        """build output feature names"""

        extra_dim = self.ambient_dim - len(self.input_feature_names)
        return list(self.input_feature_names) + [f"rff_{idx}" for idx in range(extra_dim)]

    def transform(self, matrix: np.ndarray, device: str = "cpu") -> np.ndarray:
        """transform matrix"""

        del device
        self._require_fitted()
        current = np.asarray(matrix, dtype=np.float32)
        if self.frequency_matrix is None or self.phase is None:
            raise ValueError("rff parameters are missing")
        if self.frequency_matrix.shape[0] == 0:
            return current
        phases = np.asarray(current @ self.frequency_matrix.T + self.phase[None, :], dtype=np.float32)
        features = np.sqrt(2.0 / float(self.frequency_matrix.shape[0])) * np.cos(phases)
        return np.asarray(np.concatenate([current, features.astype(np.float32)], axis=1), dtype=np.float32)

    def inverse_transform(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse transform"""

        del library_size, sample
        self._require_fitted()
        current = np.asarray(matrix, dtype=np.float32)
        return np.asarray(current[:, : self.input_dim], dtype=np.float32)

    def export_spec(self) -> dict:
        """export spec"""

        return {
            "kind": self.kind,
            "ambient_dim": self.ambient_dim,
            "seed": self.seed,
            "feature_scale": self.feature_scale,
            "input_feature_names": list(self.input_feature_names),
            "output_feature_names": list(self.output_feature_names),
        }


class AELatentProjection(ProjectionTransform):
    """ae latent projection"""

    kind = "ae_latent"

    def __init__(
        self,
        ae_model,
        artifact_tag: str | None = None,
        input_feature_names: Optional[list[str]] = None,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        self.ae_model = ae_model
        self.artifact_tag = artifact_tag
        self.latent_dim = int(latent_dim or getattr(ae_model, "latent_dim", 0))
        if input_feature_names:
            self.input_feature_names = list(input_feature_names)
            self.output_feature_names = [f"latent_{idx}" for idx in range(self.latent_dim)]
            self.is_fitted = True

    def _fit(self, matrix: np.ndarray) -> None:
        """fit projection"""

        if self.ae_model is None:
            raise ValueError("ae model is required for ae latent projection")
        if self.latent_dim <= 0:
            self.latent_dim = int(getattr(self.ae_model, "latent_dim", matrix.shape[1]))

    def _build_feature_names_out(self) -> list[str]:
        """build output feature names"""

        return [f"latent_{idx}" for idx in range(self.latent_dim)]

    def transform(self, matrix: np.ndarray, device: str = "cpu") -> np.ndarray:
        """transform matrix"""

        self._require_fitted()
        x_tensor = torch.as_tensor(np.asarray(matrix, dtype=np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            latent = self.ae_model.encode(x_tensor)
        return latent.detach().cpu().numpy().astype(np.float32)

    def inverse_transform(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse transform"""

        del sample
        self._require_fitted()
        z = torch.as_tensor(np.asarray(matrix, dtype=np.float32), dtype=torch.float32)
        lib = torch.as_tensor(np.asarray(library_size, dtype=np.float32), dtype=torch.float32)
        recon = self.ae_model.reconstruct_input(z, lib)
        return recon.detach().cpu().numpy().astype(np.float32)

    def inverse_to_raw(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse to raw"""

        z = torch.as_tensor(np.asarray(matrix, dtype=np.float32), dtype=torch.float32)
        lib = torch.as_tensor(np.asarray(library_size, dtype=np.float32), dtype=torch.float32)
        if not hasattr(self.ae_model, "reconstruct_counts"):
            raise ValueError("ae model does not support raw reconstruction")
        counts = self.ae_model.reconstruct_counts(z, lib, sample=sample)
        return counts.detach().cpu().numpy().astype(np.float32)

    def export_spec(self) -> dict:
        """export spec"""

        return {
            "kind": self.kind,
            "artifact_tag": self.artifact_tag,
            "latent_dim": self.latent_dim,
            "input_feature_names": list(self.input_feature_names),
            "output_feature_names": list(self.output_feature_names),
        }


class TransformPipeline:
    """transform pipeline"""

    def __init__(
        self,
        base_transform: BaseTransform,
        projections: list[ProjectionTransform] | None = None,
        fit_scope: str = "train",
    ):
        self.base_transform = base_transform
        self.projections = list(projections or [])
        self.fit_scope = str(fit_scope)
        self.is_fitted = False

    def fit(self, adata: sc.AnnData, hvg_adata: sc.AnnData | None = None) -> TransformPipeline:
        """fit pipeline

        Args:
            adata: adata used for fitting projections (pca etc)
            hvg_adata: optional separate adata for hvg gene selection only
                when None uses adata for both
        """

        self.base_transform.fit(hvg_adata if hvg_adata is not None else adata)
        raw_space = self.base_transform.transform(adata)
        matrix = raw_space.matrix
        feature_names = list(raw_space.feature_names)
        for projection in self.projections:
            projection.fit(matrix, feature_names)
            matrix = projection.transform(matrix)
            feature_names = projection.feature_names_out()
        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        """require fitted"""

        if not self.is_fitted:
            raise ValueError("Transform pipeline must be fitted before transformation")

    @property
    def feature_names_in(self) -> list[str]:
        """feature names in"""

        return list(self.base_transform.feature_names_out())

    def feature_names_out(self) -> list[str]:
        """feature names out"""

        if self.projections:
            return self.projections[-1].feature_names_out()
        return self.base_transform.feature_names_out()

    def to_latent(self, adata: sc.AnnData, device: str = "cpu") -> torch.Tensor:
        """transform adata to tensor"""

        matrix, _, _ = self.transform(adata, device=device)
        return torch.as_tensor(matrix, dtype=torch.float32, device=device)

    def transform(self, adata: sc.AnnData, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, list[str]]:
        """transform adata"""

        self._require_fitted()
        raw_space = self.base_transform.transform(adata)
        matrix = raw_space.matrix
        feature_names = list(raw_space.feature_names)
        for projection in self.projections:
            matrix = projection.transform(matrix, device=device)
            feature_names = projection.feature_names_out()
        return np.asarray(matrix, dtype=np.float32), raw_space.library_size, feature_names

    def transform_raw(
        self,
        raw_matrix: np.ndarray,
        library_size: np.ndarray,
        feature_names_in: list[str],
        device: str = "cpu",
    ) -> np.ndarray:
        """transform raw data"""

        self._require_fitted()
        raw_space = self.base_transform.transform_raw(raw_matrix, library_size, feature_names_in)
        matrix = raw_space.matrix
        for projection in self.projections:
            matrix = projection.transform(matrix, device=device)
        return np.asarray(matrix, dtype=np.float32)

    def inverse_to_base(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse to base"""

        self._require_fitted()
        current = np.asarray(matrix, dtype=np.float32)
        for projection in reversed(self.projections):
            current = projection.inverse_transform(current, library_size, sample=sample)
        return np.asarray(current, dtype=np.float32)

    def inverse_to_raw(
        self,
        matrix: np.ndarray,
        library_size: np.ndarray,
        sample: bool = False,
    ) -> np.ndarray:
        """inverse to raw"""

        self._require_fitted()
        if self.projections and isinstance(self.projections[-1], AELatentProjection):
            intermediate = np.asarray(matrix, dtype=np.float32)
            tail = self.projections[-1]
            current = tail.inverse_to_raw(intermediate, library_size, sample=sample)
            for projection in reversed(self.projections[:-1]):
                current = projection.inverse_transform(current, library_size, sample=sample)
            return np.asarray(current, dtype=np.float32)
        base_matrix = self.inverse_to_base(matrix, library_size, sample=sample)
        return self.base_transform.inverse_to_raw(base_matrix, library_size, sample=sample)

    def export_spec(self) -> dict:
        """export spec"""

        return {
            "base": self.base_transform.export_spec(),
            "projections": [projection.export_spec() for projection in self.projections],
            "fit_scope": self.fit_scope,
            "feature_names_out": self.feature_names_out(),
        }

    def pipeline_tag(self) -> str:
        """build pipeline tag"""

        base_spec = self.base_transform.export_spec()
        parts = [
            _slug(base_spec["kind"]),
            _slug(base_spec["feature_set"]),
            f"nhvg{base_spec['n_hvgs'] if base_spec['n_hvgs'] is not None else 'all'}",
            f"tsum{_slug(str(int(base_spec['target_sum']) if float(base_spec['target_sum']).is_integer() else base_spec['target_sum']))}",
        ]
        if not self.projections:
            parts.append("proj-none")
        for projection in self.projections:
            proj_spec = projection.export_spec()
            if proj_spec["kind"] == "identity":
                parts.append("proj-identity")
            elif proj_spec["kind"] == "pca":
                parts.append(f"proj-pca-npc{proj_spec['n_components']}")
            elif proj_spec["kind"] == "orthogonal_lift":
                parts.append(f"proj-ortholift-d{proj_spec['ambient_dim']}-s{proj_spec['seed']}")
            elif proj_spec["kind"] == "nonlinear_rff_lift":
                scale_value = proj_spec["feature_scale"]
                scale_tag = int(scale_value) if float(scale_value).is_integer() else scale_value
                parts.append(
                    f"proj-nonlinearrff-d{proj_spec['ambient_dim']}-s{proj_spec['seed']}-fs{_slug(str(scale_tag))}"
                )
            elif proj_spec["kind"] == "ae_latent":
                tag = proj_spec.get("artifact_tag") or "auto"
                parts.append(f"proj-ae-tag{_slug(tag)}-lat{proj_spec['latent_dim']}")
            else:
                parts.append(f"proj-{_slug(proj_spec['kind'])}")
        return "_".join(parts)
