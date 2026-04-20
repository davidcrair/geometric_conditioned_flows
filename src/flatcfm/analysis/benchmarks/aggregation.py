"""benchmark aggregation helpers"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from flatcfm._utils import dense_array


AVAILABLE_AGGREGATIONS = (
    "none",
    "average",
    "scaled",
    "logfc",
    "var",
    "pca",
    "pca_average",
)


@dataclass(frozen=True)
class PcaModel:
    """lightweight pca model"""

    mean: np.ndarray
    components: np.ndarray


def _group_columns_tuple(group_columns: Sequence[str]) -> tuple[str, ...]:
    """normalize group columns"""

    return tuple(str(column) for column in group_columns)


def _group_key_to_tuple(group_key) -> tuple[str, ...]:
    """normalize group key"""

    if isinstance(group_key, tuple):
        return tuple(str(item) for item in group_key)
    return (str(group_key),)


def _key_dict(group_columns: Sequence[str], group_key) -> dict[str, str]:
    """build group key dict"""

    group_key_tuple = _group_key_to_tuple(group_key)
    return {column: value for column, value in zip(group_columns, group_key_tuple, strict=False)}


def _fit_pca(matrix: np.ndarray, n_components: int) -> PcaModel:
    """fit pca"""

    dense = dense_array(matrix)
    if dense.ndim != 2:
        raise ValueError("pca input must be 2d")
    max_components = min(dense.shape[0], dense.shape[1])
    if max_components < 1:
        return PcaModel(mean=np.zeros((dense.shape[1],), dtype=np.float32), components=np.zeros((0, dense.shape[1]), dtype=np.float32))
    actual_components = min(int(n_components), max_components)
    mean = dense.mean(axis=0, dtype=np.float64)
    centered = dense - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:actual_components].astype(np.float32, copy=False)
    return PcaModel(mean=mean.astype(np.float32, copy=False), components=components)


def _transform_pca(matrix: np.ndarray, model: PcaModel) -> np.ndarray:
    """transform matrix with pca"""

    dense = dense_array(matrix)
    if model.components.shape[0] == 0:
        return np.zeros((dense.shape[0], 0), dtype=np.float32)
    return (dense - model.mean) @ model.components.T


def list_available_aggregations() -> tuple[str, ...]:
    """list available aggregations"""

    return AVAILABLE_AGGREGATIONS


def aggregate_matrix(
    matrix: np.ndarray,
    aggregation: str,
    control_matrix: np.ndarray | None = None,
    pca_model: PcaModel | None = None,
) -> np.ndarray:
    """aggregate one matrix"""

    dense = dense_array(matrix)
    if aggregation == "none":
        return dense
    if aggregation == "average":
        return dense.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    if aggregation == "var":
        return dense.var(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    if aggregation == "scaled":
        if control_matrix is None or control_matrix.shape[0] == 0:
            mean = dense.mean(axis=0, dtype=np.float64)
            std = dense.std(axis=0, dtype=np.float64)
        else:
            ctrl = dense_array(control_matrix)
            mean = ctrl.mean(axis=0, dtype=np.float64)
            std = ctrl.std(axis=0, dtype=np.float64)
        std = np.where(std < 1e-8, 1.0, std)
        return ((dense - mean) / std).mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    if aggregation == "logfc":
        if control_matrix is None or control_matrix.shape[0] == 0:
            raise ValueError("logfc aggregation requires controls")
        ctrl = dense_array(control_matrix)
        return (dense.mean(axis=0, dtype=np.float64) - ctrl.mean(axis=0, dtype=np.float64)).astype(np.float32, copy=False)
    if aggregation == "pca":
        if pca_model is None:
            raise ValueError("pca aggregation requires a fitted pca model")
        return _transform_pca(dense, pca_model)
    if aggregation == "pca_average":
        if pca_model is None:
            raise ValueError("pca_average aggregation requires a fitted pca model")
        embedding = _transform_pca(dense, pca_model)
        return embedding.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def aggregate_adata(
    adata: ad.AnnData,
    group_columns: Sequence[str],
    aggregation: str = "average",
    control_adata: ad.AnnData | None = None,
    control_group_columns: Sequence[str] | None = None,
    fit_adata: ad.AnnData | None = None,
    pca_n_components: int = 50,
) -> pd.DataFrame:
    """aggregate anndata by group"""

    group_columns_tuple = _group_columns_tuple(group_columns)
    if control_group_columns is None:
        control_group_columns_tuple = group_columns_tuple
    else:
        control_group_columns_tuple = _group_columns_tuple(control_group_columns)
    pca_model = None
    if aggregation in {"pca", "pca_average"}:
        fit_matrix = dense_array((fit_adata or adata).X)
        pca_model = _fit_pca(fit_matrix, n_components=int(pca_n_components))

    control_lookup: dict[tuple[str, ...], np.ndarray] = {}
    if control_adata is not None and control_adata.n_obs > 0:
        if control_group_columns_tuple:
            for control_key, frame in control_adata.obs.groupby(list(control_group_columns_tuple), sort=True, observed=False):
                control_lookup[_group_key_to_tuple(control_key)] = dense_array(control_adata[frame.index].X)
        else:
            control_lookup[()] = dense_array(control_adata.X)

    rows = []
    for group_key, frame in adata.obs.groupby(list(group_columns_tuple), sort=True, observed=False):
        group_key_tuple = _group_key_to_tuple(group_key)
        values = dense_array(adata[frame.index].X)
        control_key = tuple(
            _key_dict(group_columns_tuple, group_key_tuple)[column]
            for column in control_group_columns_tuple
        )
        ctrl = control_lookup.get(control_key)
        aggregated = aggregate_matrix(
            values,
            aggregation=aggregation,
            control_matrix=ctrl,
            pca_model=pca_model,
        )
        rows.append(
            {
                **_key_dict(group_columns_tuple, group_key_tuple),
                "n_cells": int(values.shape[0]),
                "representation": aggregated,
            }
        )
    return pd.DataFrame(rows)
