"""shared utilities for analysis and benchmarking"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from flatcfm._utils import dense_array as dense_x  # re-exported
from flatcfm.data.space import NormalizedLog1pBaseTransform, RawCountsBaseTransform, TransformPipeline


def take_obs_rows(adata_obj: ad.AnnData, obs_names: list[str]) -> ad.AnnData:
    """take obs rows by name"""

    index = adata_obj.obs_names.get_indexer(obs_names)
    if np.any(index < 0):
        missing = [obs_names[i] for i, item in enumerate(index) if item < 0][:5]
        raise KeyError(f"Missing obs names: {missing}")
    return adata_obj[index].copy()


def control_library_size(control_adata: ad.AnnData) -> np.ndarray:
    """compute control library size from full adata"""

    return dense_x(control_adata.X).sum(axis=1).astype(np.float32)


def build_comparison_pipeline(datamodule, comparison_space_cfg: dict) -> TransformPipeline:
    """build comparison pipeline pinned to train feature names"""

    base_cfg = comparison_space_cfg["base"]
    if str(base_cfg["kind"]) == "raw_counts":
        base_transform = RawCountsBaseTransform(
            feature_set=str(base_cfg["feature_set"]),
            n_hvgs=base_cfg["n_hvgs"],
            target_sum=float(base_cfg["target_sum"]),
        )
    elif str(base_cfg["kind"]) == "normalized_log1p":
        base_transform = NormalizedLog1pBaseTransform(
            feature_set=str(base_cfg["feature_set"]),
            n_hvgs=base_cfg["n_hvgs"],
            target_sum=float(base_cfg["target_sum"]),
        )
    else:
        raise ValueError(f"Unsupported comparison base kind: {base_cfg['kind']}")

    base_transform.feature_names_in = list(datamodule.train_pipeline.feature_names_in)
    base_transform.is_fitted = True
    pipeline = TransformPipeline(
        base_transform=base_transform,
        projections=[],
        fit_scope=str(comparison_space_cfg.get("fit_scope", "full_dataset")),
    )
    pipeline.is_fitted = True
    return pipeline
