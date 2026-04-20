"""perturbench adapter helpers"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import pandas as pd


@dataclass
class PerturbBenchTaskMetadata:
    """perturbench task metadata"""

    perturbation_key: str
    covariate_keys: tuple[str, ...] = ()
    control_value: str | int | float | None = None
    obs_map: dict[str, str] = field(default_factory=dict)


def evaluator_to_task_metadata(evaluator: Any) -> PerturbBenchTaskMetadata:
    """extract task metadata from evaluator"""

    task_config = getattr(evaluator, "task_config", {}) or {}
    perturbation_key = task_config.get("perturbation_key", "condition")
    covariate_keys = tuple(task_config.get("covariate_keys", []))
    control_value = task_config.get("control_value")
    return PerturbBenchTaskMetadata(
        perturbation_key=perturbation_key,
        covariate_keys=covariate_keys,
        control_value=control_value,
    )


def _rename_obs(adata_obj: ad.AnnData, obs_map: dict[str, str]) -> ad.AnnData:
    """rename obs columns"""

    renamed = adata_obj.copy()
    valid_map = {src: dst for src, dst in obs_map.items() if src in renamed.obs.columns}
    if valid_map:
        renamed.obs = renamed.obs.rename(columns=valid_map)
    return renamed


def _align_var(pred_adata: ad.AnnData, ref_adata: ad.AnnData) -> ad.AnnData:
    """align prediction vars to reference"""

    common = [name for name in ref_adata.var_names if name in pred_adata.var_names]
    if not common:
        raise ValueError("No overlapping var names between predictions and reference adata")

    aligned = pred_adata[:, common].copy()
    aligned = aligned[:, ref_adata.var_names.intersection(aligned.var_names)].copy()
    if list(aligned.var_names) != list(ref_adata.var_names.intersection(aligned.var_names)):
        aligned = aligned[:, list(ref_adata.var_names.intersection(aligned.var_names))].copy()
    return aligned


def to_perturbench_predictions(
    pred_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    task_metadata: PerturbBenchTaskMetadata,
    model_name: str = "flatcfm",
) -> dict[str, ad.AnnData]:
    """convert predictions to perturbench format"""

    adapted = _rename_obs(pred_adata, task_metadata.obs_map)
    aligned = _align_var(adapted, ref_adata)

    required_cols = [task_metadata.perturbation_key, *task_metadata.covariate_keys]
    missing = [col for col in required_cols if col not in aligned.obs.columns]
    if missing:
        raise ValueError(f"Missing required obs columns for perturbench export: {missing}")

    aligned.obs = pd.DataFrame(aligned.obs).copy()
    return {model_name: aligned}
