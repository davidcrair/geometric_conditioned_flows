"""metric space preparation helpers"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from flatcfm.analysis.benchmarks._utils import (
    build_comparison_pipeline as _comparison_pipeline,
    control_library_size as _control_library_size,
    take_obs_rows as _take_obs_rows,
)
from flatcfm.data.space import (
    get_or_build_pipeline,
    normalize_space_config,
    pipeline_label,
    pipeline_tag_from_config,
    transform_adata,
)
from flatcfm.data.space import TransformPipeline


@dataclass(frozen=True)
class MetricSpaceSpec:
    """metric space spec"""

    name: str = "comparison"
    kind: str = "comparison"
    pca_n_components: int = 50
    fit_split: str = "train"


@dataclass(frozen=True)
class MetricSpaceViews:
    """metric space views"""

    prediction_adata: ad.AnnData
    reference_adata: ad.AnnData
    control_adata: ad.AnnData
    obs: pd.DataFrame
    feature_names: tuple[str, ...]
    metric_space_label: str
    metric_space_spec: MetricSpaceSpec


def normalize_metric_space_spec(metric_space: MetricSpaceSpec | dict | None) -> MetricSpaceSpec:
    """normalize metric space spec"""

    if isinstance(metric_space, MetricSpaceSpec):
        return metric_space
    cfg = dict(metric_space or {})
    kind = str(cfg.get("kind", cfg.get("name", "comparison")))
    name = str(cfg.get("name", kind))
    return MetricSpaceSpec(
        name=name,
        kind=kind,
        pca_n_components=int(cfg.get("pca_n_components", 50)),
        fit_split=str(cfg.get("fit_split", "train")),
    )


def _metric_space_config(
    datamodule,
    metric_space: MetricSpaceSpec,
    comparison_space_cfg: dict | None,
) -> dict:
    """build metric space config"""

    if metric_space.kind == "comparison":
        if comparison_space_cfg is None:
            raise ValueError("comparison space config is required for comparison metric space")
        return normalize_space_config(comparison_space_cfg, default_fit_scope="full_dataset")
    if metric_space.kind == "train_base":
        return normalize_space_config(
            {
                "base": deepcopy(datamodule.space_cfg["base"]),
                "projections": [],
                "fit_scope": metric_space.fit_split,
                "chunk_size": int(datamodule.space_cfg.get("chunk_size", 2048)),
            },
            default_fit_scope=str(metric_space.fit_split),
        )
    if metric_space.kind == "train_pca":
        return normalize_space_config(
            {
                "base": deepcopy(datamodule.space_cfg["base"]),
                "projections": [
                    {
                        "kind": "pca",
                        "n_components": int(metric_space.pca_n_components),
                    }
                ],
                "fit_scope": metric_space.fit_split,
                "chunk_size": int(datamodule.space_cfg.get("chunk_size", 2048)),
            },
            default_fit_scope=str(metric_space.fit_split),
        )
    raise ValueError(f"Unsupported metric space kind: {metric_space.kind}")


def _pipeline_cache_key(space_cfg: dict) -> str:
    """build pipeline cache key"""

    return (
        f"metric_space_pipeline:{pipeline_tag_from_config(space_cfg)}:"
        f"fitscope-{space_cfg.get('fit_scope', 'train')}"
    )


def _get_metric_pipeline(
    datamodule,
    metric_space: MetricSpaceSpec,
    comparison_pipeline: TransformPipeline | None,
    comparison_space_cfg: dict | None,
    cache: dict[str, Any] | None,
) -> tuple[TransformPipeline, dict, str]:
    """load metric space pipeline"""

    if metric_space.kind == "comparison" and comparison_pipeline is not None and comparison_space_cfg is not None:
        return comparison_pipeline, _metric_space_config(datamodule, metric_space, comparison_space_cfg), pipeline_label(comparison_space_cfg)

    space_cfg = _metric_space_config(datamodule, metric_space, comparison_space_cfg)
    key = _pipeline_cache_key(space_cfg)
    if cache is not None and key in cache:
        pipeline = cache[key]
    else:
        # reuse training pipeline's feature names for train_base/train_pca
        # to avoid recomputing DEG selection which is data-dependent
        base_feature_names = None
        if metric_space.kind in {"train_base", "train_pca"} and hasattr(datamodule, "train_pipeline"):
            train_pipeline = datamodule.train_pipeline
            if train_pipeline is not None:
                base_feature_names = list(train_pipeline.feature_names_in)
        fit_adata = datamodule._fit_scope_adata(str(space_cfg.get("fit_scope", metric_space.fit_split)))
        pipeline = get_or_build_pipeline(
            fit_adata,
            space_cfg,
            space_path=None,
            base_feature_names=base_feature_names,
        )
        if cache is not None:
            cache[key] = pipeline
    return pipeline, space_cfg, pipeline_label(space_cfg)


def _observed_in_space(
    adata_obj: ad.AnnData,
    pipeline: TransformPipeline,
    space_cfg: dict,
) -> np.ndarray:
    """transform observed adata to one metric space"""

    matrix, _ = transform_adata(
        adata_obj,
        pipeline,
        device="cpu",
        chunk_size=int(space_cfg.get("chunk_size", 2048)),
    )
    return np.asarray(matrix.X, dtype=np.float32)


def _roundtrip_to_space(
    datamodule,
    adata_obj: ad.AnnData,
    pipeline: TransformPipeline,
    sample_decode: bool,
) -> np.ndarray:
    """roundtrip adata through train space then metric space"""

    model_matrix, library_size, _ = datamodule.train_pipeline.transform(adata_obj, device="cpu")
    original_library_size = np.asarray(library_size, dtype=np.float32)
    raw_counts = datamodule.train_pipeline.inverse_to_raw(
        model_matrix,
        library_size=original_library_size,
        sample=sample_decode,
    )
    return pipeline.transform_raw(
        raw_counts,
        original_library_size,
        list(datamodule.train_pipeline.feature_names_in),
        device="cpu",
    )


def _predictions_in_space(
    datamodule,
    predictions: ad.AnnData,
    control_library_size: np.ndarray,
    pipeline: TransformPipeline,
    space_cfg: dict,
    sample_decode: bool,
    comparison_pipeline: TransformPipeline | None,
    comparison_space_cfg: dict | None,
) -> np.ndarray:
    """transform predictions to one metric space

    predictions are expected to be in base space (normalized + hvg no projections)
    for comparison/base metric spaces this is a pass-through
    for metric spaces with projections apply them to the base-space predictions
    """

    pred_matrix = np.asarray(predictions.X, dtype=np.float32)
    current_label = None
    if "_prediction_space" in predictions.obs.columns and predictions.n_obs > 0:
        current_label = str(predictions.obs["_prediction_space"].iloc[0])
    if current_label is None:
        current_label = pipeline_label(datamodule.evaluation_space_cfg)
    target_label = pipeline_label(space_cfg)

    if current_label == target_label:
        return pred_matrix

    # predictions are in base space - if target has projections apply them
    # build base label from the target config with projections stripped
    target_base_cfg = deepcopy(space_cfg)
    target_base_cfg["projections"] = []
    target_base_label = pipeline_label(target_base_cfg)

    if current_label == target_base_label:
        # predictions are in the base space of the target pipeline
        # apply projections from the target pipeline
        if not pipeline.projections:
            return pred_matrix
        matrix = pred_matrix
        for projection in pipeline.projections:
            matrix = projection.transform(matrix, device="cpu")
        return np.asarray(matrix, dtype=np.float32)

    # cross-base-space conversion: predictions in one base space (eg raw_counts)
    # need to be evaluated in another (eg normalized_log1p)
    # roundtrip through raw counts using the train pipeline then re-transform
    # with the target pipeline
    train_pipeline = datamodule.train_pipeline
    if train_pipeline is not None:
        feature_names = list(train_pipeline.feature_names_in)
        raw_matrix = train_pipeline.base_transform.inverse_to_raw(
            pred_matrix, control_library_size,
        )
        converted = pipeline.base_transform.transform_raw(
            raw_matrix, control_library_size, feature_names,
        )
        matrix = np.asarray(converted.matrix, dtype=np.float32)
        for projection in pipeline.projections:
            matrix = projection.transform(matrix, device="cpu")
        return np.asarray(matrix, dtype=np.float32)

    raise ValueError(
        f"predictions are in {current_label!r} but metric space expects "
        f"{target_label!r} (base: {target_base_label!r}) - "
        f"regenerate predictions so they are in base space"
    )


def _metric_space_views_cache_key(
    metric_space_spec: MetricSpaceSpec,
    predictions: ad.AnnData,
    sample_decode: bool,
) -> str:
    """build cache key for metric space views"""

    pred_spec = predictions.uns.get("prediction_metadata", {}).get("prediction_spec", {})
    return (
        f"metric_space_views:{metric_space_spec.name}:{metric_space_spec.kind}:"
        f"sample-{int(sample_decode)}:{predictions.n_obs}:"
        f"{id(predictions)}:"
        f"{json.dumps(pred_spec, sort_keys=True)}"
    )


def prepare_metric_space_views(
    *,
    datamodule,
    raw_adata: ad.AnnData,
    predictions: ad.AnnData,
    metric_space: MetricSpaceSpec | dict | None = None,
    sample_decode: bool = False,
    comparison_pipeline: TransformPipeline | None = None,
    comparison_space_cfg: dict | None = None,
    cache: dict[str, Any] | None = None,
) -> MetricSpaceViews:
    """prepare prediction and reference views in one metric space"""

    metric_space_spec = normalize_metric_space_spec(metric_space)

    if cache is not None:
        views_key = _metric_space_views_cache_key(metric_space_spec, predictions, sample_decode)
        if views_key in cache:
            return cache[views_key]

    pipeline, space_cfg, space_label = _get_metric_pipeline(
        datamodule=datamodule,
        metric_space=metric_space_spec,
        comparison_pipeline=comparison_pipeline,
        comparison_space_cfg=comparison_space_cfg,
        cache=cache,
    )

    obs = predictions.obs.reset_index(drop=True).copy()
    if "_target_obs_name" not in obs.columns or "_control_obs_name" not in obs.columns:
        raise ValueError("prediction obs must include _target_obs_name and _control_obs_name")

    control_adata = _take_obs_rows(raw_adata, obs["_control_obs_name"].astype(str).tolist())
    target_adata = _take_obs_rows(raw_adata, obs["_target_obs_name"].astype(str).tolist())
    control_library_size = _control_library_size(control_adata)

    control_observed = _observed_in_space(control_adata, pipeline, space_cfg)
    reference_observed = _observed_in_space(target_adata, pipeline, space_cfg)
    prediction_observed = _predictions_in_space(
        datamodule=datamodule,
        predictions=predictions,
        control_library_size=control_library_size,
        pipeline=pipeline,
        space_cfg=space_cfg,
        sample_decode=sample_decode,
        comparison_pipeline=comparison_pipeline,
        comparison_space_cfg=comparison_space_cfg,
    )

    pred_adata = ad.AnnData(X=np.asarray(prediction_observed, dtype=np.float32), obs=obs.copy())
    pred_adata.var_names = list(pipeline.feature_names_out())

    ref_adata = ad.AnnData(X=np.asarray(reference_observed, dtype=np.float32), obs=obs.copy())
    ref_adata.var_names = list(pipeline.feature_names_out())

    ctrl_adata = ad.AnnData(X=np.asarray(control_observed, dtype=np.float32), obs=obs.copy())
    ctrl_adata.var_names = list(pipeline.feature_names_out())

    views = MetricSpaceViews(
        prediction_adata=pred_adata,
        reference_adata=ref_adata,
        control_adata=ctrl_adata,
        obs=obs,
        feature_names=tuple(str(name) for name in pipeline.feature_names_out()),
        metric_space_label=str(space_label),
        metric_space_spec=metric_space_spec,
    )

    if cache is not None:
        cache[views_key] = views

    return views


def metric_space_metadata(metric_space: MetricSpaceSpec | dict | None) -> dict:
    """convert metric space spec to metadata"""

    return asdict(normalize_metric_space_spec(metric_space))
