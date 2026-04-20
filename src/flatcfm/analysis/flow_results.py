"""flow results helpers"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import wasserstein_distance

from flatcfm.analysis.benchmarks import compute_w2_squared
from flatcfm.analysis.benchmarks._utils import (
    build_comparison_pipeline,
    control_library_size as _control_library_size,
    take_obs_rows as _take_obs_rows,
)
from flatcfm.analysis.prediction_export import build_prediction_adata
from flatcfm.data.space import pipeline_label, transform_adata
from flatcfm.modelcore.utils import instantiate_datamodule


SUPPORTED_FLOW_TASKS = {
    "fm", "ode", "fisher_flow", "baseline", "baseline_decoder", "baseline_linear",
    "no_effect", "additive", "linear_additive", "latent_additive", "context_mean", "perturb_mean", "decoder_only",
}


@dataclass
class FlowRunBundle:
    """flow run bundle"""

    run_dir: Path
    cfg: DictConfig
    history: dict
    metadata: dict
    checkpoint_path: Path | None
    datamodule: Any
    comparison_pipeline: Any
    comparison_space_cfg: dict
    comparison_space_label: str
    cache: dict[str, Any] = field(default_factory=dict)


def _resolve_run_dir(run_dir: str | Path | None, experiment_name: str | None) -> Path:
    """resolve run dir"""

    if run_dir is not None:
        return Path(run_dir).expanduser().resolve()
    if experiment_name is None:
        raise ValueError("run_dir or experiment_name must be provided")
    run_root = Path("artifacts/runs") / str(experiment_name)
    run_dirs = sorted(path for path in run_root.glob("*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under {run_root}")
    return run_dirs[-1].resolve()



def _prediction_path(bundle: FlowRunBundle, prediction_name: str) -> Path:
    """build prediction path"""

    return bundle.run_dir / "predictions" / str(prediction_name) / "predictions.h5ad"


def _prediction_metadata_path(bundle: FlowRunBundle, prediction_name: str) -> Path:
    """build prediction metadata path"""

    return bundle.run_dir / "predictions" / str(prediction_name) / "prediction_metadata.json"


_build_comparison_pipeline = build_comparison_pipeline


def _roundtrip_to_comparison_space(
    bundle: FlowRunBundle,
    adata_obj: ad.AnnData,
    sample_decode: bool,
) -> np.ndarray:
    """roundtrip adata through train space"""

    model_matrix, library_size, _ = bundle.datamodule.train_pipeline.transform(adata_obj, device="cpu")
    original_library_size = np.asarray(library_size, dtype=np.float32)
    raw_counts = bundle.datamodule.train_pipeline.inverse_to_raw(
        model_matrix,
        library_size=original_library_size,
        sample=sample_decode,
    )
    return bundle.comparison_pipeline.transform_raw(
        raw_counts,
        original_library_size,
        list(bundle.datamodule.train_pipeline.feature_names_in),
        device="cpu",
    )


def _observed_in_comparison_space(bundle: FlowRunBundle, adata_obj: ad.AnnData) -> np.ndarray:
    """transform observed adata to comparison space"""

    matrix, _ = transform_adata(
        adata_obj,
        bundle.comparison_pipeline,
        device="cpu",
        chunk_size=int(bundle.comparison_space_cfg.get("chunk_size", 2048)),
    )
    return np.asarray(matrix.X, dtype=np.float32)


def _predictions_in_comparison_space(
    bundle: FlowRunBundle,
    predictions: ad.AnnData,
    control_library_size: np.ndarray,
    sample_decode: bool,
) -> np.ndarray:
    """transform predictions to comparison space

    predictions are expected to be in base space which equals comparison space
    (both are normalized + hvg no projections) so this is a pass-through
    with a validation check
    """

    del control_library_size, sample_decode  # no longer needed
    pred_matrix = np.asarray(predictions.X, dtype=np.float32)
    current_label = None
    if "_prediction_space" in predictions.obs.columns and predictions.n_obs > 0:
        current_label = str(predictions.obs["_prediction_space"].iloc[0])
    if current_label is None:
        current_label = pipeline_label(bundle.datamodule.evaluation_space_cfg)
    if current_label == bundle.comparison_space_label:
        return pred_matrix

    raise ValueError(
        f"predictions are in {current_label!r} but comparison space is "
        f"{bundle.comparison_space_label!r} - regenerate predictions so they "
        f"are in base space"
    )




def _cached_prediction_spec(bundle: FlowRunBundle, prediction_name: str, pred_adata: ad.AnnData) -> dict | None:
    """load cached prediction spec"""

    metadata_path = _prediction_metadata_path(bundle, prediction_name)
    if metadata_path.exists():
        return json.loads(metadata_path.read_text()).get("prediction_spec")
    return pred_adata.uns.get("prediction_metadata", {}).get("prediction_spec")


def _prediction_cache_key(prediction_name: str, prediction_spec: dict) -> str:
    """build prediction cache key"""

    return f"predictions:{prediction_name}:{json.dumps(prediction_spec, sort_keys=True)}"


def _default_sample_decode(bundle) -> bool:
    """check for NB autoencoder in pipeline projections

    respects explicit predict_cfg.sample_decode if set in run metadata
    then falls back to inspecting pipeline projections for NB family
    """

    metadata = getattr(bundle, "metadata", None) or {}
    predict_cfg = metadata.get("task_metadata", {}).get("predict_cfg", {})
    if "sample_decode" in predict_cfg:
        return bool(predict_cfg["sample_decode"])
    pipelines = []
    datamodule = getattr(bundle, "datamodule", None)
    if datamodule is not None:
        if getattr(datamodule, "train_pipeline", None) is not None:
            pipelines.append(datamodule.train_pipeline)
        if getattr(datamodule, "evaluation_pipeline", None) is not None:
            pipelines.append(datamodule.evaluation_pipeline)
    for pipeline in pipelines:
        for projection in getattr(pipeline, "projections", []):
            ae_model = getattr(projection, "ae_model", None)
            if ae_model is not None and getattr(ae_model, "family", None) == "negative_binomial":
                return True
    return False


def _comparison_frame_key(prediction_name: str, predictions: ad.AnnData, sample_decode: bool) -> str:
    """build comparison cache key"""

    prediction_spec = predictions.uns.get("prediction_metadata", {}).get("prediction_spec", {})
    return (
        f"comparison:{prediction_name}:sample-{int(sample_decode)}:"
        f"{json.dumps(prediction_spec, sort_keys=True)}"
    )


def _build_comparison_frame(
    bundle: FlowRunBundle,
    predictions: ad.AnnData,
    prediction_name: str,
    sample_decode: bool,
) -> dict:
    """build comparison frame"""

    obs = predictions.obs.reset_index(drop=True).copy()
    if "_target_obs_name" not in obs.columns or "_control_obs_name" not in obs.columns:
        raise ValueError("prediction obs must include _target_obs_name and _control_obs_name")

    control_adata = _take_obs_rows(bundle.datamodule.adata_full, obs["_control_obs_name"].astype(str).tolist())
    target_adata = _take_obs_rows(bundle.datamodule.adata_full, obs["_target_obs_name"].astype(str).tolist())
    control_library_size = _control_library_size(control_adata)

    frame = {
        "prediction_name": prediction_name,
        "obs": obs,
        "feature_names": list(bundle.comparison_pipeline.feature_names_out()),
        "control_observed": _observed_in_comparison_space(bundle, control_adata),
        "control_decoded": _roundtrip_to_comparison_space(bundle, control_adata, sample_decode=sample_decode),
        "perturbed_observed": _observed_in_comparison_space(bundle, target_adata),
        "perturbed_ground_truth_decoded": _roundtrip_to_comparison_space(bundle, target_adata, sample_decode=sample_decode),
        "perturbed_predicted_decoded": _predictions_in_comparison_space(
            bundle,
            predictions,
            control_library_size=control_library_size,
            sample_decode=sample_decode,
        ),
        "comparison_space_label": bundle.comparison_space_label,
        "sample_decode": bool(sample_decode),
    }
    return frame


def _get_comparison_frame(
    bundle: FlowRunBundle,
    predictions: ad.AnnData,
    prediction_name: str,
    sample_decode: bool,
) -> dict:
    """get cached comparison frame"""

    key = _comparison_frame_key(prediction_name, predictions, sample_decode)
    if key not in bundle.cache:
        bundle.cache[key] = _build_comparison_frame(bundle, predictions, prediction_name, sample_decode)
    return bundle.cache[key]


def load_flow_run(
    run_dir: str | Path | None = None,
    experiment_name: str | None = None,
) -> FlowRunBundle:
    """load flow run bundle"""

    resolved_run_dir = _resolve_run_dir(run_dir, experiment_name)
    cfg = OmegaConf.load(resolved_run_dir / "run_config.yaml")
    task_name = str(cfg.task.name)
    if task_name not in SUPPORTED_FLOW_TASKS:
        raise ValueError(f"Unsupported flow task for results notebook: {task_name}")

    history = json.loads((resolved_run_dir / "history.json").read_text())
    metadata = json.loads((resolved_run_dir / "run_metadata.json").read_text())
    checkpoint_path = metadata.get("checkpoint_path")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (resolved_run_dir / checkpoint_path).resolve()

    datamodule = instantiate_datamodule(cfg)
    datamodule.setup("predict")

    comparison_space_cfg = deepcopy(datamodule.evaluation_space_cfg)
    comparison_space_cfg["projections"] = []
    comparison_space_cfg["fit_scope"] = "full_dataset"
    comparison_pipeline = _build_comparison_pipeline(datamodule, comparison_space_cfg)
    return FlowRunBundle(
        run_dir=resolved_run_dir,
        cfg=cfg,
        history=history,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        datamodule=datamodule,
        comparison_pipeline=comparison_pipeline,
        comparison_space_cfg=comparison_space_cfg,
        comparison_space_label=pipeline_label(comparison_space_cfg),
    )


def _prediction_bundle(
    bundle: FlowRunBundle,
    prediction_name: str,
    prediction_overrides: dict | None,
) -> FlowRunBundle:
    """build bundle for one prediction spec"""

    if prediction_overrides is None and prediction_name == str(bundle.cfg.predict.get("name", "held_out")):
        return bundle

    override_cfg = {"predict": {"name": str(prediction_name)}}
    if prediction_overrides is not None:
        override_cfg = OmegaConf.merge(override_cfg, {"predict": prediction_overrides})
    cfg = OmegaConf.merge(bundle.cfg, OmegaConf.create(override_cfg))
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup("predict")
    comparison_space_cfg = deepcopy(datamodule.evaluation_space_cfg)
    comparison_space_cfg["projections"] = []
    comparison_space_cfg["fit_scope"] = "full_dataset"
    comparison_pipeline = _build_comparison_pipeline(datamodule, comparison_space_cfg)
    return FlowRunBundle(
        run_dir=bundle.run_dir,
        cfg=cfg,
        history=bundle.history,
        metadata=bundle.metadata,
        checkpoint_path=bundle.checkpoint_path,
        datamodule=datamodule,
        comparison_pipeline=comparison_pipeline,
        comparison_space_cfg=comparison_space_cfg,
        comparison_space_label=pipeline_label(comparison_space_cfg),
    )


def load_flow_predictions(
    bundle: FlowRunBundle,
    prediction_name: str = "held_out",
    prediction_overrides: dict | None = None,
) -> ad.AnnData:
    """load cached predictions from disk

    predictions must be pre-generated via the predict cli:
        PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict predict.run_dir=...

    Args:
        bundle: flow run bundle
        prediction_name: name of prediction set
        prediction_overrides: optional prediction spec overrides
    """

    active_bundle = _prediction_bundle(bundle, prediction_name, prediction_overrides)
    expected_spec = dict(active_bundle.datamodule.predict_spec)
    cache_key = _prediction_cache_key(prediction_name, expected_spec)
    if cache_key in bundle.cache:
        return bundle.cache[cache_key]

    pred_path = _prediction_path(bundle, prediction_name)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions file not found: {pred_path}\n"
            f"  run: PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict "
            f"predict.run_dir={bundle.run_dir}"
        )

    pred_adata = ad.read_h5ad(pred_path)
    cached_spec = _cached_prediction_spec(bundle, prediction_name, pred_adata)
    if cached_spec != expected_spec:
        raise ValueError(
            f"predictions exist but spec mismatch for {bundle.run_dir}\n"
            f"  cached:   {cached_spec}\n"
            f"  expected: {expected_spec}\n"
            f"  regenerate: PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict "
            f"predict.run_dir={bundle.run_dir}"
        )

    bundle.cache[cache_key] = pred_adata
    return pred_adata


def load_predictions_lightweight(
    run_dir: str | Path,
    prediction_name: str = "held_out",
    expected_prediction_spec: dict | None = None,
) -> ad.AnnData:
    """load predictions from disk without instantiating a full datamodule

    this is much cheaper than load_flow_predictions which requires a
    FlowRunBundle (and thus a full datamodule load) just to read the h5ad

    Args:
        run_dir: path to the run directory
        prediction_name: name of prediction set
        expected_prediction_spec: if provided validates on-disk spec matches
    """

    run_dir = Path(run_dir).expanduser().resolve()
    pred_path = run_dir / "predictions" / str(prediction_name) / "predictions.h5ad"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions file not found: {pred_path}\n"
            f"  run: .venv/bin/python -m flatcfm.modelcore.predict "
            f"predict.run_dir={run_dir}"
        )

    # lightweight spec validation from metadata json (no datamodule needed)
    if expected_prediction_spec is not None:
        metadata_path = run_dir / "predictions" / str(prediction_name) / "prediction_metadata.json"
        if metadata_path.exists():
            cached_spec = json.loads(metadata_path.read_text()).get("prediction_spec")
            if cached_spec != expected_prediction_spec:
                raise ValueError(
                    f"predictions exist but spec mismatch for {run_dir}\n"
                    f"  cached:   {cached_spec}\n"
                    f"  expected: {expected_prediction_spec}\n"
                    f"  regenerate: .venv/bin/python -m flatcfm.modelcore.predict "
                    f"predict.run_dir={run_dir}"
                )

    return ad.read_h5ad(pred_path)


def get_or_build_flow_predictions(
    bundle: FlowRunBundle,
    prediction_name: str = "held_out",
    prediction_overrides: dict | None = None,
    **_kwargs,
) -> ad.AnnData:
    """backwards compat alias for load_flow_predictions"""

    return load_flow_predictions(bundle, prediction_name, prediction_overrides)


def get_flow_prediction_bundle(
    bundle: FlowRunBundle,
    prediction_name: str = "held_out",
    prediction_overrides: dict | None = None,
) -> FlowRunBundle:
    """build flow bundle for one prediction spec"""

    return _prediction_bundle(bundle, prediction_name, prediction_overrides)


def select_heldout_perturbation(
    bundle: FlowRunBundle,
    predictions: ad.AnnData,
    mode: str = "worst_w1",
    prediction_name: str = "held_out",
    sample_decode: bool | None = None,
) -> str:
    """select heldout perturbation"""

    from .benchmarking import compute_flow_summary_metrics

    metrics = compute_flow_summary_metrics(
        bundle,
        predictions,
        prediction_name=prediction_name,
        sample_decode=sample_decode,
    )["per_perturbation"]
    if metrics.empty:
        raise ValueError("No heldout perturbations available for this run")
    if mode == "worst_w1":
        return str(metrics.iloc[0]["perturbation"])
    if mode == "first":
        return str(sorted(metrics["perturbation"].tolist())[0])
    raise ValueError(f"Unsupported perturbation selection mode: {mode}")


def build_flow_distribution_views(
    bundle: FlowRunBundle,
    predictions: ad.AnnData,
    perturbation: str,
    prediction_name: str = "held_out",
    sample_decode: bool | None = None,
) -> dict:
    """build decoded distribution views for one perturbation"""

    if sample_decode is None:
        sample_decode = _default_sample_decode(bundle)

    frame = _get_comparison_frame(bundle, predictions, prediction_name, bool(sample_decode))
    perturbation_column = bundle.datamodule.schema.perturbation_source
    mask = frame["obs"][perturbation_column].astype(str).to_numpy() == str(perturbation)
    if not np.any(mask):
        raise ValueError(f"Perturbation {perturbation!r} not found in heldout predictions")

    predicted = np.asarray(frame["perturbed_predicted_decoded"][mask], dtype=np.float32)
    observed = np.asarray(frame["perturbed_observed"][mask], dtype=np.float32)
    gene_w1 = np.array(
        [wasserstein_distance(predicted[:, idx], observed[:, idx]) for idx in range(predicted.shape[1])],
        dtype=np.float64,
    )
    gene_metrics = pd.DataFrame(
        {
            "feature": frame["feature_names"],
            "prediction_vs_observed_w1": gene_w1,
        }
    ).sort_values("prediction_vs_observed_w1", ascending=False).reset_index(drop=True)

    return {
        "perturbation": str(perturbation),
        "feature_names": list(frame["feature_names"]),
        "control_observed": np.asarray(frame["control_observed"][mask], dtype=np.float32),
        "control_decoded": np.asarray(frame["control_decoded"][mask], dtype=np.float32),
        "perturbed_observed": observed,
        "perturbed_ground_truth_decoded": np.asarray(frame["perturbed_ground_truth_decoded"][mask], dtype=np.float32),
        "perturbed_predicted_decoded": predicted,
        "gene_metrics": gene_metrics,
        "comparison_space_label": frame["comparison_space_label"],
        "sample_decode": bool(sample_decode),
        "w2_squared": compute_w2_squared(predicted, observed),
    }
