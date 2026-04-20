"""benchmark routing helpers"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
from omegaconf import OmegaConf

from .benchmarks import (
    Evaluation,
    Evaluator,
    MetricSpaceSpec,
    build_deg_cache,
    normalize_metric_space_spec,
    prepare_metric_space_views,
    reduce_group_metrics,
)
from flatcfm.data.space import normalize_space_config, pipeline_label
from flatcfm.modelcore.predictors import PredictorDataBundle, load_predictor_from_config

from .flow_results import _default_sample_decode

_RUNS_ROOT = Path("artifacts/runs")

_TASK_TO_PREDICTOR_KIND: dict[str, str] = {
    "fm": "run_fm",
    "ode": "run_ode",
    "baseline": "run_baseline",
    "baseline_decoder": "run_baseline",
    "baseline_linear": "run_baseline",
    "no_effect": "run_baseline",
    "additive": "run_baseline",
    "linear_additive": "run_baseline",
    "latent_additive": "run_baseline",
    "context_mean": "run_baseline",
    "perturb_mean": "run_baseline",
    "decoder_only": "run_baseline",
    "mean_flow": "run_fm",
    "fisher_flow": "run_fm",
}


def _all_run_dirs(experiment_name: str) -> list[Path]:
    """resolve all run directories for an experiment name sorted by timestamp"""

    run_root = _RUNS_ROOT / str(experiment_name)
    run_dirs = sorted(path for path in run_root.glob("*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"no runs found under {run_root}")
    return [d.resolve() for d in run_dirs]


# keys in run_config.yaml that define training behavior
# everything else (paths, callbacks, logger, trainer hardware) is ignored
_TRAINING_CONFIG_KEYS = ("model", "task", "loss", "space", "splitter", "data", "condition", "seed", "ae_geometry", "ae_schedule")


def _training_config_fingerprint(run_dir: Path) -> str:
    """compute a fingerprint of the training-relevant config for a run

    two runs with the same fingerprint used the same hyperparameters
    """

    import hashlib

    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return ""
    cfg = OmegaConf.load(config_path)
    subset = {}
    for k in _TRAINING_CONFIG_KEYS:
        if k not in cfg:
            subset[k] = None
        elif OmegaConf.is_config(cfg[k]):
            subset[k] = OmegaConf.to_container(cfg[k], resolve=True)
        else:
            subset[k] = cfg[k]
    canonical = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _compatible_run_dirs(experiment_name: str) -> list[Path]:
    """resolve run dirs that share the same training config as the latest run"""

    all_dirs = _all_run_dirs(experiment_name)
    latest_fp = _training_config_fingerprint(all_dirs[-1])
    return [d for d in all_dirs if _training_config_fingerprint(d) == latest_fp]


def _latest_run_dir(experiment_name: str) -> Path:
    """resolve most recent run directory for an experiment name"""

    return _all_run_dirs(experiment_name)[-1]


def resolve_experiment(experiment_name: str) -> dict:
    """build predictor config from an experiment name

    resolves the latest run directory and infers the predictor kind
    from the task name in run_config.yaml

    Args:
        experiment_name: name under artifacts/runs/
    """

    run_dir = _latest_run_dir(experiment_name)
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"no run_config.yaml in {run_dir}")
    cfg = OmegaConf.load(config_path)
    task_name = str(cfg.task.name)
    kind = _TASK_TO_PREDICTOR_KIND.get(task_name)
    if kind is None:
        raise ValueError(
            f"cannot infer predictor kind for task {task_name!r} in {experiment_name} - "
            f"known tasks: {sorted(_TASK_TO_PREDICTOR_KIND)}"
        )
    return {"kind": kind, "name": experiment_name, "run_dir": str(run_dir)}


def _space_signature_from_cfg(space_cfg: dict) -> tuple:
    """full space signature from a space config dict"""

    cfg = normalize_space_config(space_cfg, default_fit_scope="train")
    base = dict(cfg["base"])
    return (
        str(base.get("kind")),
        str(base.get("feature_set")),
        None if base.get("n_hvgs") is None else int(base.get("n_hvgs")),
        float(base.get("target_sum", 1e4)),
        str(base.get("hvg_batch_key")) if base.get("hvg_batch_key") is not None else None,
        str(cfg.get("fit_scope", "train")),
    )


def _base_space_signature(bundle) -> tuple:
    """base space signature for compatibility checks"""

    cfg = normalize_space_config(bundle.datamodule.space_cfg, default_fit_scope="train")
    base = dict(cfg["base"])
    return (
        str(base.get("kind")),
        str(base.get("feature_set")),
        None if base.get("n_hvgs") is None else int(base.get("n_hvgs")),
        float(base.get("target_sum", 1e4)),
        str(base.get("hvg_batch_key")) if base.get("hvg_batch_key") is not None else None,
    )


def validate_run_dirs(run_dirs: dict[str, str | Path] | list[str]) -> None:
    """validate that all run dirs share the same base space config and gene set

    reads run_config.yaml for config signature checks and run_metadata.json
    for actual feature name validation since the same config can produce
    different hvg sets when fit on different data

    Args:
        run_dirs: mapping of name -> run directory path or list of experiment names
    """

    if isinstance(run_dirs, list):
        run_dirs = {name: str(_latest_run_dir(name)) for name in run_dirs}
    signatures: dict[str, tuple] = {}
    feature_name_lists: dict[str, list[str] | None] = {}
    for name, rd in run_dirs.items():
        rd_path = Path(rd)
        config_path = rd_path / "run_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"no run_config.yaml in {rd_path} for {name!r}")
        cfg = OmegaConf.load(config_path)
        space_cfg = OmegaConf.to_container(cfg.space, resolve=True)
        signatures[name] = _space_signature_from_cfg(space_cfg)

        # load actual feature names from run metadata
        metadata_path = rd_path / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            feature_name_lists[name] = metadata.get("feature_names")
        else:
            feature_name_lists[name] = None

    unique_sigs = set(signatures.values())
    if len(unique_sigs) > 1:
        lines = []
        for name, sig in signatures.items():
            kind, fset, nhvg, tsum, batch_key, fit_scope = sig
            lines.append(
                f"  {name}: {kind} {fset} n_hvgs={nhvg} tsum={int(tsum)} "
                f"batch={batch_key} scope={fit_scope}"
            )
        raise ValueError(
            f"incompatible space configs across {len(unique_sigs)} groups:\n"
            + "\n".join(lines)
        )

    # validate actual gene names match across runs
    known_features: dict[str, list[str]] = {
        name: fnames for name, fnames in feature_name_lists.items() if fnames is not None
    }
    if len(known_features) >= 2:
        ref_name, ref_features = next(iter(known_features.items()))
        for name, features in known_features.items():
            if features != ref_features:
                n_ref = len(ref_features)
                n_other = len(features)
                ref_set = set(ref_features)
                other_set = set(features)
                only_ref = ref_set - other_set
                only_other = other_set - ref_set
                raise ValueError(
                    f"feature name mismatch between {ref_name!r} ({n_ref} genes) "
                    f"and {name!r} ({n_other} genes): "
                    f"{len(only_ref)} genes only in {ref_name!r}, "
                    f"{len(only_other)} genes only in {name!r} - "
                    f"runs must share the same hvg gene set for cross-model comparison"
                )


def _ensure_run_predictor_space_compatibility(anchor_bundle, predictor_cfg: dict, load_flow_run_fn) -> None:
    """validate run predictor base space compatibility"""

    kind = str(predictor_cfg.get("kind", "")).lower()
    if kind not in {"run_fm", "fm_run", "run_ode", "ode_run", "run_baseline", "baseline_run"}:
        return
    if predictor_cfg.get("run_dir") is None:
        return

    other_bundle = load_flow_run_fn(run_dir=str(predictor_cfg["run_dir"]))
    anchor_sig = _base_space_signature(anchor_bundle)
    other_sig = _base_space_signature(other_bundle)
    if anchor_sig != other_sig:
        raise ValueError(
            "benchmark across different base spaces is not supported "
            f"anchor={anchor_sig} predictor={other_sig} "
            f"anchor_run={anchor_bundle.run_dir} predictor_run={other_bundle.run_dir}"
        )


def select_benchmark_backend(cfg: dict) -> str:
    """select benchmark backend"""

    return str(cfg.get("evaluation", {}).get("backend", "native"))


def run_benchmark(native_fn=None, perturbench_fn=None, backend: str = "native") -> dict[str, Any]:
    """run benchmark backend"""

    output = {}
    if backend in {"native", "both"} and native_fn is not None:
        output["native"] = native_fn()
    if backend in {"perturbench", "both"} and perturbench_fn is not None:
        output["perturbench"] = perturbench_fn()
    return output


def build_flow_evaluation(
    bundle,
    predictions,
    prediction_name: str = "held_out",
    sample_decode: bool | None = None,
    group_columns: tuple[str, ...] | None = None,
    metric_space: MetricSpaceSpec | dict | None = None,
) -> Evaluation:
    """build evaluation from flow bundle"""

    if sample_decode is None:
        sample_decode = _default_sample_decode(bundle)
    views = prepare_metric_space_views(
        datamodule=bundle.datamodule,
        raw_adata=bundle.datamodule.adata_full,
        predictions=predictions,
        metric_space=metric_space or MetricSpaceSpec(name="comparison", kind="comparison", fit_split="full_dataset"),
        sample_decode=bool(sample_decode),
        comparison_pipeline=bundle.comparison_pipeline,
        comparison_space_cfg=bundle.comparison_space_cfg,
        cache=bundle.cache,
    )
    return Evaluation(
        prediction_adata=views.prediction_adata,
        reference_adata=views.reference_adata,
        control_adata=views.control_adata,
        perturbation_key=bundle.datamodule.schema.perturbation_source,
        covariate_keys=tuple(field.source_column for field in bundle.datamodule.schema.sample_covariates),
        control_value=bundle.datamodule.schema.control_value,
        group_columns=group_columns,
    )


def _metric_space_spec(analysis_space: str, pca_n_components: int) -> MetricSpaceSpec:
    """build metric space spec from short name"""

    if analysis_space == "comparison":
        return MetricSpaceSpec(name="comparison", kind="comparison", fit_split="full_dataset")
    if analysis_space == "train_base":
        return MetricSpaceSpec(name="train_base", kind="train_base", fit_split="train")
    if analysis_space == "train_pca":
        return MetricSpaceSpec(
            name=f"train_pca_{int(pca_n_components)}",
            kind="train_pca",
            pca_n_components=int(pca_n_components),
            fit_split="train",
        )
    raise ValueError(f"Unsupported analysis space: {analysis_space}")


def _pivot_metric_rows(
    metrics_df: pd.DataFrame,
    group_columns: tuple[str, ...],
    metric_names: tuple[str, ...],
) -> pd.DataFrame:
    """pivot tidy metric rows to wide format"""

    if metrics_df.empty:
        return pd.DataFrame(columns=[*group_columns, "n_cells", *metric_names])
    wide = (
        metrics_df.pivot_table(
            index=[*group_columns, "n_ref"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"n_ref": "n_cells"})
    )
    wide.columns.name = None
    for column in metric_names:
        if column not in wide.columns:
            wide[column] = float("nan")
    return wide


def evaluate_flow_predictions(
    bundle,
    predictions: ad.AnnData,
    metric_space: MetricSpaceSpec | dict | None = None,
    metrics: tuple[str, ...] = ("mean_gene_w1", "w2_squared"),
    prediction_name: str = "held_out",
    group_columns: tuple[str, ...] | None = None,
    reductions: tuple[str, ...] | None = ("unweighted_mean", "cell_weighted_mean"),
    sample_decode: bool | None = None,
    aggregation: str | None = None,
    model_name: str | None = None,
    max_samples: int = 2000,
    top_ks: tuple[int | None, ...] = (50, 100, 200),
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
    include_logfc_norms: bool = False,
    deg_cache: dict | None = None,
) -> dict[str, Any]:
    """evaluate flow predictions in one metric space"""

    if sample_decode is None:
        sample_decode = _default_sample_decode(bundle)

    metric_space_spec = (
        metric_space
        if isinstance(metric_space, MetricSpaceSpec)
        else normalize_metric_space_spec(metric_space or {"kind": "comparison"})
    )
    views = prepare_metric_space_views(
        datamodule=bundle.datamodule,
        raw_adata=bundle.datamodule.adata_full,
        predictions=predictions,
        metric_space=metric_space_spec,
        sample_decode=bool(sample_decode),
        comparison_pipeline=bundle.comparison_pipeline,
        comparison_space_cfg=bundle.comparison_space_cfg,
        cache=bundle.cache,
    )
    evaluation = Evaluation(
        prediction_adata=views.prediction_adata,
        reference_adata=views.reference_adata,
        control_adata=views.control_adata,
        perturbation_key=bundle.datamodule.schema.perturbation_source,
        covariate_keys=tuple(field.source_column for field in bundle.datamodule.schema.sample_covariates),
        control_value=bundle.datamodule.schema.control_value,
        model_name=str(model_name or "flatcfm"),
        group_columns=group_columns,
    )
    per_group = evaluation.evaluate(
        metrics=metrics,
        aggregation=aggregation,
        pca_n_components=int(metric_space_spec.pca_n_components),
        top_ks=top_ks,
        max_samples=int(max_samples),
        fdr_alpha=float(fdr_alpha),
        min_cells=int(min_cells),
        deg_cache=deg_cache,
    )
    if not per_group.empty:
        per_group = per_group.assign(
            prediction_name=str(prediction_name),
            metric_space=str(metric_space_spec.name),
            metric_space_label=str(views.metric_space_label),
        )
    if include_logfc_norms and not per_group.empty:
        norm_frame = evaluation.reference_logfc_norms()
        if not norm_frame.empty:
            merge_cols = [c for c in norm_frame.columns if c in per_group.columns and c != "ref_logfc_norm"]
            per_group = per_group.merge(norm_frame, on=merge_cols, how="left")
    if reductions is None:
        summary = pd.DataFrame()
    else:
        summary = reduce_group_metrics(per_group, reductions=reductions)
    return {
        "per_group": per_group,
        "summary": summary,
        "metric_space": metric_space_spec,
        "metric_space_label": views.metric_space_label,
        "sample_decode": bool(sample_decode),
    }


def compute_grouped_flow_metrics(
    bundle,
    predictions: ad.AnnData,
    group_column: str,
    prediction_name: str = "held_out",
    sample_decode: bool | None = None,
    metric_names: tuple[str, ...] = ("mean_gene_w1", "w2_squared"),
    aggregation: str = "none",
    analysis_space: str = "comparison",
    pca_n_components: int = 50,
) -> pd.DataFrame:
    """compute grouped flow metrics"""

    out = evaluate_flow_predictions(
        bundle,
        predictions,
        metric_space=_metric_space_spec(analysis_space, pca_n_components),
        metrics=metric_names,
        prediction_name=prediction_name,
        group_columns=(str(group_column),),
        reductions=None,
        sample_decode=sample_decode,
        aggregation=aggregation,
    )
    grouped = _pivot_metric_rows(out["per_group"], (str(group_column),), metric_names)
    if "mean_gene_w1" in grouped.columns and "w2_squared" in grouped.columns:
        grouped = grouped.sort_values(["mean_gene_w1", "w2_squared"], ascending=[False, False])
    else:
        grouped = grouped.sort_values(group_column)
    return grouped.reset_index(drop=True)


def compute_per_perturbation_space_metrics(
    bundle,
    predictions: ad.AnnData,
    prediction_name: str = "held_out",
    sample_decode: bool | None = None,
    analysis_space: str = "comparison",
    pca_n_components: int = 50,
    metric_names: tuple[str, ...] = ("w2_squared",),
) -> pd.DataFrame:
    """compute per perturbation metrics in one analysis space"""

    perturbation_column = bundle.datamodule.schema.perturbation_source
    group_columns = (str(perturbation_column),)
    out = evaluate_flow_predictions(
        bundle,
        predictions,
        metric_space=_metric_space_spec(analysis_space, pca_n_components),
        metrics=metric_names,
        prediction_name=prediction_name,
        group_columns=group_columns,
        reductions=None,
        sample_decode=sample_decode,
    )
    wide = _pivot_metric_rows(out["per_group"], group_columns, metric_names)
    wide["analysis_space"] = str(out["metric_space_label"])
    return wide.sort_values(str(perturbation_column)).reset_index(drop=True)


def compute_flow_summary_metrics(
    bundle,
    predictions: ad.AnnData,
    prediction_name: str = "held_out",
    sample_decode: bool | None = None,
) -> dict[str, Any]:
    """compute flow summary metrics"""

    if sample_decode is None:
        sample_decode = _default_sample_decode(bundle)

    perturbation_column = bundle.datamodule.schema.perturbation_source
    per_perturbation = compute_grouped_flow_metrics(
        bundle,
        predictions,
        group_column=perturbation_column,
        prediction_name=prediction_name,
        sample_decode=sample_decode,
    ).rename(columns={perturbation_column: "perturbation"})
    best_val = min(bundle.history.get("val_loss", [float("nan")]) or [float("nan")])
    summary = evaluate_flow_predictions(
        bundle,
        predictions,
        metric_space=MetricSpaceSpec(name="comparison", kind="comparison", fit_split="full_dataset"),
        metrics=("w2_squared",),
        prediction_name=prediction_name,
        group_columns=(str(perturbation_column),),
        reductions=("unweighted_mean",),
        sample_decode=sample_decode,
    )["summary"]
    mean_w2 = float(summary.iloc[0]["value"]) if not summary.empty else float("nan")
    return {
        "best_val_loss": float(best_val),
        "mean_w2_squared": mean_w2,
        "per_perturbation": per_perturbation,
        "comparison_space_label": str(bundle.comparison_space_label),
        "sample_decode": bool(sample_decode),
    }


def build_perturbench_style_evaluator(
    reference_adata: ad.AnnData,
    perturbation_key: str,
    covariate_keys: tuple[str, ...] = (),
    control_value: str | None = None,
    group_columns: tuple[str, ...] | None = None,
    control_adata: ad.AnnData | None = None,
    reference_filters: dict[str, Any] | None = None,
) -> Evaluator:
    """build perturbench style evaluator"""

    return Evaluator(
        reference_adata=reference_adata,
        perturbation_key=perturbation_key,
        covariate_keys=tuple(covariate_keys),
        control_value=control_value,
        group_columns=group_columns,
        control_adata=control_adata,
        reference_filters=reference_filters,
    )


def compute_benchmark_group_metrics(
    bundle,
    predictions,
    group_column: str = "product_dose",
    prediction_name: str = "held_out",
    sample_decode: bool | None = None,
) -> pd.DataFrame:
    """compute benchmark group metrics"""

    out = evaluate_flow_predictions(
        bundle,
        predictions,
        metric_space=MetricSpaceSpec(name="comparison", kind="comparison", fit_split="full_dataset"),
        prediction_name=prediction_name,
        group_columns=(str(group_column),),
        reductions=None,
        sample_decode=sample_decode,
        metrics=("mean_gene_w1", "w2_squared"),
        aggregation="none",
    )
    metrics = out["per_group"]
    if metrics.empty:
        return pd.DataFrame(columns=[group_column, "n_cells", "mean_gene_w1", "w2_squared"])
    wide = (
        metrics.pivot_table(
            index=[group_column, "n_pred"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"n_pred": "n_cells"})
    )
    wide.columns.name = None
    for column in ["mean_gene_w1", "w2_squared"]:
        if column not in wide.columns:
            wide[column] = float("nan")
    return wide.sort_values(["mean_gene_w1", "w2_squared"], ascending=[False, False]).reset_index(drop=True)


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    """benchmark suite spec"""

    anchor_run_dir: str
    predictors: tuple[dict, ...]
    prediction_name: str = "held_out"
    prediction_spec: dict | None = None
    metric_spaces: tuple[dict, ...] = ({"name": "comparison", "kind": "comparison", "fit_split": "full_dataset"},)
    metrics: tuple[str, ...] = ("mean_gene_w1", "w2_squared")
    group_columns: tuple[str, ...] | None = None
    reductions: tuple[str, ...] = ("unweighted_mean", "cell_weighted_mean")
    aggregation: str | None = None
    all_runs: bool = False


def _resolve_experiments_all_runs(experiments: list[str]) -> list[dict]:
    """resolve experiments expanding to all compatible runs per experiment

    only includes runs whose training config matches the latest run
    experiments with a single compatible run get no suffix
    experiments with multiple compatible runs get ::run_0 through ::run_N suffixes
    """

    predictors = []
    for name in experiments:
        run_dirs = _compatible_run_dirs(name)
        if len(run_dirs) == 1:
            predictors.append(resolve_experiment(name))
        else:
            base = resolve_experiment(name)
            kind = base["kind"]
            for i, run_dir in enumerate(run_dirs):
                predictors.append({
                    "kind": kind,
                    "name": f"{name}::run_{i}",
                    "run_dir": str(run_dir),
                })
    return predictors


def _to_suite_spec(spec: dict | BenchmarkSuiteSpec) -> BenchmarkSuiteSpec:
    """normalize benchmark suite spec

    accepts either explicit predictors list or an experiments list
    of experiment names that get auto-resolved via resolve_experiment
    """

    if isinstance(spec, BenchmarkSuiteSpec):
        return spec
    if not isinstance(spec, dict):
        raise TypeError("run_benchmark_suite spec must be dict or BenchmarkSuiteSpec")

    all_runs = bool(spec.get("all_runs", False))

    # resolve experiments list to predictors if provided
    if "experiments" in spec and "predictors" not in spec:
        experiments = [str(name) for name in spec["experiments"]]
        if all_runs:
            predictors = _resolve_experiments_all_runs(experiments)
        else:
            predictors = [resolve_experiment(name) for name in experiments]
        anchor_run_dir = spec.get("anchor_run_dir") or predictors[0]["run_dir"]
    elif "predictors" in spec:
        predictors = [dict(item) for item in spec["predictors"]]
        anchor_run_dir = spec.get("anchor_run_dir") or spec.get("run_dir")
    else:
        raise KeyError("run_benchmark_suite spec must include experiments or predictors")

    if anchor_run_dir is None:
        raise KeyError("run_benchmark_suite spec must include anchor_run_dir")
    return BenchmarkSuiteSpec(
        anchor_run_dir=str(anchor_run_dir),
        predictors=tuple(predictors),
        prediction_name=str(spec.get("prediction_name", "held_out")),
        prediction_spec=dict(spec["prediction_spec"]) if spec.get("prediction_spec") is not None else None,
        metric_spaces=tuple(dict(item) for item in spec.get("metric_spaces", [{"name": "comparison", "kind": "comparison"}])),
        metrics=tuple(str(item) for item in spec.get("metrics", ("mean_gene_w1", "w2_squared"))),
        group_columns=tuple(str(item) for item in spec["group_columns"]) if spec.get("group_columns") is not None else None,
        reductions=tuple(str(item) for item in spec.get("reductions", ("unweighted_mean", "cell_weighted_mean"))),
        aggregation=None if spec.get("aggregation") is None else str(spec.get("aggregation")),
        all_runs=all_runs,
    )


def run_benchmark_suite(spec: dict | BenchmarkSuiteSpec) -> dict[str, Any]:
    """run benchmark suite for predictors

    all predictions must be pre-generated via the predict cli

    loads the anchor bundle once and uses lightweight prediction loading
    for all other predictors to avoid loading N copies of the full dataset
    """

    from .flow_results import load_flow_run, load_predictions_lightweight

    suite = _to_suite_spec(spec)

    # validate all run dirs share the same base space config (lightweight check)
    run_dirs = {}
    for predictor_cfg in suite.predictors:
        run_dir = predictor_cfg.get("run_dir")
        if run_dir is not None:
            run_dirs[predictor_cfg.get("name", run_dir)] = run_dir
    if run_dirs:
        validate_run_dirs(run_dirs)

    anchor_bundle = load_flow_run(run_dir=suite.anchor_run_dir)
    train_bundle = PredictorDataBundle(
        flow_bundle=anchor_bundle,
        prediction_name="train",
        prediction_overrides={"split": "train", "target_subset": "perturbed"},
    )

    # build expected prediction spec from anchor for validation
    # this is what the old per-predictor load_flow_predictions validated against
    expected_prediction_spec: dict | None = None
    if suite.prediction_spec is None:
        # default prediction spec from anchor datamodule
        expected_prediction_spec = dict(anchor_bundle.datamodule.predict_spec)
    # when prediction_spec overrides are provided we skip the lightweight check
    # because the expected spec depends on merging overrides with each predictor's
    # config which would require loading that config's datamodule

    # precompute deg cache once for all models when deg metrics are requested
    _deg_metrics = {"top_k_recall", "deg_jaccard", "deg_overlap_at_k", "sig_deg_recall", "sig_deg_intersect_count"}
    needs_deg = bool(_deg_metrics & set(suite.metrics))
    deg_caches: dict[str, dict] = {}
    if needs_deg:
        # build a temporary evaluation from the first predictor's predictions
        # to get the reference/control structure; the cache only depends on
        # reference and control data which is shared across all models
        first_pred = None
        for predictor_cfg in suite.predictors:
            run_dir = predictor_cfg.get("run_dir")
            if run_dir is not None:
                first_pred = load_predictions_lightweight(
                    run_dir=run_dir,
                    prediction_name=suite.prediction_name,
                    expected_prediction_spec=expected_prediction_spec,
                )
                break
        if first_pred is not None:
            for metric_space in suite.metric_spaces:
                ms_spec = normalize_metric_space_spec(metric_space)
                views = prepare_metric_space_views(
                    datamodule=anchor_bundle.datamodule,
                    raw_adata=anchor_bundle.datamodule.adata_full,
                    predictions=first_pred,
                    metric_space=ms_spec,
                    sample_decode=bool(_default_sample_decode(anchor_bundle)),
                    comparison_pipeline=anchor_bundle.comparison_pipeline,
                    comparison_space_cfg=anchor_bundle.comparison_space_cfg,
                    cache=anchor_bundle.cache,
                )
                eval_for_cache = Evaluation(
                    prediction_adata=views.prediction_adata,
                    reference_adata=views.reference_adata,
                    control_adata=views.control_adata,
                    perturbation_key=anchor_bundle.datamodule.schema.perturbation_source,
                    covariate_keys=tuple(
                        field.source_column for field in anchor_bundle.datamodule.schema.sample_covariates
                    ),
                    control_value=anchor_bundle.datamodule.schema.control_value,
                    group_columns=suite.group_columns,
                )
                deg_caches[ms_spec.name] = build_deg_cache(eval_for_cache)

    predictions: dict[str, ad.AnnData] = {}
    per_group_frames = []
    summary_frames = []

    for predictor_cfg in suite.predictors:
        name = str(predictor_cfg.get("name", predictor_cfg.get("kind", "unknown")))
        run_dir = predictor_cfg.get("run_dir")

        # load predictions directly from disk instead of instantiating
        # a full datamodule per predictor
        if run_dir is not None:
            pred_adata = load_predictions_lightweight(
                run_dir=run_dir,
                prediction_name=suite.prediction_name,
                expected_prediction_spec=expected_prediction_spec,
            )
        else:
            # fallback to full predictor path for non-run predictors
            prediction_bundle = PredictorDataBundle(
                flow_bundle=anchor_bundle,
                prediction_name=suite.prediction_name,
                prediction_overrides=suite.prediction_spec,
            )
            predictor = load_predictor_from_config(predictor_cfg)
            if predictor.requires_training:
                predictor.fit(train_bundle)
            pred_adata = predictor.predict(prediction_bundle)
            name = predictor.name

        predictions[name] = pred_adata
        for metric_space in suite.metric_spaces:
            ms_spec = normalize_metric_space_spec(metric_space)
            out = evaluate_flow_predictions(
                anchor_bundle,
                pred_adata,
                metric_space=metric_space,
                metrics=suite.metrics,
                prediction_name=suite.prediction_name,
                group_columns=suite.group_columns,
                reductions=suite.reductions,
                aggregation=suite.aggregation,
                model_name=name,
                deg_cache=deg_caches.get(ms_spec.name),
            )
            if not out["per_group"].empty:
                per_group_frames.append(out["per_group"])
            if not out["summary"].empty:
                summary_frames.append(out["summary"])

    per_group_metrics = pd.concat(per_group_frames, ignore_index=True) if per_group_frames else pd.DataFrame()
    summary_metrics = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    out = {
        "per_group_metrics": per_group_metrics,
        "summary_metrics": summary_metrics,
        "predictions": predictions,
        "anchor_run_dir": str(anchor_bundle.run_dir),
    }
    if suite.all_runs and not summary_metrics.empty:
        out["multi_run_summary"] = aggregate_over_runs(summary_metrics)
    return out


# -- multi-run aggregation -----------------------------------------------------

_RUN_SUFFIX_RE = re.compile(r"::run_\d+$")


def aggregate_over_runs(df: pd.DataFrame) -> pd.DataFrame:
    """aggregate multi-run benchmark metrics to mean and std per model

    model names with ::run_N suffixes are grouped by base name
    models without suffixes pass through with n_runs=1 and std=NaN
    """

    df = df.copy()
    df["model_name"] = df["model_name"].str.replace(_RUN_SUFFIX_RE, "", regex=True)
    group_cols = [c for c in df.columns if c not in ("value", "n_groups")]
    agg = df.groupby(group_cols, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"count": "n_runs"})
    return agg


def format_mean_std_table(
    agg_df: pd.DataFrame,
    metrics: list[str],
    metric_lower_is_better: dict[str, bool],
) -> pd.DataFrame:
    """pivot aggregated multi-run metrics into a display table with mean +/- std

    returns a styled dataframe ready for display()
    """

    import numpy as np

    # build mean +/- std strings per (model, reduction, metric)
    rows = []
    for _, row in agg_df.iterrows():
        metric = row["metric"]
        if metric not in metrics:
            continue
        mean_val = row["mean"]
        std_val = row["std"]
        n_runs = row["n_runs"]
        if n_runs > 1 and not np.isnan(std_val):
            text = f"{mean_val:.4f} \u00b1 {std_val:.4f}"
        else:
            text = f"{mean_val:.4f}"
        rows.append({
            "model_name": row["model_name"],
            "metric_space": row.get("metric_space", ""),
            "reduction": row.get("reduction", ""),
            "metric": metric,
            "display": text,
            "mean": mean_val,
            "n_runs": n_runs,
        })

    if not rows:
        return pd.DataFrame()

    display_df = pd.DataFrame(rows)
    pivot = display_df.pivot_table(
        index=["model_name", "metric_space", "reduction"],
        columns="metric",
        values="display",
        aggfunc="first",
    )
    pivot.columns.name = None

    # also build numeric mean pivot for bolding best
    mean_pivot = display_df.pivot_table(
        index=["model_name", "metric_space", "reduction"],
        columns="metric",
        values="mean",
        aggfunc="first",
    )

    def _highlight_best(col):
        metric = col.name
        lower = metric_lower_is_better.get(metric, True)
        numeric = mean_pivot[metric]
        if lower:
            best = numeric.min()
        else:
            best = numeric.max()
        return ["font-weight: bold" if np.isclose(v, best) else "" for v in numeric]

    # reorder columns to match input metric order
    ordered_cols = [m for m in metrics if m in pivot.columns]
    pivot = pivot[ordered_cols]

    styled = pivot.reset_index().style.apply(
        lambda col: _highlight_best(col) if col.name in ordered_cols else [""] * len(col),
        axis=0,
    ).set_caption("multi-run mean \u00b1 std (bold = best)")
    return styled
