"""space config and pipeline helpers"""

from __future__ import annotations

import hashlib
import json as _json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import pickle

import anndata as ad
import numpy as np

from .geometry import (
    AELatentProjection,
    BaseTransform,
    CompositionBaseTransform,
    IdentityProjection,
    NonlinearRFFLiftProjection,
    NormalizedLog1pBaseTransform,
    OrthogonalLiftProjection,
    PCAProjection,
    RawCountsBaseTransform,
    TransformPipeline,
)


@dataclass(frozen=True)
class AEProjectionArtifacts:
    """ae projection artifact bundle"""

    projection_path: Path
    metadata_path: Path
    checkpoint_path: Path


def _config_hash(cfg: dict) -> str:
    """deterministic short hash of a config dict for cache keys"""

    canonical = _json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def normalize_space_config(space_cfg: dict | None, default_fit_scope: str = "train") -> dict:
    """normalize space config"""

    cfg = deepcopy(space_cfg or {})
    base_cfg = deepcopy(cfg.get("base", {}))
    projections = [deepcopy(item) for item in cfg.get("projections", [])]
    feature_set = str(base_cfg.get("feature_set", "all_genes"))
    n_hvgs = base_cfg.get("n_hvgs")
    if feature_set == "all_genes":
        n_hvgs = None
    hvg_batch_key = base_cfg.get("hvg_batch_key")
    base_normalized = {
        "kind": str(base_cfg.get("kind", "normalized_log1p")),
        "feature_set": feature_set,
        "n_hvgs": n_hvgs,
        "target_sum": float(base_cfg.get("target_sum", 1e4)),
        "hvg_batch_key": str(hvg_batch_key) if hvg_batch_key is not None else None,
    }
    if feature_set == "hvg_union_deg":
        base_normalized["deg_control_column"] = base_cfg.get("deg_control_column")
        base_normalized["deg_control_value"] = base_cfg.get("deg_control_value")
        base_normalized["deg_perturbation_column"] = base_cfg.get("deg_perturbation_column")
        base_normalized["deg_cell_type_column"] = base_cfg.get("deg_cell_type_column")
        base_normalized["deg_n_top_genes"] = int(base_cfg.get("deg_n_top_genes", 25))
    normalized = {
        "base": base_normalized,
        "projections": [],
        "ae_export_artifact_tag": cfg.get("ae_export_artifact_tag"),
        "fit_scope": str(cfg.get("fit_scope", default_fit_scope)),
        "hvg_fit_scope": str(cfg.get("hvg_fit_scope", cfg.get("fit_scope", default_fit_scope))),
        "chunk_size": int(cfg.get("chunk_size", 2048)),
    }
    for projection in projections:
        kind = str(projection.get("kind"))
        if kind == "pca":
            normalized["projections"].append(
                {
                    "kind": "pca",
                    "n_components": int(projection.get("n_components", 50)),
                }
            )
        elif kind == "orthogonal_lift":
            normalized["projections"].append(
                {
                    "kind": "orthogonal_lift",
                    "ambient_dim": int(projection.get("ambient_dim", 2)),
                    "seed": int(projection.get("seed", 0)),
                }
            )
        elif kind == "nonlinear_rff_lift":
            normalized["projections"].append(
                {
                    "kind": "nonlinear_rff_lift",
                    "ambient_dim": int(projection.get("ambient_dim", 2)),
                    "seed": int(projection.get("seed", 0)),
                    "feature_scale": float(projection.get("feature_scale", 1.0)),
                }
            )
        elif kind == "ae_latent":
            normalized["projections"].append(
                {
                    "kind": "ae_latent",
                    "artifact_tag": projection.get("artifact_tag"),
                }
            )
        elif kind == "identity":
            normalized["projections"].append({"kind": "identity"})
        else:
            raise ValueError(f"Unsupported projection kind: {kind}")
    return normalized


def normalize_evaluation_space_config(space_cfg: dict, evaluation_space_cfg: dict | None) -> dict:
    """normalize evaluation space config"""

    eval_cfg = deepcopy(evaluation_space_cfg or {})
    if not eval_cfg or bool(eval_cfg.get("copy_from_space", False)):
        merged = normalize_space_config(space_cfg, default_fit_scope="full_dataset")
        if "fit_scope" in eval_cfg:
            merged["fit_scope"] = str(eval_cfg["fit_scope"])
        else:
            merged["fit_scope"] = "full_dataset"
        return merged
    return normalize_space_config(eval_cfg, default_fit_scope="full_dataset")


def _base_tag(base_cfg: dict) -> str:
    """build base tag"""

    target_sum = base_cfg["target_sum"]
    target_sum_value = int(target_sum) if float(target_sum).is_integer() else target_sum
    return "_".join(
        [
            str(base_cfg["kind"]).lower(),
            str(base_cfg["feature_set"]).lower(),
            f"nhvg{base_cfg['n_hvgs'] if base_cfg['n_hvgs'] is not None else 'all'}",
            f"tsum{str(target_sum_value).lower()}",
        ]
    )


def _projection_tag(projection_cfg: dict) -> str:
    """build projection tag"""

    kind = projection_cfg["kind"]
    if kind == "identity":
        return "identity"
    if kind == "pca":
        return f"pca_npc{projection_cfg['n_components']}"
    if kind == "orthogonal_lift":
        return f"ortholift_d{projection_cfg['ambient_dim']}_s{projection_cfg['seed']}"
    if kind == "nonlinear_rff_lift":
        feature_scale = projection_cfg["feature_scale"]
        feature_scale_value = int(feature_scale) if float(feature_scale).is_integer() else feature_scale
        return (
            f"nonlinearrff_d{projection_cfg['ambient_dim']}_s{projection_cfg['seed']}"
            f"_fs{str(feature_scale_value).lower()}"
        )
    if kind == "ae_latent":
        tag = projection_cfg.get("artifact_tag") or "auto"
        return f"ae_tag{str(tag).lower()}"
    raise ValueError(f"Unsupported projection kind: {kind}")


def pipeline_tag_from_config(space_cfg: dict | None) -> str:
    """build pipeline tag"""

    cfg = normalize_space_config(space_cfg)
    parts = [_base_tag(cfg["base"])]
    if not cfg["projections"]:
        parts.append("proj_none")
    else:
        parts.extend([_projection_tag(item) for item in cfg["projections"]])
    return "__".join(parts)


def upstream_pipeline_config_for_ae(space_cfg: dict | None) -> dict:
    """build upstream config for ae"""

    cfg = normalize_space_config(space_cfg)
    projections = []
    for projection in cfg["projections"]:
        if projection["kind"] == "ae_latent":
            break
        projections.append(deepcopy(projection))
    return {
        "base": deepcopy(cfg["base"]),
        "projections": projections,
        "fit_scope": str(cfg["fit_scope"]),
        "chunk_size": int(cfg["chunk_size"]),
    }


def upstream_pipeline_tag_for_ae(space_cfg: dict | None) -> str:
    """build upstream tag for ae"""

    return pipeline_tag_from_config(upstream_pipeline_config_for_ae(space_cfg))


def pipeline_label(space_cfg: dict | None) -> str:
    """build pipeline label"""

    cfg = normalize_space_config(space_cfg)
    parts = [cfg["base"]["kind"]]
    parts.extend([item["kind"] for item in cfg["projections"]])
    return " -> ".join(parts)


def _build_base_transform(
    base_cfg: dict,
    base_feature_names: list[str] | None = None,
) -> BaseTransform:
    """build base transform

    when base_feature_names is provided feature selection is skipped
    and the given names are used directly (avoids recomputing DEGs)
    """

    kind = base_cfg["kind"]
    hvg_batch_key = base_cfg.get("hvg_batch_key")
    if kind == "raw_counts":
        return RawCountsBaseTransform(
            feature_set=base_cfg["feature_set"],
            n_hvgs=base_cfg["n_hvgs"],
            target_sum=base_cfg["target_sum"],
            hvg_batch_key=hvg_batch_key,
            deg_control_column=base_cfg.get("deg_control_column"),
            deg_control_value=base_cfg.get("deg_control_value"),
            deg_perturbation_column=base_cfg.get("deg_perturbation_column"),
            deg_cell_type_column=base_cfg.get("deg_cell_type_column"),
            deg_n_top_genes=int(base_cfg.get("deg_n_top_genes", 25)),
            precomputed_feature_names=base_feature_names,
        )
    if kind == "normalized_log1p":
        return NormalizedLog1pBaseTransform(
            feature_set=base_cfg["feature_set"],
            n_hvgs=base_cfg["n_hvgs"],
            target_sum=base_cfg["target_sum"],
            hvg_batch_key=hvg_batch_key,
            deg_control_column=base_cfg.get("deg_control_column"),
            deg_control_value=base_cfg.get("deg_control_value"),
            deg_perturbation_column=base_cfg.get("deg_perturbation_column"),
            deg_cell_type_column=base_cfg.get("deg_cell_type_column"),
            deg_n_top_genes=int(base_cfg.get("deg_n_top_genes", 25)),
            precomputed_feature_names=base_feature_names,
        )
    if kind == "composition":
        return CompositionBaseTransform(
            feature_set=base_cfg["feature_set"],
            n_hvgs=base_cfg["n_hvgs"],
            target_sum=base_cfg["target_sum"],
            hvg_batch_key=hvg_batch_key,
            deg_control_column=base_cfg.get("deg_control_column"),
            deg_control_value=base_cfg.get("deg_control_value"),
            deg_perturbation_column=base_cfg.get("deg_perturbation_column"),
            deg_cell_type_column=base_cfg.get("deg_cell_type_column"),
            deg_n_top_genes=int(base_cfg.get("deg_n_top_genes", 25)),
            precomputed_feature_names=base_feature_names,
        )
    raise ValueError(f"Unsupported base transform kind: {kind}")


def resolve_ae_projection_artifacts(paths_cfg: dict, dataset_name: str, artifact_tag: str) -> AEProjectionArtifacts:
    """resolve ae projection artifacts"""

    space_root = Path(paths_cfg.get("space_dir", "artifacts/spaces"))
    model_root = Path(paths_cfg.get("model_dir", "artifacts/models"))
    safe_tag = artifact_tag.lower().replace("/", "-").replace(" ", "-")
    return AEProjectionArtifacts(
        projection_path=space_root / f"{dataset_name}_ae_projection_{safe_tag}.pkl",
        metadata_path=model_root / f"{dataset_name}_ae_projection_{safe_tag}.json",
        checkpoint_path=model_root / f"{dataset_name}_ae_model_{safe_tag}.ckpt",
    )


def save_pipeline(pipeline: TransformPipeline, path: Path) -> TransformPipeline:
    """save pipeline"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(pipeline, handle)
    return pipeline


def load_pipeline(path: Path) -> TransformPipeline:
    """load pipeline"""

    with path.open("rb") as handle:
        return pickle.load(handle)


def save_ae_projection(projection: AELatentProjection, path: Path) -> AELatentProjection:
    """save ae projection"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(projection, handle)
    return projection


def load_ae_projection(path: Path) -> AELatentProjection:
    """load ae projection"""

    if not path.exists():
        raise FileNotFoundError(f"Missing ae projection artifact: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _ae_projection_is_stale(space_path: Path, space_cfg: dict) -> bool:
    """check if a cached pipeline is stale because its ae projection was retrained

    compares the mtime of ae projection pickles against the cached pipeline
    if any ae projection is newer the cache is stale
    """

    cfg = normalize_space_config(space_cfg)
    projections = cfg.get("projections", [])
    ae_tags = [p.get("artifact_tag", "") for p in projections if p.get("kind") == "ae_latent" and p.get("artifact_tag")]
    if not ae_tags:
        return False
    pipeline_mtime = space_path.stat().st_mtime
    space_dir = space_path.parent
    for tag in ae_tags:
        for ae_pkl in space_dir.glob(f"*ae_projection*{tag}*.pkl"):
            if ae_pkl.stat().st_mtime > pipeline_mtime:
                return True
    return False


def get_or_build_pipeline(
    adata_obj: ad.AnnData,
    space_cfg: dict,
    space_path: Path | None = None,
    ae_projection_resolver=None,
    hvg_adata: ad.AnnData | None = None,
    base_feature_names: list[str] | None = None,
) -> TransformPipeline:
    """load or fit pipeline

    Args:
        adata_obj: adata used for fitting projections (pca etc)
        space_cfg: space config
        space_path: optional cache path
        ae_projection_resolver: optional resolver for ae projections
        hvg_adata: optional separate adata for hvg selection only
            when None uses adata_obj for both hvg selection and projection fitting
        base_feature_names: optional precomputed feature names to skip
            recomputing feature selection (avoids rerunning DEG selection)
    """

    if space_path is not None and space_path.exists():
        if not _ae_projection_is_stale(space_path, space_cfg):
            return load_pipeline(space_path)
        import logging
        logging.getLogger(__name__).info(
            "rebuilding pipeline %s — ae projection artifact is newer than cache", space_path
        )

    cfg = normalize_space_config(space_cfg)

    # reuse cached feature names to avoid expensive DEG recomputation
    if base_feature_names is None and space_path is not None:
        base_hash = _config_hash(cfg["base"])
        feature_cache = space_path.parent / f"feature_names_{base_hash}.json"
        if feature_cache.exists():
            import logging
            base_feature_names = _json.loads(feature_cache.read_text())
            logging.getLogger(__name__).info(
                "reusing %d cached feature names from %s", len(base_feature_names), feature_cache.name,
            )

    base_transform = _build_base_transform(cfg["base"], base_feature_names=base_feature_names)
    projections = []
    for projection_cfg in cfg["projections"]:
        kind = projection_cfg["kind"]
        if kind == "identity":
            projections.append(IdentityProjection())
        elif kind == "pca":
            projections.append(PCAProjection(n_components=projection_cfg["n_components"]))
        elif kind == "orthogonal_lift":
            projections.append(
                OrthogonalLiftProjection(
                    ambient_dim=projection_cfg["ambient_dim"],
                    seed=projection_cfg["seed"],
                )
            )
        elif kind == "nonlinear_rff_lift":
            projections.append(
                NonlinearRFFLiftProjection(
                    ambient_dim=projection_cfg["ambient_dim"],
                    seed=projection_cfg["seed"],
                    feature_scale=projection_cfg["feature_scale"],
                )
            )
        elif kind == "ae_latent":
            if ae_projection_resolver is None:
                raise ValueError("ae projection resolver is required for ae latent projections")
            projections.append(ae_projection_resolver(projection_cfg))
        else:
            raise ValueError(f"Unsupported projection kind: {kind}")
    pipeline = TransformPipeline(
        base_transform=base_transform,
        projections=projections,
        fit_scope=str(cfg["fit_scope"]),
    )
    pipeline.fit(adata_obj, hvg_adata=hvg_adata)
    if space_path is not None:
        save_pipeline(pipeline, space_path)
        # cache feature names so other pipelines with the same base
        # config can skip expensive DEG recomputation
        base_hash = _config_hash(cfg["base"])
        feature_cache = space_path.parent / f"feature_names_{base_hash}.json"
        if not feature_cache.exists():
            feature_cache.write_text(_json.dumps(list(pipeline.feature_names_in)))
    return pipeline


def _chunk_slices(n_rows: int, chunk_size: int) -> list[slice]:
    """make chunk slices"""

    return [slice(start, min(start + chunk_size, n_rows)) for start in range(0, n_rows, chunk_size)]


def transform_adata(
    adata_obj: ad.AnnData,
    pipeline: TransformPipeline,
    device: str = "cpu",
    chunk_size: int = 2048,
) -> tuple[ad.AnnData, dict]:
    """transform adata"""

    outputs = []
    library_size = []
    for chunk in _chunk_slices(adata_obj.n_obs, chunk_size):
        curr = adata_obj[chunk].copy()
        projected, curr_lib, _ = pipeline.transform(curr, device=device)
        outputs.append(projected)
        library_size.append(curr_lib)

    matrix = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)
    transformed = ad.AnnData(X=matrix, obs=adata_obj.obs.copy())
    transformed.var_names = pipeline.feature_names_out()
    metadata = {
        "library_size": np.concatenate(library_size, axis=0) if library_size else np.zeros((0,), dtype=np.float32),
        "feature_names": transformed.var_names.tolist(),
        "base_feature_names": list(pipeline.feature_names_in),
        "space_spec": pipeline.export_spec(),
    }
    return transformed, metadata
