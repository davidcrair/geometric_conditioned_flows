"""collaborator classes extracted from BasePerturbationDataModule"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import anndata as ad
import numpy as np
import pandas as pd
import torch

from flatcfm.data.dataset import slice_condition_batch
from flatcfm.data.space import AELatentProjection, TransformPipeline

from .schema import ConditionSchema
from .space import (
    get_or_build_pipeline,
    load_ae_projection,
    normalize_evaluation_space_config,
    normalize_space_config,
    pipeline_tag_from_config,
    resolve_ae_projection_artifacts,
    transform_adata,
    upstream_pipeline_tag_for_ae,
)


def _merge_dict(base: dict, override: dict) -> dict:
    """merge nested dicts"""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


class PipelineManager:
    """manages train and evaluation pipeline lifecycle"""

    def __init__(
        self,
        space_cfg: dict,
        evaluation_space_cfg: dict,
        paths_cfg: dict,
        data_cfg: dict,
        artifact_tag: str,
    ):
        self.space_cfg = space_cfg
        self.evaluation_space_cfg = evaluation_space_cfg
        self.paths_cfg = paths_cfg
        self.data_cfg = data_cfg
        self._artifact_tag = artifact_tag

    def artifact_prefix(self) -> str:
        """build artifact prefix"""

        data_name = str(self.data_cfg.get("name", "dataset"))
        tag = self._artifact_tag
        if tag == data_name:
            return data_name
        return f"{data_name}_{tag}"

    def update_artifact_tag(self, tag: str) -> None:
        """update artifact tag"""

        self._artifact_tag = tag

    def resolve_pipeline_path(self, space_cfg: dict) -> Path:
        """resolve pipeline cache path"""

        cfg = normalize_space_config(space_cfg, default_fit_scope="train")
        pipeline_tag = pipeline_tag_from_config(cfg)
        fit_scope = str(cfg.get("fit_scope", "train"))
        hvg_fit_scope = str(cfg.get("hvg_fit_scope", fit_scope))
        space_root = Path(self.paths_cfg.get("space_dir", "artifacts/spaces"))
        suffix = f"_hvgscope-{hvg_fit_scope}" if hvg_fit_scope != fit_scope else ""
        return space_root / f"{self.artifact_prefix()}_pipeline_{pipeline_tag}_fitscope-{fit_scope}{suffix}.pkl"

    def ae_export_artifact_tag(self) -> str:
        """resolve ae export artifact tag"""

        explicit_tag = self.space_cfg.get("ae_export_artifact_tag")
        if explicit_tag is not None and str(explicit_tag).strip():
            return str(explicit_tag)
        return pipeline_tag_from_config(self.space_cfg)

    def resolve_ae_projection_bundle(self, artifact_tag: str) -> tuple[str, Path, Path, Path]:
        """resolve ae projection bundle"""

        bundle_tag = f"{self.artifact_prefix()}_{artifact_tag}"
        bundle = resolve_ae_projection_artifacts(self.paths_cfg, str(self.data_cfg.get("name", "dataset")), bundle_tag)
        return bundle_tag, bundle.projection_path, bundle.metadata_path, bundle.checkpoint_path

    def build_ae_projection(self, projection_cfg: dict) -> AELatentProjection:
        """build ae projection"""

        artifact_tag = projection_cfg.get("artifact_tag") or upstream_pipeline_tag_for_ae(self.space_cfg)
        _, projection_path, _, _ = self.resolve_ae_projection_bundle(str(artifact_tag))
        return load_ae_projection(projection_path)

    def prepare_pipelines(
        self,
        fit_scope_adata_fn: Callable[[str], ad.AnnData],
    ) -> tuple[TransformPipeline, TransformPipeline]:
        """prepare train and evaluation pipelines

        Args:
            fit_scope_adata_fn: callable that returns adata for a given fit scope
        """

        train_cfg = normalize_space_config(self.space_cfg, default_fit_scope="train")
        eval_cfg = normalize_space_config(self.evaluation_space_cfg, default_fit_scope="full_dataset")

        train_fit_scope = str(train_cfg.get("fit_scope", "train"))
        train_hvg_scope = str(train_cfg.get("hvg_fit_scope", train_fit_scope))
        train_hvg_adata = fit_scope_adata_fn(train_hvg_scope) if train_hvg_scope != train_fit_scope else None

        eval_fit_scope = str(eval_cfg.get("fit_scope", "full_dataset"))
        eval_hvg_scope = str(eval_cfg.get("hvg_fit_scope", eval_fit_scope))
        eval_hvg_adata = fit_scope_adata_fn(eval_hvg_scope) if eval_hvg_scope != eval_fit_scope else None

        train_pipeline = get_or_build_pipeline(
            fit_scope_adata_fn(train_fit_scope),
            train_cfg,
            space_path=self.resolve_pipeline_path(train_cfg),
            ae_projection_resolver=self.build_ae_projection,
            hvg_adata=train_hvg_adata,
        )
        evaluation_pipeline = get_or_build_pipeline(
            fit_scope_adata_fn(eval_fit_scope),
            eval_cfg,
            space_path=self.resolve_pipeline_path(eval_cfg),
            ae_projection_resolver=self.build_ae_projection,
            hvg_adata=eval_hvg_adata,
        )
        return train_pipeline, evaluation_pipeline


class ConditionEncoder:
    """builds vocab maps and condition batches from schema"""

    def __init__(self, schema: ConditionSchema):
        self.schema = schema

    def build_vocab_maps(self, obs: pd.DataFrame) -> tuple[dict, dict]:
        """build vocab maps and covariate dicts from obs dataframe

        Returns:
            tuple of (vocab_maps, covariate_dicts)
        """

        maps = {
            "perturbations": {
                value: idx for idx, value in enumerate(sorted(obs[self.schema.perturbation_source].astype(str).unique()))
            },
            "perturbation_covariates": {},
            "sample_covariates": {},
        }
        for field in self.schema.perturbation_covariates:
            maps["perturbation_covariates"][field.name] = {
                value: idx for idx, value in enumerate(sorted(obs[field.source_column].astype(str).unique()))
            }
        for field in self.schema.sample_covariates:
            maps["sample_covariates"][field.name] = {
                value: idx for idx, value in enumerate(sorted(obs[field.source_column].astype(str).unique()))
            }
        covariate_dicts = {
            "perturbation_num_categories": len(maps["perturbations"]),
            "perturbation_covariates": {
                key: len(value) for key, value in maps["perturbation_covariates"].items()
            },
            "sample_covariates": {
                key: len(value) for key, value in maps["sample_covariates"].items()
            },
        }
        return maps, covariate_dicts

    def build_condition_batch(self, adata_obj: ad.AnnData, vocab_maps: dict) -> dict:
        """build condition batch tensors for adata"""

        obs = adata_obj.obs
        perturbations = torch.tensor(
            [vocab_maps["perturbations"][str(value)] for value in obs[self.schema.perturbation_source].astype(str)],
            dtype=torch.long,
        )
        perturbation_covariates = {}
        for field in self.schema.perturbation_covariates:
            mapping = vocab_maps["perturbation_covariates"][field.name]
            perturbation_covariates[field.name] = torch.tensor(
                [mapping[str(value)] for value in obs[field.source_column].astype(str)],
                dtype=torch.long,
            )
        sample_covariates = {}
        for field in self.schema.sample_covariates:
            mapping = vocab_maps["sample_covariates"][field.name]
            sample_covariates[field.name] = torch.tensor(
                [mapping[str(value)] for value in obs[field.source_column].astype(str)],
                dtype=torch.long,
            )
        return {
            "perturbations": perturbations,
            "perturbation_covariates": perturbation_covariates,
            "sample_covariates": sample_covariates,
        }


class PredictionBuilder:
    """resolves prediction specs and builds prediction datasets"""

    def __init__(
        self,
        schema: ConditionSchema,
        predict_cfg: dict,
        splitter_cfg: dict,
        condition_encoder: ConditionEncoder,
    ):
        self.schema = schema
        self.predict_cfg = predict_cfg
        self.splitter_cfg = splitter_cfg
        self.condition_encoder = condition_encoder

    def resolve_prediction_spec(self) -> dict:
        """resolve prediction spec"""

        default_spec = {
            "name": str(self.predict_cfg.get("name", "held_out")),
            "split": "held_out",
            "target_subset": "perturbed",
            "target_filters": {
                "obs_equals": {},
                "obs_in": {},
            },
            "control_source": {
                "split": "all_controls",
                "match_cell_types_from": "test_cell_type",
            },
        }
        override_spec = {
            key: value
            for key, value in self.predict_cfg.items()
            if key in {"name", "split", "target_subset", "target_filters", "control_source"}
        }
        spec = _merge_dict(default_spec, override_spec)
        spec["name"] = str(spec.get("name", "held_out"))
        spec["split"] = str(spec.get("split", "held_out"))
        spec["target_subset"] = str(spec.get("target_subset", "perturbed"))
        control_source = dict(spec.get("control_source", {}))
        control_source["split"] = str(control_source.get("split", "all_controls"))
        control_source["match_cell_types_from"] = str(control_source.get("match_cell_types_from", "test_cell_type"))
        spec["control_source"] = control_source
        target_filters = dict(spec.get("target_filters", {}))
        target_filters["obs_equals"] = dict(target_filters.get("obs_equals", {}))
        target_filters["obs_in"] = {
            key: list(value) if isinstance(value, (list, tuple, set)) else [value]
            for key, value in dict(target_filters.get("obs_in", {})).items()
        }
        spec["target_filters"] = target_filters

        if spec["split"] not in {"held_out", "train", "val"}:
            raise ValueError(f"Unsupported predict.split: {spec['split']}")
        if spec["target_subset"] not in {"perturbed", "control", "all"}:
            raise ValueError(f"Unsupported predict.target_subset: {spec['target_subset']}")
        if control_source["split"] not in {"all_controls", "train_controls", "val_controls"}:
            raise ValueError(f"Unsupported predict.control_source.split: {control_source['split']}")
        if control_source["match_cell_types_from"] not in {"none", "target", "held_out", "test_cell_type"}:
            raise ValueError(
                "Unsupported predict.control_source.match_cell_types_from: "
                f"{control_source['match_cell_types_from']}"
            )
        return spec

    def control_mask(self, adata_full: ad.AnnData) -> np.ndarray:
        """build control mask"""

        return (adata_full.obs[self.schema.control_column].to_numpy() == self.schema.control_value).astype(bool)

    def cell_type_column(self, adata_full: ad.AnnData) -> str | None:
        """resolve cell type column"""

        if "cell_type" in adata_full.obs.columns:
            return "cell_type"
        for field in self.schema.sample_covariates:
            if field.name == "cell_type" or field.source_column == "cell_type":
                if field.source_column in adata_full.obs.columns:
                    return field.source_column
        return None

    def held_out_cell_types(
        self,
        adata_full: ad.AnnData,
        prediction_split_mask_fn: Callable[[str], np.ndarray],
    ) -> list[str]:
        """resolve held out cell types"""

        column = self.cell_type_column(adata_full)
        if column is None:
            return []
        held_out_mask = prediction_split_mask_fn("held_out")
        ctrl_mask = self.control_mask(adata_full)
        values = adata_full.obs.loc[held_out_mask & ~ctrl_mask, column].astype(str).unique().tolist()
        if not values:
            values = adata_full.obs.loc[held_out_mask, column].astype(str).unique().tolist()
        return sorted(values)

    def apply_target_filters(self, adata_obj: ad.AnnData, target_filters: dict) -> ad.AnnData:
        """apply target filters"""

        if adata_obj.n_obs == 0:
            return adata_obj.copy()

        mask = np.ones(adata_obj.n_obs, dtype=bool)
        for column, value in dict(target_filters.get("obs_equals", {})).items():
            if column not in adata_obj.obs.columns:
                raise KeyError(f"Missing target filter column: {column}")
            mask &= adata_obj.obs[column].astype(str).to_numpy() == str(value)
        for column, values in dict(target_filters.get("obs_in", {})).items():
            if column not in adata_obj.obs.columns:
                raise KeyError(f"Missing target filter column: {column}")
            allowed = {str(value) for value in values}
            mask &= adata_obj.obs[column].astype(str).isin(allowed).to_numpy()
        return adata_obj[mask].copy()

    def select_target_adata(
        self,
        spec: dict,
        adata_full: ad.AnnData,
        prediction_split_mask_fn: Callable[[str], np.ndarray],
    ) -> ad.AnnData:
        """select target adata"""

        mask = np.asarray(prediction_split_mask_fn(spec["split"]), dtype=bool)
        ctrl_mask = self.control_mask(adata_full)
        if spec["target_subset"] == "perturbed":
            mask &= ~ctrl_mask
        elif spec["target_subset"] == "control":
            mask &= ctrl_mask
        target_adata = adata_full[mask].copy()
        return self.apply_target_filters(target_adata, spec["target_filters"])

    def select_control_adata(
        self,
        spec: dict,
        target_adata: ad.AnnData,
        adata_full: ad.AnnData,
        prediction_split_mask_fn: Callable[[str], np.ndarray],
    ) -> ad.AnnData:
        """select control adata"""

        ctrl_mask = self.control_mask(adata_full)
        split_mode = spec["control_source"]["split"]
        if split_mode == "all_controls":
            mask = ctrl_mask
        elif split_mode == "train_controls":
            mask = ctrl_mask & np.asarray(prediction_split_mask_fn("train"), dtype=bool)
        elif split_mode == "val_controls":
            mask = ctrl_mask & np.asarray(prediction_split_mask_fn("val"), dtype=bool)
        else:
            raise ValueError(f"Unsupported control split: {split_mode}")

        match_mode = spec["control_source"]["match_cell_types_from"]
        cell_type_col = self.cell_type_column(adata_full)
        allowed_cell_types: list[str] = []
        if cell_type_col is not None and match_mode != "none":
            if match_mode == "target":
                allowed_cell_types = sorted(target_adata.obs[cell_type_col].astype(str).unique().tolist())
            elif match_mode == "held_out":
                allowed_cell_types = self.held_out_cell_types(adata_full, prediction_split_mask_fn)
            elif match_mode == "test_cell_type":
                test_cell_type = self.splitter_cfg.get("test_cell_type")
                if test_cell_type is not None:
                    allowed_cell_types = [str(test_cell_type)]
            else:
                raise ValueError(f"Unsupported match cell type mode: {match_mode}")
        if allowed_cell_types:
            mask &= adata_full.obs[cell_type_col].astype(str).isin(allowed_cell_types).to_numpy()
        return adata_full[mask].copy()

    def match_controls(self, control_obs: pd.DataFrame, target_obs: pd.DataFrame) -> list[int]:
        """match controls to targets"""

        def group_key(row: pd.Series) -> tuple[str, ...]:
            """build control matching key"""

            if not self.schema.sample_covariates:
                return ()
            return tuple(str(row[field.source_column]) for field in self.schema.sample_covariates)

        fallback = np.arange(len(control_obs), dtype=int)
        grouped_controls: dict[tuple[str, ...], np.ndarray] = {}
        for idx, (_, row) in enumerate(control_obs.iterrows()):
            key = group_key(row)
            grouped_controls.setdefault(key, []).append(idx)

        grouped_controls = {
            key: np.asarray(indices, dtype=int)
            for key, indices in grouped_controls.items()
        }
        group_offsets: dict[tuple[str, ...], int] = {}
        chosen_idx = []
        for _, row in target_obs.iterrows():
            key = group_key(row)
            candidates = grouped_controls.get(key, fallback)
            offset = group_offsets.get(key, 0)
            chosen_idx.append(int(candidates[offset % len(candidates)]))
            group_offsets[key] = offset + 1
        return chosen_idx

    def build_prediction_dataset(
        self,
        target_adata: ad.AnnData,
        control_adata: ad.AnnData,
        control_model_adata: ad.AnnData,
        control_metadata: dict,
        vocab_maps: dict,
    ) -> "PredictionDataset":
        """build prediction dataset"""

        from .datamodules import PredictionDataset

        target_batch = self.condition_encoder.build_condition_batch(target_adata, vocab_maps)
        control_obs = control_adata.obs.reset_index(names="_control_obs_name")
        chosen_control_idx = self.match_controls(control_obs, target_adata.obs)
        x_ctrl = torch.as_tensor(np.asarray(control_model_adata.X)[chosen_control_idx], dtype=torch.float32)
        return PredictionDataset(
            x_ctrl=x_ctrl,
            condition_batch=target_batch,
            obs=target_adata.obs.copy(),
            control_obs_names=control_obs.iloc[chosen_control_idx]["_control_obs_name"].astype(str).tolist(),
            control_library_size=np.asarray(control_metadata["library_size"])[chosen_control_idx],
        )

    def build_predict_dataset_from_adatas(
        self,
        target_adata: ad.AnnData,
        control_adata: ad.AnnData,
        train_pipeline: TransformPipeline,
        space_cfg: dict,
        vocab_maps: dict,
    ) -> tuple["PredictionDataset", dict]:
        """build predict dataset from adatas

        Returns:
            tuple of (predict_dataset, predict_metadata)
        """

        control_model_adata, control_metadata = transform_adata(
            control_adata,
            train_pipeline,
            device="cpu",
            chunk_size=int(space_cfg.get("chunk_size", 2048)),
        )
        predict_metadata = dict(control_metadata)
        predict_dataset = self.build_prediction_dataset(
            target_adata=target_adata,
            control_adata=control_adata,
            control_model_adata=control_model_adata,
            control_metadata=control_metadata,
            vocab_maps=vocab_maps,
        )
        return predict_dataset, predict_metadata
