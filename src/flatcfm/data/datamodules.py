"""lightning datamodules for flatcfm"""

from __future__ import annotations

import logging

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import anndata as ad
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pertpy
import torch
from torch.utils.data import DataLoader, Dataset

from flatcfm._utils import dense_array, to_plain_dict

logger = logging.getLogger(__name__)

from flatcfm.data.dataset import CondFMDataset, ConditionFirstBatchSampler, slice_condition_batch
from flatcfm.data.simulations import make_gaussian_to_moons
from flatcfm.data.splitters import (
    apply_holdout_masks,
    build_holdout_manifest,
    load_cell_names_csv,
    load_manifest_json,
    save_cell_names_csv,
    save_manifest_json,
    select_subsample_cell_names,
    select_stratified_cell_names,
    validate_no_leakage,
)
from flatcfm.data.space import AELatentProjection, TransformPipeline

from .ae_dataloader import make_ae_dataloader
from .accessors import resolve_sciplex_artifacts
from ._datamodule_parts import ConditionEncoder, PipelineManager, PredictionBuilder
from .schema import ConditionSchema
from .space import (
    get_or_build_pipeline,
    load_ae_projection,
    normalize_evaluation_space_config,
    normalize_space_config,
    pipeline_label,
    pipeline_tag_from_config,
    resolve_ae_projection_artifacts,
    save_ae_projection,
    transform_adata,
    upstream_pipeline_tag_for_ae,
)



@dataclass(frozen=True)
class RuntimeArtifacts:
    """runtime artifacts"""

    tag: str


class PredictionDataset(Dataset):
    """prediction dataset"""

    def __init__(
        self,
        x_ctrl: torch.Tensor,
        condition_batch: dict,
        obs: pd.DataFrame,
        control_obs_names: list[str],
        control_library_size: np.ndarray,
    ):
        self.x_ctrl = x_ctrl
        self.condition_batch = condition_batch
        self.obs = obs.reset_index(names="_target_obs_name")
        self.control_obs_names = list(control_obs_names)
        self.control_library_size = np.asarray(control_library_size, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.x_ctrl.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def _make_pair_collate(dataset: CondFMDataset, pool_multiplier: int = 1):
    """build pair collate

    Args:
        dataset: the fm dataset containing control and perturbed data
        pool_multiplier: sample pool_multiplier * B controls so the training step can OT-assign
    """

    def collate(batch_items):
        pert_local_idx = torch.as_tensor([item["pert_local_idx"] for item in batch_items], dtype=torch.long)
        x_1 = dataset.perturbed_data.index_select(0, pert_local_idx)
        n_controls = x_1.size(0) * pool_multiplier
        idx_0 = torch.randint(0, dataset.control_data.size(0), (n_controls,))
        x_0 = dataset.control_data.index_select(0, idx_0)
        cond_batch = slice_condition_batch(dataset.pert_condition_batch, pert_local_idx)
        return {
            "x_0": x_0,
            "x_1": x_1,
            "cond_batch": cond_batch,
        }

    return collate


def _make_prediction_collate(dataset: PredictionDataset):
    """build prediction collate"""

    def collate(batch_items):
        batch_idx = torch.as_tensor(batch_items, dtype=torch.long)
        return {
            "x_ctrl": dataset.x_ctrl.index_select(0, batch_idx),
            "cond_batch": slice_condition_batch(dataset.condition_batch, batch_idx),
            "obs": dataset.obs.iloc[batch_idx.tolist()].copy(),
            "control_obs_name": [dataset.control_obs_names[idx] for idx in batch_idx.tolist()],
            "control_library_size": dataset.control_library_size[batch_idx.tolist()],
        }

    return collate


class BasePerturbationDataModule(pl.LightningDataModule):
    """base datamodule"""

    @staticmethod
    def _resolve_evaluation_space(space_cfg: dict, evaluation_space_cfg: dict) -> dict:
        """resolve evaluation space config"""

        return normalize_evaluation_space_config(space_cfg, evaluation_space_cfg)

    def __init__(
        self,
        data: dict,
        splitter: dict,
        space: dict,
        condition: dict,
        paths: dict,
        evaluation_space: dict | None = None,
        ae_geometry: dict | None = None,
        task: dict | None = None,
        predict: dict | None = None,
        trainer: dict | None = None,
        **extra_kwargs,
    ):
        super().__init__()
        del extra_kwargs
        self.data_cfg = to_plain_dict(data)
        self.splitter_cfg = to_plain_dict(splitter)
        self.space_cfg = normalize_space_config(to_plain_dict(space), default_fit_scope="train")
        self.evaluation_space_cfg = self._resolve_evaluation_space(
            self.space_cfg,
            to_plain_dict(evaluation_space),
        )
        self.condition_cfg = to_plain_dict(condition)
        self.paths_cfg = to_plain_dict(paths)
        self.ae_geometry_cfg = to_plain_dict(ae_geometry)
        self.task_cfg = to_plain_dict(task)
        self.predict_cfg = to_plain_dict(predict)
        self.trainer_cfg = to_plain_dict(trainer)
        self.schema = ConditionSchema.from_config(self.condition_cfg)
        self.task_name = str(self.task_cfg.get("name", "fm"))
        self.batch_size = int(self.task_cfg.get("batch_size", 128))
        self.use_sampler = bool(self.task_cfg.get("use_sampler", True))
        self.steps_per_epoch = int(self.task_cfg.get("steps_per_epoch", 100))
        self.predict_batch_size = int(self.predict_cfg.get("batch_size", self.batch_size))
        self.num_workers = int(self.trainer_cfg.get("num_workers", 0))
        self.pin_memory = bool(self.trainer_cfg.get("pin_memory", False))

        self.artifacts = RuntimeArtifacts(tag=str(self.data_cfg.get("name", "runtime")))
        self.train_pipeline: TransformPipeline | None = None
        self.evaluation_pipeline: TransformPipeline | None = None
        self.adata_full = None
        self.masks = None
        self.vocab_maps = None
        self.covariate_dicts = None
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None
        self.train_metadata = {}
        self.val_metadata = {}
        self.predict_metadata = {}
        self.predict_spec = {}
        self.ae_train_loader = None
        self.ae_val_loader = None
        self.ae_train_adata = None
        self.ae_val_adata = None

        self._pipeline_mgr = PipelineManager(
            space_cfg=self.space_cfg,
            evaluation_space_cfg=self.evaluation_space_cfg,
            paths_cfg=self.paths_cfg,
            data_cfg=self.data_cfg,
            artifact_tag=self.artifacts.tag,
        )
        self._condition_encoder = ConditionEncoder(self.schema)
        self._prediction_builder = PredictionBuilder(
            schema=self.schema,
            predict_cfg=self.predict_cfg,
            splitter_cfg=self.splitter_cfg,
            condition_encoder=self._condition_encoder,
        )

    def _artifact_prefix(self) -> str:
        """build artifact prefix"""

        return self._pipeline_mgr.artifact_prefix()

    def _resolve_pipeline_path(self, space_cfg: dict) -> Path:
        """resolve pipeline cache path"""

        return self._pipeline_mgr.resolve_pipeline_path(space_cfg)

    def _ae_export_artifact_tag(self) -> str:
        """resolve ae export artifact tag"""

        return self._pipeline_mgr.ae_export_artifact_tag()

    def _resolve_ae_projection_bundle(self, artifact_tag: str) -> tuple[str, Path, Path, Path]:
        """resolve ae projection bundle"""

        return self._pipeline_mgr.resolve_ae_projection_bundle(artifact_tag)

    def _fit_scope_adata(self, fit_scope: str) -> ad.AnnData:
        """select fit scope adata"""

        raise NotImplementedError

    def _prediction_split_mask(self, split: str) -> np.ndarray:
        """build prediction split mask"""

        raise NotImplementedError

    def _prepare_pipelines(self) -> None:
        """prepare pipelines"""

        self.train_pipeline, self.evaluation_pipeline = self._pipeline_mgr.prepare_pipelines(
            self._fit_scope_adata,
        )

    def _build_vocab_maps(self) -> tuple[dict, dict]:
        """build vocab maps"""

        return self._condition_encoder.build_vocab_maps(self.adata_full.obs)

    def _build_condition_batch(self, adata_obj: ad.AnnData) -> dict:
        """build condition batch"""

        return self._condition_encoder.build_condition_batch(adata_obj, self.vocab_maps)

    def _build_predict_dataset_from_adatas(self, target_adata: ad.AnnData, control_adata: ad.AnnData) -> None:
        """build predict dataset from adatas"""

        self.predict_dataset, predict_metadata = self._prediction_builder.build_predict_dataset_from_adatas(
            target_adata=target_adata,
            control_adata=control_adata,
            train_pipeline=self.train_pipeline,
            space_cfg=self.space_cfg,
            vocab_maps=self.vocab_maps,
        )
        self.predict_metadata = predict_metadata
        self.predict_metadata["prediction_spec"] = self.predict_spec

    def get_input_dim(self) -> int:
        """get input dim"""

        if self.train_dataset is None and self.ae_train_loader is None:
            self.setup("fit")
        if self.task_name == "ae":
            sample_batch = next(iter(self.ae_train_loader))
            return int(sample_batch["x_input"].shape[1])
        return int(self.train_dataset.data.shape[1])

    def get_space_mode(self) -> str:
        """get train space mode"""

        return pipeline_label(self.space_cfg)

    def get_evaluation_space_mode(self) -> str:
        """get evaluation space mode"""

        return pipeline_label(self.evaluation_space_cfg)

    def get_model_feature_names(self) -> list[str]:
        """get model feature names"""

        if self.train_metadata:
            return list(self.train_metadata["feature_names"])
        if self.train_pipeline is not None:
            return list(self.train_pipeline.feature_names_out())
        if self.adata_full is None:
            self.setup("fit")
        return list(self.adata_full.var_names)

    def get_export_feature_names(self) -> list[str]:
        """get export feature names

        returns base space feature names (hvg gene names) since predictions
        are always exported in base space
        """

        if self.train_pipeline is not None:
            return list(self.train_pipeline.feature_names_in)
        if self.predict_metadata.get("feature_names"):
            return list(self.predict_metadata["feature_names"])
        return self.get_model_feature_names()

    def get_model_init_kwargs(self) -> dict:
        """get model init kwargs"""

        kwargs = {
            "input_dim": self.get_input_dim(),
            "covariate_dicts": self.covariate_dicts,
            "feature_names": self.get_model_feature_names(),
            "schema": self.schema.to_dict(),
            "space_mode": self.get_space_mode(),
            "space_config": self.space_cfg,
            "evaluation_space_config": self.evaluation_space_cfg,
            "vocab_maps": self.vocab_maps,
        }
        if hasattr(self, "_ae_n_perturbations"):
            kwargs["n_perturbations"] = self._ae_n_perturbations
            kwargs["n_cell_types"] = self._ae_n_cell_types
        dim_weights = self._get_dimension_weights()
        if dim_weights is not None:
            kwargs["dimension_weights"] = dim_weights
        return kwargs

    def _get_dimension_weights(self) -> list[float] | None:
        """get per-dimension variance weights from pca projection if available"""

        if self.train_pipeline is None or not self.train_pipeline.projections:
            return None
        from flatcfm.data.geometry import PCAProjection
        for proj in self.train_pipeline.projections:
            if isinstance(proj, PCAProjection) and proj.model is not None:
                return proj.model.explained_variance_ratio_.tolist()
        return None

    def decode_predictions(self, predictions: torch.Tensor, library_size: np.ndarray) -> np.ndarray:
        """decode predictions from training space to base space

        for models without projections predictions are already in base space
        for non-ae projections (pca ortho lift) use exact inverse_to_base
        for ae projections go through raw counts since ae decode is lossy
        """

        matrix = predictions.detach().cpu().float().numpy()
        if self.train_pipeline is None:
            return matrix
        if not self.train_pipeline.projections:
            return matrix  # already in base space

        lib = np.asarray(library_size, dtype=np.float32)
        sample = bool(self.predict_cfg.get("sample_decode", False))

        if isinstance(self.train_pipeline.projections[-1], AELatentProjection):
            raw = self.train_pipeline.inverse_to_raw(matrix, library_size=lib, sample=sample)
            return np.asarray(
                self.train_pipeline.base_transform.transform_raw(
                    raw, lib, list(self.train_pipeline.feature_names_in)
                ).matrix,
                dtype=np.float32,
            )
        return self.train_pipeline.inverse_to_base(matrix, library_size=lib, sample=sample)

    def export_prediction_outputs(self, outputs: list[dict]) -> tuple[np.ndarray, pd.DataFrame, list[str], dict]:
        """export prediction outputs"""

        pred_chunks = []
        obs_frames = []
        library_sizes = []
        for output in outputs:
            pred_chunks.append(output["predictions"].detach().cpu())
            obs_frame = output["obs"].copy()
            obs_frame["_control_obs_name"] = output["control_obs_name"]
            obs_frames.append(obs_frame)
            library_sizes.append(np.asarray(output["control_library_size"], dtype=np.float32))

        pred_matrix = torch.cat(pred_chunks, dim=0) if pred_chunks else torch.zeros((0, self.get_input_dim()))
        obs = pd.concat(obs_frames, axis=0, ignore_index=True) if obs_frames else pd.DataFrame()
        decoded = self.decode_predictions(pred_matrix, np.concatenate(library_sizes, axis=0) if library_sizes else np.zeros((0,)))
        feature_names = self.get_export_feature_names()
        if decoded.shape[0] > 0 and decoded.shape[1] != len(feature_names):
            raise ValueError(
                f"decoded predictions have {decoded.shape[1]} features but "
                f"export feature names have {len(feature_names)} - "
                f"shape mismatch in decode_predictions"
            )
        # predictions are now always in base space (normalized + hvg no projections)
        base_space_cfg = deepcopy(self.space_cfg)
        base_space_cfg["projections"] = []
        prediction_space_label = pipeline_label(base_space_cfg)
        obs["_prediction_space"] = prediction_space_label
        metadata = {
            "task_name": self.task_name,
            "space_mode": self.get_space_mode(),
            "evaluation_space_mode": self.get_evaluation_space_mode(),
            "prediction_space_label": prediction_space_label,
            "space_config": self.space_cfg,
            "evaluation_space_config": self.evaluation_space_cfg,
            "train_pipeline_spec": self.train_pipeline.export_spec() if self.train_pipeline is not None else None,
            "evaluation_pipeline_spec": self.evaluation_pipeline.export_spec() if self.evaluation_pipeline is not None else None,
            "canonical_space_kind": "base",
            "n_predictions": int(decoded.shape[0]),
            "prediction_spec": self.predict_spec,
            "feature_names": feature_names,
        }
        return decoded, obs, feature_names, metadata

    def get_reference_adata(self, feature_names: list[str]) -> ad.AnnData:
        """get reference adata"""

        raise NotImplementedError

    def export_ae_artifacts(self, ae_model: torch.nn.Module, checkpoint_path: str | None, run_dir: Path) -> dict:
        """export ae artifacts"""

        del ae_model, checkpoint_path, run_dir
        return {}

    def train_dataloader(self):
        """train dataloader"""

        if self.task_name == "ae":
            return self.ae_train_loader
        pool_multiplier = (
            int(self.task_cfg.get("ot_pool_multiplier", 1))
            if self.task_name == "fm" and bool(self.task_cfg.get("use_ot_coupling", False))
            else 1
        )
        collate = _make_pair_collate(self.train_dataset, pool_multiplier=pool_multiplier)
        if self.use_sampler:
            sampler = ConditionFirstBatchSampler(
                self.train_dataset,
                batch_size=self.batch_size,
                steps_per_epoch=self.steps_per_epoch,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=collate,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """val dataloader"""

        if self.task_name == "ae":
            return self.ae_val_loader
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_make_pair_collate(self.val_dataset),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        """predict dataloader"""

        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            collate_fn=_make_prediction_collate(self.predict_dataset),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class SciplexDataModule(BasePerturbationDataModule):
    """sciplex datamodule"""

    def __init__(
        self,
        data: dict,
        splitter: dict,
        space: dict,
        condition: dict,
        paths: dict,
        evaluation_space: dict | None = None,
        ae_geometry: dict | None = None,
        task: dict | None = None,
        predict: dict | None = None,
        trainer: dict | None = None,
        **extra_kwargs,
    ):
        super().__init__(
            data=data,
            splitter=splitter,
            space=space,
            evaluation_space=evaluation_space,
            condition=condition,
            paths=paths,
            ae_geometry=ae_geometry,
            task=task,
            predict=predict,
            trainer=trainer,
            **extra_kwargs,
        )
        self.artifacts = resolve_sciplex_artifacts(self.splitter_cfg, self.paths_cfg)
        self._pipeline_mgr.update_artifact_tag(self.artifacts.tag)

    def _prepare_canonical_subsample(self) -> ad.AnnData:
        """prepare canonical subsample"""

        self.artifacts.subsample_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts.split_artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.artifacts.subsample_h5ad_path.exists():
            return ad.read_h5ad(self.artifacts.subsample_h5ad_path)

        source = self.data_cfg.get("source", "pertpy")
        if source != "pertpy":
            data_path = self.data_cfg.get("data_path")
            if data_path is None:
                raise FileNotFoundError("data.data_path must be set when data.source is not pertpy")
            adata_full = ad.read_h5ad(data_path)
        else:
            adata_full = pertpy.data.sciplex3_raw()

        # optionally keep only highest dose per drug (perturbench style)
        if self.data_cfg.get("max_dose_only", False):
            dose_col = "dose"
            vehicle_col = "vehicle"
            product_col = "product_name"
            is_ctrl = adata_full.obs[vehicle_col].astype(int) == 1
            dose_numeric = pd.to_numeric(adata_full.obs[dose_col], errors="coerce")
            max_dose_per_drug = dose_numeric.groupby(adata_full.obs[product_col]).transform("max")
            keep = is_ctrl | (dose_numeric == max_dose_per_drug)
            n_before = adata_full.n_obs
            adata_full = adata_full[keep.to_numpy(dtype=bool)].copy()
            logger.info(
                "max_dose_only: kept %d / %d cells (controls + highest dose per drug)",
                adata_full.n_obs, n_before,
            )

        if self.artifacts.subsample_cells_csv_path.exists():
            cell_names = load_cell_names_csv(self.artifacts.subsample_cells_csv_path)
            # filter to cells that still exist after dose filtering
            cell_names = [c for c in cell_names if c in set(adata_full.obs_names)]
        else:
            base_cell_names = select_subsample_cell_names(
                adata_full,
                n_cells=int(self.splitter_cfg.get("subsample_n_cells", 100_000)),
                seed=int(self.splitter_cfg.get("subsample_seed", 0)),
            )
            if self._include_all_controls():
                # keep the perturbed cells picked by the seeded subsample but
                # add every control cell in the full dataset so controls match
                # perts across every split
                vehicle_col = "vehicle"
                is_ctrl_full = adata_full.obs[vehicle_col].astype(int) == 1
                all_control_names = adata_full.obs_names[is_ctrl_full.to_numpy(dtype=bool)].tolist()
                base_set = set(base_cell_names)
                pert_from_base = [c for c in base_cell_names if c not in set(all_control_names)]
                merged: list[str] = []
                seen: set[str] = set()
                for c in pert_from_base + all_control_names:
                    if c in seen:
                        continue
                    seen.add(c)
                    merged.append(c)
                cell_names = merged
                logger.info(
                    "include_all_controls: %d perts (from seeded subsample) + %d controls (all) = %d cells",
                    len(pert_from_base), len(all_control_names), len(cell_names),
                )
            else:
                cell_names = base_cell_names
            save_cell_names_csv(cell_names, self.artifacts.subsample_cells_csv_path)
        adata_sub = adata_full[cell_names].copy()
        adata_sub.write_h5ad(self.artifacts.subsample_h5ad_path)
        return adata_sub

    def _include_all_controls(self) -> bool:
        """include-all-controls flag from splitter config"""

        return bool(self.splitter_cfg.get("include_all_controls", False))

    def _prepare_manifest(self, adata_obj: ad.AnnData) -> dict:
        """prepare manifest"""

        if self.artifacts.holdout_json_path.exists():
            return load_manifest_json(self.artifacts.holdout_json_path)
        manifest = build_holdout_manifest(
            adata_obj,
            type(
                "Cfg",
                (),
                {
                    "seed": int(self.splitter_cfg.get("seed", 42)),
                    "test_cell_type": str(self.splitter_cfg.get("test_cell_type", "K562")),
                    "holdout_fraction": float(self.splitter_cfg.get("holdout_fraction", 0.5)),
                    "split_policy": str(self.splitter_cfg.get("split_policy", "strict_no_leakage")),
                    "disjoint_partition_cell_types": tuple(
                        self.splitter_cfg.get("disjoint_partition_cell_types", []) or []
                    ),
                },
            )(),
        )
        save_manifest_json(manifest, self.artifacts.holdout_json_path)
        return manifest

    def _split_val_mask_for_adata(self, adata_obj: ad.AnnData) -> np.ndarray:
        """build local val mask"""

        control_mask = (adata_obj.obs[self.schema.control_column].to_numpy() == self.schema.control_value).astype(bool)
        pert_mask = ~control_mask
        rng = np.random.default_rng(int(self.splitter_cfg.get("seed", 42)))
        val_fraction = float(self.splitter_cfg.get("val_fraction", 0.1))

        val_mask = np.zeros(adata_obj.n_obs, dtype=bool)
        for mask in [control_mask, pert_mask]:
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            n_val = max(1, int(np.floor(idx.size * val_fraction)))
            chosen = rng.choice(idx, size=n_val, replace=False)
            val_mask[chosen] = True
        return val_mask

    def _build_train_val_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """build train val masks"""

        train_mask = np.asarray(self.masks["is_train"], dtype=bool)
        ctrl_mask = np.asarray(self.masks["is_ctrl"], dtype=bool) & train_mask
        pert_mask = np.asarray(self.masks["is_pert_any"], dtype=bool) & train_mask
        rng = np.random.default_rng(int(self.splitter_cfg.get("seed", 42)))
        val_fraction = float(self.splitter_cfg.get("val_fraction", 0.1))

        val_mask = np.zeros(self.adata_full.n_obs, dtype=bool)
        include_all_controls = self._include_all_controls()
        # iterate in the same order either way so the rng state consumed when
        # choosing val perts is identical to the baseline splitter and perts
        # land in the exact same (train val) buckets as without the flag
        # when the flag is set the ctrl val assignment is discarded so all
        # controls remain in both train and val
        for mask_kind, mask in [("ctrl", ctrl_mask), ("pert", pert_mask)]:
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            n_val = max(1, int(np.floor(idx.size * val_fraction)))
            chosen = rng.choice(idx, size=n_val, replace=False)
            if include_all_controls and mask_kind == "ctrl":
                # consume rng identically but discard the assignment
                continue
            val_mask[chosen] = True

        train_out = train_mask & ~val_mask
        val_out = val_mask
        if include_all_controls:
            all_ctrl = np.asarray(self.masks["is_ctrl"], dtype=bool)
            train_out = train_out | all_ctrl
            val_out = val_out | all_ctrl
        return train_out, val_out

    def _fit_scope_adata(self, fit_scope: str) -> ad.AnnData:
        """select fit scope adata"""

        if fit_scope == "full_dataset":
            return self.adata_full
        if fit_scope == "train":
            train_mask = np.asarray(self.masks["is_train"], dtype=bool)
            return self.adata_full[train_mask].copy()
        raise ValueError(f"Unsupported fit_scope: {fit_scope}")

    def _prediction_split_mask(self, split: str) -> np.ndarray:
        """build sciplex prediction split mask"""

        if split == "held_out":
            held_out = np.asarray(self.masks["is_held_out"], dtype=bool)
            if self._include_all_controls():
                held_out = held_out | np.asarray(self.masks["is_ctrl"], dtype=bool)
            return held_out
        if split == "train":
            train_mask, _ = self._build_train_val_masks()
            return np.asarray(train_mask, dtype=bool)
        if split == "val":
            _, val_mask = self._build_train_val_masks()
            return np.asarray(val_mask, dtype=bool)
        raise ValueError(f"Unsupported prediction split: {split}")

    def _load_or_select_ae_cell_names(self) -> list[str]:
        """load or select ae training cells"""

        if self.artifacts.ae_train_cells_csv_path.exists():
            return load_cell_names_csv(self.artifacts.ae_train_cells_csv_path)

        train_adata = self.adata_full[np.asarray(self.masks["is_train"], dtype=bool)].copy()
        cell_names = select_stratified_cell_names(
            train_adata,
            n_cells=int(self.splitter_cfg.get("ae_subsample_n_cells", 50_000)),
            seed=int(self.splitter_cfg.get("ae_subsample_seed", 42)),
            group_cols=tuple(self.splitter_cfg.get("ae_subsample_group_cols", ["cell_type", "vehicle"])),
        )
        save_cell_names_csv(cell_names, self.artifacts.ae_train_cells_csv_path)
        return cell_names

    def _ae_geometry_matrix(self, adata_obj: ad.AnnData) -> np.ndarray:
        """build ambient ae geometry matrix"""

        matrix, _, _ = self.train_pipeline.transform(adata_obj, device="cpu")
        return np.asarray(matrix, dtype=np.float32)

    def _ae_geometry_cache_paths(self) -> tuple[Path, Path]:
        """resolve ae geometry cache paths"""

        base_tag = pipeline_tag_from_config(self.space_cfg)
        cache_root = self.ae_geometry_cfg.get("cache_path")
        if cache_root:
            cache_dir = Path(cache_root)
            train_path = cache_dir / f"{self._artifact_prefix()}_ae_geometry_train_{base_tag}.npy"
            val_path = cache_dir / f"{self._artifact_prefix()}_ae_geometry_val_{base_tag}.npy"
            return train_path, val_path
        return (
            Path(self.paths_cfg.get("space_dir", "artifacts/spaces")) / f"{self._artifact_prefix()}_ae_geometry_train_{base_tag}.npy",
            Path(self.paths_cfg.get("space_dir", "artifacts/spaces")) / f"{self._artifact_prefix()}_ae_geometry_val_{base_tag}.npy",
        )

    def _compute_phate_potential(self, matrix: np.ndarray) -> np.ndarray:
        """compute phate embedding coordinates

        returns the phate coordinate embedding so the dataloader computes
        pairwise distances in phate space per batch (matching mioflow-lite)
        """

        import phate

        phate_cfg = self.ae_geometry_cfg.get("phate", {})
        n_landmark = min(int(phate_cfg.get("n_landmark", 2000)), matrix.shape[0])
        operator = phate.PHATE(
            n_components=int(phate_cfg.get("n_components", 2)),
            knn=int(phate_cfg.get("knn", 5)),
            n_landmark=n_landmark,
            t=phate_cfg.get("t", "auto"),
            verbose=bool(phate_cfg.get("verbose", False)),
        )
        embedding = operator.fit_transform(matrix)
        return np.asarray(embedding, dtype=np.float32)

    def _compute_phate_diff_potential(self, matrix: np.ndarray) -> np.ndarray:
        """compute full phate diffusion potential per cell

        returns the (n_cells, n_landmarks) potential matrix where each row is a
        cell's potential vector pairwise l2 between rows equals the true phate
        potential distance before any mds compression
        """

        import phate

        phate_cfg = self.ae_geometry_cfg.get("phate", {})
        n_landmark = min(int(phate_cfg.get("n_landmark", 2000)), matrix.shape[0])
        operator = phate.PHATE(
            n_components=2,
            knn=int(phate_cfg.get("knn", 5)),
            n_landmark=n_landmark,
            t=phate_cfg.get("t", "auto"),
            verbose=bool(phate_cfg.get("verbose", False)),
        )
        operator.fit(matrix)
        potential = np.asarray(operator.diff_potential, dtype=np.float32)
        return potential

    def _compute_phate_diff_potential_per_cell_type(
        self, matrix: np.ndarray, cell_types: np.ndarray
    ) -> np.ndarray:
        """compute full diff potential independently per cell type

        returned matrix is block diagonal cell i of cell type c has its
        potential vector in the column block assigned to c and zeros elsewhere
        cross cell type pairs in torch pdist get the sum of both cells own
        blocks but the distance loss masks them out anyway so those values
        are never used
        """

        unique_types = np.unique(cell_types)
        per_ct_potentials: list[tuple[np.ndarray, np.ndarray]] = []
        total_cols = 0
        for ct in unique_types:
            mask = cell_types == ct
            pot = self._compute_phate_diff_potential(matrix[mask])
            per_ct_potentials.append((mask, pot))
            total_cols += pot.shape[1]

        result = np.zeros((matrix.shape[0], total_cols), dtype=np.float32)
        col_offset = 0
        for mask, pot in per_ct_potentials:
            m = pot.shape[1]
            result[mask, col_offset : col_offset + m] = pot
            col_offset += m
        return result

    def _compute_phate_per_cell_type(self, matrix: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
        """compute phate embeddings independently per cell type

        each cell type gets its own phate operator so the manifold
        structure reflects within-cell-type perturbation variation
        rather than between-cell-type differences
        """

        unique_types = np.unique(cell_types)
        n_components = int(self.ae_geometry_cfg.get("phate", {}).get("n_components", 2))
        embeddings = np.zeros((matrix.shape[0], n_components), dtype=np.float32)

        for ct in unique_types:
            mask = cell_types == ct
            ct_embedding = self._compute_phate_potential(matrix[mask])
            embeddings[mask] = ct_embedding

        return embeddings

    def _prepare_ae_geometry_embeddings(self, train_adata: ad.AnnData, val_adata: ad.AnnData) -> tuple[np.ndarray | None, np.ndarray | None]:
        """prepare ae geometry embeddings"""

        geometry_mode = self.ae_geometry_cfg.get("mode", "none")
        if geometry_mode == "none":
            return None, None
        if geometry_mode == "ambient_euclidean":
            train_embed = self._ae_geometry_matrix(train_adata)
            val_embed = self._ae_geometry_matrix(val_adata) if val_adata.n_obs > 0 else np.zeros((0, train_embed.shape[1]), dtype=np.float32)
            return np.asarray(train_embed, dtype=np.float32), np.asarray(val_embed, dtype=np.float32)
        if geometry_mode not in {"phate_potential", "phate_diff_potential"}:
            raise ValueError(f"Unsupported ae geometry mode: {geometry_mode}")

        use_diff_potential = geometry_mode == "phate_diff_potential"
        per_cell_type = bool(self.ae_geometry_cfg.get("per_cell_type", False))
        # when true fits phate once on (train concat val) then slices the result
        # back into train and val portions so val targets live in the same
        # landmark basis as train targets eliminating the basis mismatch that
        # makes the val distance loss an apples to oranges comparison
        unified_fit = bool(self.ae_geometry_cfg.get("unified_train_val_fit", True))
        phate_cfg = self.ae_geometry_cfg.get("phate", {})
        # include phate hyperparams and mode in cache key so different configs
        # never collide and old 2d caches do not get reused for diff potential
        knn = int(phate_cfg.get("knn", 5))
        if use_diff_potential:
            cache_suffix = f"_k{knn}_diffpot"
        else:
            n_comp = int(phate_cfg.get("n_components", 2))
            cache_suffix = f"_k{knn}_d{n_comp}"
        if per_cell_type:
            cache_suffix += "_per_ct"
        if unified_fit:
            cache_suffix += "_unified"
        train_path, val_path = self._ae_geometry_cache_paths()
        train_path = train_path.with_name(train_path.stem + cache_suffix + train_path.suffix)
        val_path = val_path.with_name(val_path.stem + cache_suffix + val_path.suffix)

        ct_col = str(self.ae_geometry_cfg.get("cell_type_column", "cell_type"))

        def _compute(matrix: np.ndarray, cell_types: np.ndarray | None) -> np.ndarray:
            if use_diff_potential:
                if per_cell_type:
                    return self._compute_phate_diff_potential_per_cell_type(matrix, cell_types)
                return self._compute_phate_diff_potential(matrix)
            if per_cell_type:
                return self._compute_phate_per_cell_type(matrix, cell_types)
            return self._compute_phate_potential(matrix)

        if unified_fit and val_adata.n_obs > 0:
            # fit one phate on train concat val then slice back
            if train_path.exists() and val_path.exists():
                train_embed = np.load(train_path)
                val_embed = np.load(val_path)
            else:
                train_matrix = self._ae_geometry_matrix(train_adata)
                val_matrix = self._ae_geometry_matrix(val_adata)
                combined = np.concatenate([train_matrix, val_matrix], axis=0)
                combined_ct = None
                if per_cell_type:
                    combined_ct = np.concatenate(
                        [
                            train_adata.obs[ct_col].to_numpy(),
                            val_adata.obs[ct_col].to_numpy(),
                        ]
                    )
                combined_embed = _compute(combined, combined_ct)
                n_train = int(train_matrix.shape[0])
                train_embed = combined_embed[:n_train]
                val_embed = combined_embed[n_train:]
                train_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(train_path, train_embed)
                np.save(val_path, val_embed)
        else:
            # legacy path independent phate fits on train and val
            if train_path.exists():
                train_embed = np.load(train_path)
            else:
                train_matrix = self._ae_geometry_matrix(train_adata)
                train_ct = train_adata.obs[ct_col].to_numpy() if per_cell_type else None
                train_embed = _compute(train_matrix, train_ct)
                train_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(train_path, train_embed)

            if val_adata.n_obs == 0:
                val_embed = np.zeros((0, train_embed.shape[1]), dtype=np.float32)
            elif val_path.exists():
                val_embed = np.load(val_path)
            else:
                val_matrix = self._ae_geometry_matrix(val_adata)
                val_ct = val_adata.obs[ct_col].to_numpy() if per_cell_type else None
                val_embed = _compute(val_matrix, val_ct)
                val_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(val_path, val_embed)

        return np.asarray(train_embed, dtype=np.float32), np.asarray(val_embed, dtype=np.float32)

    def _prepare_cond_datasets(self) -> None:
        """prepare conditional datasets"""

        train_mask, val_mask = self._build_train_val_masks()
        train_adata = self.adata_full[train_mask].copy()
        val_adata = self.adata_full[val_mask].copy()

        train_model_adata, self.train_metadata = transform_adata(
            train_adata,
            self.train_pipeline,
            device="cpu",
            chunk_size=int(self.space_cfg.get("chunk_size", 2048)),
        )
        val_model_adata, self.val_metadata = transform_adata(
            val_adata,
            self.train_pipeline,
            device="cpu",
            chunk_size=int(self.space_cfg.get("chunk_size", 2048)),
        )
        self.train_dataset = CondFMDataset(
            train_model_adata,
            self._build_condition_batch(train_adata),
            control_col=self.schema.control_column,
            control_value=self.schema.control_value,
            use_pca=False,
            use_norm=False,
        )
        self.val_dataset = CondFMDataset(
            val_model_adata,
            self._build_condition_batch(val_adata),
            control_col=self.schema.control_column,
            control_value=self.schema.control_value,
            use_pca=False,
            use_norm=False,
        )

    def _prepare_ae_loaders(self) -> None:
        """prepare ae loaders"""

        input_feature_names = list(self.train_pipeline.feature_names_in)
        ae_cell_names = self._load_or_select_ae_cell_names()
        ae_pool = self.adata_full[ae_cell_names].copy()
        val_mask = self._split_val_mask_for_adata(ae_pool)
        train_adata = ae_pool[~val_mask].copy()
        val_adata = ae_pool[val_mask].copy()
        train_distances, val_distances = self._prepare_ae_geometry_embeddings(train_adata, val_adata)
        self.ae_train_adata = train_adata.copy()
        self.ae_val_adata = val_adata.copy()
        train_x_input, train_input_lib_size, _ = self.train_pipeline.transform(train_adata, device="cpu")
        val_x_input, val_input_lib_size, _ = self.train_pipeline.transform(val_adata, device="cpu")
        train_x_raw = dense_array(train_adata[:, input_feature_names].X)
        val_x_raw = dense_array(val_adata[:, input_feature_names].X)
        self.train_metadata = {"feature_names": self.train_pipeline.feature_names_out()}
        self.val_metadata = {"feature_names": self.train_pipeline.feature_names_out()}

        # cell type and perturbation IDs (used by distance masking and orojar)
        ct_col = str(self.ae_geometry_cfg.get("cell_type_column", "cell_type"))
        all_ct_cats = ae_pool.obs[ct_col].astype("category").cat.categories
        train_ct_ids = train_adata.obs[ct_col].astype("category").cat.set_categories(all_ct_cats).cat.codes.to_numpy()
        val_ct_ids = val_adata.obs[ct_col].astype("category").cat.set_categories(all_ct_cats).cat.codes.to_numpy()

        pert_col = str(self.schema.perturbation_source)
        all_pert_cats = ae_pool.obs[pert_col].astype("category").cat.categories
        train_pert_ids = train_adata.obs[pert_col].astype("category").cat.set_categories(all_pert_cats).cat.codes.to_numpy()
        val_pert_ids = val_adata.obs[pert_col].astype("category").cat.set_categories(all_pert_cats).cat.codes.to_numpy()
        self._ae_n_perturbations = len(all_pert_cats)
        self._ae_n_cell_types = len(all_ct_cats)

        self.ae_train_loader = make_ae_dataloader(
            train_adata,
            distances=train_distances,
            cell_type_ids=train_ct_ids,
            perturbation_ids=train_pert_ids,
            batch_size=self.batch_size,
            shuffle=True,
            x_raw=train_x_raw,
            x_input=train_x_input,
            library_size=train_input_lib_size,
            input_library_size=train_input_lib_size,
            input_space_kind=str(self.space_cfg["base"]["kind"]),
            target_sum=float(self.space_cfg["base"]["target_sum"]),
            num_workers=self.num_workers,
        )
        self.ae_val_loader = make_ae_dataloader(
            val_adata,
            distances=val_distances,
            cell_type_ids=val_ct_ids,
            perturbation_ids=val_pert_ids,
            batch_size=self.batch_size,
            shuffle=False,
            x_raw=val_x_raw,
            x_input=val_x_input,
            library_size=val_input_lib_size,
            input_library_size=val_input_lib_size,
            input_space_kind=str(self.space_cfg["base"]["kind"]),
            target_sum=float(self.space_cfg["base"]["target_sum"]),
            num_workers=self.num_workers,
        )

    def _prepare_predict_dataset(self) -> None:
        """prepare predict dataset"""

        self.predict_spec = self._prediction_builder.resolve_prediction_spec()
        target_adata = self._prediction_builder.select_target_adata(
            self.predict_spec, self.adata_full, self._prediction_split_mask,
        )
        control_adata = self._prediction_builder.select_control_adata(
            self.predict_spec, target_adata, self.adata_full, self._prediction_split_mask,
        )
        if target_adata.n_obs == 0:
            raise ValueError("Prediction target selection produced zero cells")
        if control_adata.n_obs == 0:
            raise ValueError("Prediction control selection produced zero cells")
        self._build_predict_dataset_from_adatas(target_adata, control_adata)

    def setup(self, stage: str | None = None) -> None:
        """setup stage"""

        if self.adata_full is None:
            self.adata_full = self._prepare_canonical_subsample()
            manifest = self._prepare_manifest(self.adata_full)
            self.masks = apply_holdout_masks(self.adata_full, manifest)
            validate_no_leakage(
                self.adata_full, self.masks,
                product_name_col="product_name",
                split_policy=manifest.get("split_policy", "strict_no_leakage"),
            )
            self._prepare_pipelines()
            self.vocab_maps, self.covariate_dicts = self._build_vocab_maps()

        if stage in {None, "fit", "validate"}:
            if self.task_name == "ae":
                self._prepare_ae_loaders()
            else:
                self._prepare_cond_datasets()

        if stage in {None, "predict"}:
            self._prepare_predict_dataset()

    def get_reference_adata(self, feature_names: list[str]) -> ad.AnnData:
        """get reference adata"""

        ref_model_adata, _ = transform_adata(
            self.adata_full,
            self.evaluation_pipeline,
            device="cpu",
            chunk_size=int(self.evaluation_space_cfg.get("chunk_size", 2048)),
        )
        return ref_model_adata[:, feature_names].copy()

    def export_ae_artifacts(self, ae_model: torch.nn.Module, checkpoint_path: str | None, run_dir: Path) -> dict:
        """export ae artifacts"""

        del run_dir
        artifact_tag = self._ae_export_artifact_tag()
        bundle_tag, projection_path, metadata_path, ae_model_path = self._resolve_ae_projection_bundle(artifact_tag)
        metadata = {
            "artifact_tag": bundle_tag,
            "dataset_name": str(self.data_cfg.get("name", "dataset")),
            "feature_names": list(self.train_pipeline.feature_names_out()),
            "projection_kind": "ae_latent",
            "space_config": self.space_cfg,
            "target_sum": float(self.space_cfg["base"]["target_sum"]),
            "latent_dim": int(getattr(ae_model, "latent_dim", 0)),
            "geometry_mode": self.ae_geometry_cfg.get("mode", "none"),
            "ae_family": getattr(ae_model, "family", "negative_binomial"),
            "upstream_pipeline_spec": self.train_pipeline.export_spec(),
        }

        if checkpoint_path is not None:
            src = Path(checkpoint_path).expanduser().resolve()
            ae_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, ae_model_path)

        ae_projection = AELatentProjection(
            ae_model=ae_model.cpu(),
            artifact_tag=bundle_tag,
            input_feature_names=list(self.train_pipeline.feature_names_out()),
            latent_dim=int(getattr(ae_model, "latent_dim", 0)),
        )
        save_ae_projection(ae_projection, projection_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return {
            "ae_model_path": str(ae_model_path),
            "ae_projection_path": str(projection_path),
            "ae_metadata_path": str(metadata_path),
            "ae_artifact_tag": bundle_tag,
        }


class ToyDataModule(BasePerturbationDataModule):
    """toy datamodule"""

    def _load_toy_adata(self) -> ad.AnnData:
        """load toy adata"""

        data_path = self.data_cfg.get("data_path")
        if data_path:
            return ad.read_h5ad(data_path)
        dataset_name = str(self.data_cfg.get("dataset_name", "gaussian_to_moons"))
        if dataset_name != "gaussian_to_moons":
            raise ValueError(f"Unsupported toy dataset: {dataset_name}")
        return make_gaussian_to_moons(n_samples=int(self.data_cfg.get("n_samples", 2000)))

    def _build_train_val_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """build train val masks"""

        control_mask = (self.adata_full.obs[self.schema.control_column].to_numpy() == self.schema.control_value).astype(bool)
        pert_mask = ~control_mask
        rng = np.random.default_rng(int(self.splitter_cfg.get("seed", 42)))
        val_fraction = float(self.splitter_cfg.get("val_fraction", 0.1))
        val_mask = np.zeros(self.adata_full.n_obs, dtype=bool)
        for mask in [control_mask, pert_mask]:
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            n_val = max(1, int(np.floor(idx.size * val_fraction)))
            chosen = rng.choice(idx, size=n_val, replace=False)
            val_mask[chosen] = True
        return ~val_mask, val_mask

    def _fit_scope_adata(self, fit_scope: str) -> ad.AnnData:
        """select fit scope adata"""

        if fit_scope == "full_dataset":
            return self.adata_full
        if fit_scope == "train":
            train_mask, _ = self._build_train_val_masks()
            return self.adata_full[train_mask].copy()
        raise ValueError(f"Unsupported fit_scope: {fit_scope}")

    def _prediction_split_mask(self, split: str) -> np.ndarray:
        """build toy prediction split mask"""

        train_mask, val_mask = self._build_train_val_masks()
        if split == "held_out":
            return np.asarray(val_mask, dtype=bool)
        if split == "train":
            return np.asarray(train_mask, dtype=bool)
        if split == "val":
            return np.asarray(val_mask, dtype=bool)
        raise ValueError(f"Unsupported prediction split: {split}")

    def _prepare_ae_geometry_embeddings(self, train_adata: ad.AnnData, val_adata: ad.AnnData) -> tuple[np.ndarray | None, np.ndarray | None]:
        """prepare toy ae geometry embeddings"""

        geometry_mode = self.ae_geometry_cfg.get("mode", "none")
        if geometry_mode == "none":
            return None, None
        if geometry_mode != "ambient_euclidean":
            raise ValueError(f"Unsupported toy ae geometry mode: {geometry_mode}")
        train_embed, _, _ = self.train_pipeline.transform(train_adata, device="cpu")
        if val_adata.n_obs == 0:
            val_embed = np.zeros((0, train_embed.shape[1]), dtype=np.float32)
        else:
            val_embed, _, _ = self.train_pipeline.transform(val_adata, device="cpu")
        return np.asarray(train_embed, dtype=np.float32), np.asarray(val_embed, dtype=np.float32)

    def _export_generic_ae_artifacts(self, ae_model: torch.nn.Module, checkpoint_path: str | None, run_dir: Path) -> dict:
        """export generic ae artifacts"""

        del run_dir
        artifact_tag = self._ae_export_artifact_tag()
        bundle_tag, projection_path, metadata_path, ae_model_path = self._resolve_ae_projection_bundle(artifact_tag)
        metadata = {
            "artifact_tag": bundle_tag,
            "dataset_name": str(self.data_cfg.get("name", "dataset")),
            "feature_names": list(self.train_pipeline.feature_names_out()),
            "projection_kind": "ae_latent",
            "space_config": self.space_cfg,
            "target_sum": float(self.space_cfg["base"]["target_sum"]),
            "latent_dim": int(getattr(ae_model, "latent_dim", 0)),
            "geometry_mode": self.ae_geometry_cfg.get("mode", "none"),
            "ae_family": getattr(ae_model, "family", "negative_binomial"),
            "upstream_pipeline_spec": self.train_pipeline.export_spec(),
        }

        if checkpoint_path is not None:
            src = Path(checkpoint_path).expanduser().resolve()
            ae_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, ae_model_path)

        ae_projection = AELatentProjection(
            ae_model=ae_model.cpu(),
            artifact_tag=bundle_tag,
            input_feature_names=list(self.train_pipeline.feature_names_out()),
            latent_dim=int(getattr(ae_model, "latent_dim", 0)),
        )
        save_ae_projection(ae_projection, projection_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return {
            "ae_model_path": str(ae_model_path),
            "ae_projection_path": str(projection_path),
            "ae_metadata_path": str(metadata_path),
            "ae_artifact_tag": bundle_tag,
        }

    def setup(self, stage: str | None = None) -> None:
        """setup stage"""

        if self.adata_full is None:
            self.adata_full = self._load_toy_adata()
            self._prepare_pipelines()
            self.vocab_maps, self.covariate_dicts = self._build_vocab_maps()

        if stage in {None, "fit", "validate"}:
            train_mask, val_mask = self._build_train_val_masks()
            train_adata = self.adata_full[train_mask].copy()
            val_adata = self.adata_full[val_mask].copy()
            if self.task_name == "ae":
                train_distances, val_distances = self._prepare_ae_geometry_embeddings(train_adata, val_adata)
                self.ae_train_adata = train_adata.copy()
                self.ae_val_adata = val_adata.copy()
                input_feature_names = list(self.train_pipeline.feature_names_in)
                train_x_input, train_input_lib_size, _ = self.train_pipeline.transform(train_adata, device="cpu")
                val_x_input, val_input_lib_size, _ = self.train_pipeline.transform(val_adata, device="cpu")
                train_x_raw = dense_array(train_adata[:, input_feature_names].X)
                val_x_raw = dense_array(val_adata[:, input_feature_names].X)
                self.train_metadata = {"feature_names": self.train_pipeline.feature_names_out()}
                self.val_metadata = {"feature_names": self.train_pipeline.feature_names_out()}
                self.ae_train_loader = make_ae_dataloader(
                    train_adata,
                    distances=train_distances,
                    batch_size=self.batch_size,
                    shuffle=True,
                    x_raw=train_x_raw,
                    x_input=train_x_input,
                    library_size=train_input_lib_size,
                    input_library_size=train_input_lib_size,
                    input_space_kind=str(self.space_cfg["base"]["kind"]),
                    target_sum=float(self.space_cfg["base"]["target_sum"]),
                )
                self.ae_val_loader = make_ae_dataloader(
                    val_adata,
                    distances=val_distances,
                    batch_size=self.batch_size,
                    shuffle=False,
                    x_raw=val_x_raw,
                    x_input=val_x_input,
                    library_size=val_input_lib_size,
                    input_library_size=val_input_lib_size,
                    input_space_kind=str(self.space_cfg["base"]["kind"]),
                    target_sum=float(self.space_cfg["base"]["target_sum"]),
                )
            else:
                train_model_adata, self.train_metadata = transform_adata(
                    train_adata,
                    self.train_pipeline,
                    device="cpu",
                    chunk_size=int(self.space_cfg.get("chunk_size", 2048)),
                )
                val_model_adata, self.val_metadata = transform_adata(
                    val_adata,
                    self.train_pipeline,
                    device="cpu",
                    chunk_size=int(self.space_cfg.get("chunk_size", 2048)),
                )
                self.train_dataset = CondFMDataset(
                    train_model_adata,
                    self._build_condition_batch(train_adata),
                    control_col=self.schema.control_column,
                    control_value=self.schema.control_value,
                    use_pca=False,
                    use_norm=False,
                )
                self.val_dataset = CondFMDataset(
                    val_model_adata,
                    self._build_condition_batch(val_adata),
                    control_col=self.schema.control_column,
                    control_value=self.schema.control_value,
                    use_pca=False,
                    use_norm=False,
                )

        if stage in {None, "predict"}:
            self.predict_spec = self._prediction_builder.resolve_prediction_spec()
            target_adata = self._prediction_builder.select_target_adata(
                self.predict_spec, self.adata_full, self._prediction_split_mask,
            )
            control_adata = self._prediction_builder.select_control_adata(
                self.predict_spec, target_adata, self.adata_full, self._prediction_split_mask,
            )
            if target_adata.n_obs == 0:
                raise ValueError("Prediction target selection produced zero cells")
            if control_adata.n_obs == 0:
                raise ValueError("Prediction control selection produced zero cells")
            self._build_predict_dataset_from_adatas(target_adata, control_adata)

    def get_reference_adata(self, feature_names: list[str]) -> ad.AnnData:
        """get reference adata"""

        ref_model_adata, _ = transform_adata(
            self.adata_full,
            self.evaluation_pipeline,
            device="cpu",
            chunk_size=int(self.evaluation_space_cfg.get("chunk_size", 2048)),
        )
        return ref_model_adata[:, feature_names].copy()

    def export_ae_artifacts(self, ae_model: torch.nn.Module, checkpoint_path: str | None, run_dir: Path) -> dict:
        """export ae artifacts"""

        return self._export_generic_ae_artifacts(ae_model, checkpoint_path, run_dir)
