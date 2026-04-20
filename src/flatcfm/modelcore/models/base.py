"""base lightning model"""

from __future__ import annotations

from abc import ABC, abstractmethod

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from flatcfm._utils import to_plain_dict
from flatcfm.data.dataset import condition_batch_to_device
from flatcfm.models.flow import ConditionEncoder


class BasePerturbationModel(ABC, pl.LightningModule):
    """base perturbation model"""

    task_name = "base"

    def __init__(
        self,
        model_cfg: dict,
        task_cfg: dict,
        loss_cfg: dict,
        predict_cfg: dict,
        input_dim: int,
        covariate_dicts: dict,
        feature_names: list[str],
        schema: dict,
        space_mode: str,
        space_config: dict | None = None,
        evaluation_space_config: dict | None = None,
        **extra_kwargs,
    ):
        super().__init__()
        self.model_init_kwargs = dict(extra_kwargs)
        self.model_cfg = to_plain_dict(model_cfg)
        self.task_cfg = to_plain_dict(task_cfg)
        self.loss_cfg = to_plain_dict(loss_cfg)
        self.predict_cfg = to_plain_dict(predict_cfg)
        self.input_dim = int(input_dim)
        self.covariate_dicts = covariate_dicts
        self.feature_names = list(feature_names)
        self.schema = schema
        self.space_mode = str(space_mode)
        self.space_config = to_plain_dict(space_config)
        self.evaluation_space_config = to_plain_dict(evaluation_space_config)

    def _build_condition_encoder(self) -> ConditionEncoder:
        """build condition encoder"""

        pretrained = self._resolve_pretrained_perturbation_embeddings()
        return ConditionEncoder(
            covariate_dicts=self.covariate_dicts,
            condition_dim=int(self.model_cfg.get("condition_dim", 128)),
            embedding_dim=int(self.model_cfg.get("embedding_dim", 64)),
            projection_dim=int(self.model_cfg.get("projection_dim", 64)),
            pretrained_perturbation_embeddings=pretrained,
            embedding_dropout=float(self.model_cfg.get("embedding_dropout", 0.0)),
            perturbation_dropout=float(self.model_cfg.get("perturbation_dropout", 0.0)),
            sample_covariate_dropout=float(self.model_cfg.get("sample_covariate_dropout", 0.0)),
        )

    def _resolve_pretrained_perturbation_embeddings(self) -> torch.Tensor | None:
        """build a frozen embedding tensor aligned to the perturbation vocab

        reads drug_embedding_path from model_cfg which points to a pt file
        written by scripts/build_drug_embeddings.py payload shape is
            {"embeddings": dict[drug_name -> tensor], "dim": int, ...}
        drugs missing from the payload fall back to zero vectors
        """

        from pathlib import Path

        path = self.model_cfg.get("drug_embedding_path")
        if not path:
            return None
        vocab_maps = self.model_init_kwargs.get("vocab_maps")
        if vocab_maps is None:
            raise ValueError(
                "drug_embedding_path set but vocab_maps was not passed to the model "
                "this is a datamodule wiring bug"
            )
        pert_vocab = vocab_maps.get("perturbations", {})
        if not pert_vocab:
            raise ValueError("vocab_maps['perturbations'] is empty")

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        emb_map = payload["embeddings"]
        dim = int(payload["dim"])
        # build tensor in vocab index order so emb_pert(idx) returns the
        # correct drug embedding for each vocabulary token
        num_perts = len(pert_vocab)
        tensor = torch.zeros((num_perts, dim), dtype=torch.float32)
        missing_names: list[str] = []
        for name, idx in pert_vocab.items():
            vec = emb_map.get(str(name))
            if vec is None:
                missing_names.append(str(name))
                continue
            tensor[idx] = torch.as_tensor(vec, dtype=torch.float32)
        if missing_names:
            rank_zero_warn(
                f"{len(missing_names)} perturbations have no pretrained embedding and got zero vectors "
                f"examples: {missing_names[:5]}"
            )
        return tensor

    def _move_condition_batch(self, cond_batch: dict) -> dict:
        """move condition batch"""

        return condition_batch_to_device(cond_batch, self.device)

    def _log_losses(self, stage: str, total_loss: torch.Tensor, individual_losses: dict, batch_size: int) -> None:
        """log losses"""

        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for name, value in individual_losses.items():
            self.log(
                f"{stage}_{name}",
                torch.tensor(float(value), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        optimizer_name = str(self.task_cfg.get("optimizer", "adamw")).lower()
        lr = float(self.task_cfg.get("lr", 1e-4))
        weight_decay = float(self.task_cfg.get("weight_decay", 0.0))
        # optional override to apply a different weight decay to the
        # conditioning nn.Embedding tables AdamW applies the global
        # weight_decay to embeddings too but they often benefit from a
        # heavier penalty since one row per vocab token is high risk for
        # overfitting to (pert x cell_type) co adaptations leave at None
        # to fall back to the global weight_decay
        embedding_weight_decay = self.task_cfg.get("embedding_weight_decay")

        if "optimizer" not in self.task_cfg:
            rank_zero_warn(
                f"No optimizer specified in task_cfg; defaulting to AdamW (lr={lr}, weight_decay={weight_decay})."
            )

        if embedding_weight_decay is not None and optimizer_name in {"adam", "adamw"}:
            # split params into (cond embeddings, everything else) so they
            # can get separate weight decay values
            embedding_params = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "cond_encoder" in name and (".emb_pert" in name or ".emb_sample_cov" in name or ".emb_pert_cov" in name):
                    embedding_params.append(param)
                else:
                    other_params.append(param)
            param_groups = [
                {"params": other_params, "weight_decay": weight_decay},
                {"params": embedding_params, "weight_decay": float(embedding_weight_decay)},
            ]
            rank_zero_warn(
                f"using split weight decay: backbone={weight_decay} embeddings={embedding_weight_decay}"
            )
            opt_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
            return opt_cls(param_groups, lr=lr)

        if optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @abstractmethod
    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """compute loss for a single batch"""

        ...

    @abstractmethod
    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions from a batch"""

        ...

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """training step"""

        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """validation step"""

        del batch_idx
        return self._shared_step(batch, "val")

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """predict step"""

        del batch_idx, dataloader_idx
        with torch.no_grad():
            preds = self._predict(batch)
        return {
            "predictions": preds.detach().cpu(),
            "obs": batch["obs"],
            "control_obs_name": batch["control_obs_name"],
            "control_library_size": batch["control_library_size"],
        }

    def export_metadata(self) -> dict:
        """export metadata"""

        return {
            "task_name": self.task_name,
            "feature_names": self.feature_names,
            "schema_output_obs_map": self.schema.get("output_obs_map", {}),
            "space_mode": self.space_mode,
            "space_config": self.space_config,
            "evaluation_space_config": self.evaluation_space_config,
        }
