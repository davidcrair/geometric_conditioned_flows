"""baseline lightning models"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from flatcfm.data.dataset import slice_condition_batch
from flatcfm.models.decoder import ConditionedDecoder
from flatcfm.modelcore.models.base import BasePerturbationModel


def _sample_covariate_key(cond_batch: dict, index: int) -> str:
    """build key from sample covariates only"""
    parts = []
    for name in sorted(cond_batch["sample_covariates"].keys()):
        parts.append(f"sample_cov:{name}={int(cond_batch['sample_covariates'][name][index])}")
    return "|".join(parts)


def _perturbation_covariate_key(cond_batch: dict, index: int) -> str:
    """build key from perturbation identity and perturbation covariates only"""
    parts = [f"perturbation={int(cond_batch['perturbations'][index])}"]
    for name in sorted(cond_batch["perturbation_covariates"].keys()):
        parts.append(f"pert_cov:{name}={int(cond_batch['perturbation_covariates'][name][index])}")
    return "|".join(parts)


def _tensor_condition_key(cond_batch: dict, index: int, include_perturbation: bool) -> str:
    """build condition key"""
    parts = []
    if include_perturbation:
        parts.append(f"perturbation={int(cond_batch['perturbations'][index])}")
    for name in sorted(cond_batch["perturbation_covariates"].keys()):
        parts.append(f"pert_cov:{name}={int(cond_batch['perturbation_covariates'][name][index])}")
    for name in sorted(cond_batch["sample_covariates"].keys()):
        parts.append(f"sample_cov:{name}={int(cond_batch['sample_covariates'][name][index])}")
    return "|".join(parts)


class _StatisticalBaselineModel(BasePerturbationModel):
    """base for simple statistical baselines that don't need real gradient training

    uses manual_optimization to avoid needing a dummy parameter just to
    satisfy lightning's automatic optimizer setup
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.automatic_optimization = False

    def configure_optimizers(self):
        """no optimizer needed for statistical baselines"""

        return None

    def on_train_start(self):
        """compute statistics once before training loop"""
        datamodule = self.trainer.datamodule
        dataset = datamodule.train_dataset
        self._fit_statistics(
            dataset.control_data.to(self.device), dataset.perturbed_data.to(self.device), dataset.pert_condition_batch
        )
        self.trainer.should_stop = True

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """not used for statistical baselines"""

        raise NotImplementedError("statistical baselines do not use _shared_step")

    def training_step(self, batch, batch_idx):
        loss = torch.tensor(0.0, device=self.device)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def validation_step(self, batch, batch_idx):
        loss = torch.tensor(0.0, device=self.device)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def _fit_statistics(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: dict) -> None:
        pass

    def _sample(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions"""

        return self._sample(batch["x_ctrl"].to(self.device), batch["cond_batch"])


class NoEffectModel(_StatisticalBaselineModel):
    """no effect baseline"""

    task_name = "no_effect"

    def _sample(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        del cond_batch
        return x_control


class AdditiveModel(_StatisticalBaselineModel):
    """additive baseline"""

    task_name = "additive"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("delta", torch.zeros(self.input_dim))

    def _fit_statistics(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: dict) -> None:
        del cond_perturbed
        self.delta = x_perturbed.mean(dim=0) - x_control.mean(dim=0)

    def _sample(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        del cond_batch
        return x_control + self.delta


class _KeyedMeanBaselineModel(_StatisticalBaselineModel):
    """base for keyed mean baselines that store per-condition means"""

    _include_perturbation_in_key: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("global_mean", torch.zeros(self.input_dim))
        self.means = torch.nn.ParameterDict()
        self._keys: list[tuple[str, str]] = []

    def _compute_global_mean(self, x_control: torch.Tensor, x_perturbed: torch.Tensor) -> torch.Tensor:
        """compute global mean from data"""

        raise NotImplementedError

    def _compute_group_mean(
        self,
        x_control: torch.Tensor,
        x_perturbed_group: torch.Tensor,
    ) -> torch.Tensor:
        """compute mean for a single condition group"""

        raise NotImplementedError

    def _fit_statistics(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: dict) -> None:
        keys = [
            _tensor_condition_key(cond_perturbed, i, include_perturbation=self._include_perturbation_in_key)
            for i in range(x_perturbed.shape[0])
        ]
        self.global_mean = self._compute_global_mean(x_control, x_perturbed)
        for key in sorted(set(keys)):
            mask = torch.tensor([item == key for item in keys], dtype=torch.bool, device=x_perturbed.device)
            safe_key = key.replace("|", "_").replace("=", "_").replace(":", "_")
            self.means[safe_key] = torch.nn.Parameter(
                self._compute_group_mean(x_control, x_perturbed[mask]),
                requires_grad=False,
            )
            self._keys.append((key, safe_key))

    def _sample(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        out = torch.zeros_like(x_control)
        keys = [
            _tensor_condition_key(cond_batch, i, include_perturbation=self._include_perturbation_in_key)
            for i in range(x_control.shape[0])
        ]
        key_map = {orig: safe for orig, safe in self._keys}
        for key in sorted(set(keys)):
            mask = torch.tensor([item == key for item in keys], dtype=torch.bool, device=x_control.device)
            safe_key = key_map.get(key)
            if safe_key is not None and safe_key in self.means:
                out[mask] = self.means[safe_key]
            else:
                out[mask] = self.global_mean
        return out

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["_keys"] = self._keys

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._keys = checkpoint.get("_keys", [])
        # pre-populate ParameterDict so load_state_dict finds expected keys
        state_dict = checkpoint.get("state_dict", {})
        for key in state_dict:
            if key.startswith("means."):
                param_name = key[len("means.") :]
                if param_name not in self.means:
                    self.means[param_name] = torch.nn.Parameter(
                        torch.zeros(self.input_dim),
                        requires_grad=False,
                    )


class PerturbMeanModel(_StatisticalBaselineModel):
    """perturb mean baseline

    computes the mean perturbation effect delta_p for each perturbation p
    averaged over sample covariates then predicts x_ctrl + delta_p

    for each (sample_covariate c, perturbation p) pair in training data
    delta_{c,p} = mean(perturbed_{c,p}) - mean(control_c)
    then delta_p = mean over c of delta_{c,p}
    """

    task_name = "perturb_mean"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("global_effect", torch.zeros(self.input_dim))
        self.effects = torch.nn.ParameterDict()
        self._keys: list[tuple[str, str]] = []

    def on_train_start(self):
        """compute perturbation effects from training data"""
        datamodule = self.trainer.datamodule
        dataset = datamodule.train_dataset
        x_control = dataset.control_data.to(self.device)
        x_perturbed = dataset.perturbed_data.to(self.device)
        cond_perturbed = dataset.pert_condition_batch
        ctrl_cond = slice_condition_batch(dataset.condition_batch_full, dataset.control_global_idx)
        self._fit_effects(x_control, x_perturbed, cond_perturbed, ctrl_cond)
        self.trainer.should_stop = True

    def _fit_effects(
        self,
        x_control: torch.Tensor,
        x_perturbed: torch.Tensor,
        cond_perturbed: dict,
        ctrl_cond: dict,
    ) -> None:
        # group control cells by sample covariate and compute per-context means
        ctrl_sample_keys = [_sample_covariate_key(ctrl_cond, i) for i in range(x_control.shape[0])]
        ctrl_means: dict[str, torch.Tensor] = {}
        for key in sorted(set(ctrl_sample_keys)):
            mask = torch.tensor([k == key for k in ctrl_sample_keys], dtype=torch.bool, device=x_control.device)
            ctrl_means[key] = x_control[mask].mean(dim=0)

        # build per-perturbed-cell keys
        n_pert = x_perturbed.shape[0]
        pert_sample_keys = [_sample_covariate_key(cond_perturbed, i) for i in range(n_pert)]
        pert_keys = [_perturbation_covariate_key(cond_perturbed, i) for i in range(n_pert)]

        # compute delta_{c,p} for each (sample_covariate, perturbation) pair
        # then average over sample covariates for each perturbation
        deltas_by_pert: dict[str, list[torch.Tensor]] = defaultdict(list)
        seen_pairs: set[tuple[str, str]] = set()
        for i in range(n_pert):
            pair = (pert_sample_keys[i], pert_keys[i])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
        for sample_key, pert_key in sorted(seen_pairs):
            # mask for perturbed cells matching this (sample_cov, perturbation) pair
            mask = torch.tensor(
                [pert_sample_keys[i] == sample_key and pert_keys[i] == pert_key for i in range(n_pert)],
                dtype=torch.bool,
                device=x_perturbed.device,
            )
            mu_pert = x_perturbed[mask].mean(dim=0)
            mu_ctrl = ctrl_means.get(sample_key)
            if mu_ctrl is None:
                # no control cells for this context so skip
                continue
            deltas_by_pert[pert_key].append(mu_pert - mu_ctrl)

        # average deltas over sample covariates for each perturbation
        all_effects = []
        for pert_key in sorted(deltas_by_pert.keys()):
            delta_p = torch.stack(deltas_by_pert[pert_key]).mean(dim=0)
            safe_key = pert_key.replace("|", "_").replace("=", "_").replace(":", "_")
            self.effects[safe_key] = torch.nn.Parameter(delta_p, requires_grad=False)
            self._keys.append((pert_key, safe_key))
            all_effects.append(delta_p)

        # global fallback effect
        if all_effects:
            self.global_effect = torch.stack(all_effects).mean(dim=0)
        else:
            self.global_effect = x_perturbed.mean(dim=0) - x_control.mean(dim=0)

    def _sample(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        out = torch.zeros_like(x_control)
        keys = [_perturbation_covariate_key(cond_batch, i) for i in range(x_control.shape[0])]
        key_map = {orig: safe for orig, safe in self._keys}
        for key in sorted(set(keys)):
            mask = torch.tensor([item == key for item in keys], dtype=torch.bool, device=x_control.device)
            safe_key = key_map.get(key)
            if safe_key is not None and safe_key in self.effects:
                out[mask] = x_control[mask] + self.effects[safe_key]
            else:
                out[mask] = x_control[mask] + self.global_effect
        return out

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions"""
        return self._sample(batch["x_ctrl"].to(self.device), batch["cond_batch"])

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["_keys"] = self._keys

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._keys = checkpoint.get("_keys", [])
        state_dict = checkpoint.get("state_dict", {})
        for key in state_dict:
            if key.startswith("effects."):
                param_name = key[len("effects."):]
                if param_name not in self.effects:
                    self.effects[param_name] = torch.nn.Parameter(
                        torch.zeros(self.input_dim),
                        requires_grad=False,
                    )


class ContextMeanModel(_KeyedMeanBaselineModel):
    """context mean baseline"""

    task_name = "context_mean"
    _include_perturbation_in_key = False

    def _compute_global_mean(self, x_control: torch.Tensor, x_perturbed: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_control, x_perturbed], dim=0).mean(dim=0)

    def _compute_group_mean(self, x_control: torch.Tensor, x_perturbed_group: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_control, x_perturbed_group], dim=0).mean(dim=0)


class LatentAdditiveModel(BasePerturbationModel):
    """latent additive baseline (perturbench)

    encode control to latent space add a perturbation-specific
    latent shift then decode back to gene space
    z_ctrl = encoder(x_ctrl)
    z_pert = pert_encoder(one_hot_pert)
    x' = decoder(z_ctrl + z_pert)
    """

    task_name = "latent_additive"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_perts = self.covariate_dicts.get("perturbation_num_categories", 2)
        n_layers = int(self.model_cfg.get("n_layers", 2))
        encoder_width = int(self.model_cfg.get("encoder_width", 128))
        latent_dim = int(self.model_cfg.get("latent_dim", 32))
        dropout = float(self.model_cfg.get("dropout", 0.0))

        def _build_mlp(in_dim: int, hidden: int, out_dim: int, n: int, drop: float) -> torch.nn.Sequential:
            layers = [torch.nn.Linear(in_dim, hidden), torch.nn.ReLU()]
            if drop > 0:
                layers.append(torch.nn.Dropout(drop))
            for _ in range(n - 1):
                layers.extend([torch.nn.Linear(hidden, hidden), torch.nn.ReLU()])
                if drop > 0:
                    layers.append(torch.nn.Dropout(drop))
            layers.append(torch.nn.Linear(hidden, out_dim))
            return torch.nn.Sequential(*layers)

        self.gene_encoder = _build_mlp(self.input_dim, encoder_width, latent_dim, n_layers, dropout)
        self.pert_encoder = _build_mlp(n_perts, encoder_width, latent_dim, n_layers, dropout)
        self.decoder = _build_mlp(latent_dim, encoder_width, self.input_dim, n_layers, dropout)

    def forward(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        pert_ids = cond_batch["perturbations"]
        n_perts = self.pert_encoder[0].in_features
        one_hot = torch.nn.functional.one_hot(pert_ids, n_perts).float()
        z_ctrl = self.gene_encoder(x_control)
        z_pert = self.pert_encoder(one_hot)
        return self.decoder(z_ctrl + z_pert)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)
        x_0 = x_0[: x_1.size(0)]
        cond_batch = self._move_condition_batch(batch["cond_batch"])
        x_pred = self.forward(x_0, cond_batch)
        loss = F.mse_loss(x_pred, x_1)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x_1.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions"""

        return self.forward(batch["x_ctrl"].to(self.device), self._move_condition_batch(batch["cond_batch"]))


class LinearAdditiveModel(BasePerturbationModel):
    """linear additive baseline: x' = x + Linear(one_hot_pert)

    learns a per-perturbation shift vector via a single linear layer
    from one-hot perturbation identity to gene space
    no condition encoder no dose or cell type conditioning
    equivalent to perturbench LinearAdditive
    """

    task_name = "linear_additive"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_perts = self.covariate_dicts.get("perturbation_num_categories", 2)
        self.effect_proj = torch.nn.Linear(n_perts, self.input_dim)

    def forward(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        pert_ids = cond_batch["perturbations"]
        one_hot = torch.nn.functional.one_hot(pert_ids, self.effect_proj.in_features).float()
        return x_control + self.effect_proj(one_hot)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)
        x_0 = x_0[: x_1.size(0)]
        cond_batch = self._move_condition_batch(batch["cond_batch"])
        x_pred = self.forward(x_0, cond_batch)
        loss = F.mse_loss(x_pred, x_1)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x_1.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions"""

        return self.forward(batch["x_ctrl"].to(self.device), self._move_condition_batch(batch["cond_batch"]))


class LinearModel(BasePerturbationModel):
    """linear additive baseline: x' = x + proj(E_phi(B))"""

    task_name = "linear"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cond_encoder = self._build_condition_encoder()
        self.effect_proj = torch.nn.Linear(self.cond_encoder.output_dim, self.input_dim)

    def forward(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        cond_emb = self.cond_encoder(cond_batch)
        return x_control + self.effect_proj(cond_emb)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)
        x_0 = x_0[: x_1.size(0)]
        cond_batch = self._move_condition_batch(batch["cond_batch"])
        x_pred = self.forward(x_0, cond_batch)
        loss = F.mse_loss(x_pred, x_1)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x_1.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions"""

        return self.forward(batch["x_ctrl"].to(self.device), self._move_condition_batch(batch["cond_batch"]))


class DecoderOnlyModel(BasePerturbationModel):
    """decoder only baseline with one-hot condition encoding (perturbench)"""

    task_name = "decoder_only"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(self.model_cfg.get("hidden_dim", 256))
        cond_dim = ConditionedDecoder.compute_cond_dim(self.covariate_dicts)
        self.decoder_module = ConditionedDecoder(self.input_dim, cond_dim, self.hidden_dim)

    def forward(self, x_control: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        cond_onehot = ConditionedDecoder.build_onehot(cond_batch, self.covariate_dicts)
        return self.decoder_module(x_control, cond_onehot)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        # In batch, x_0 is control, x_1 is perturbed.
        # FlowMatching uses random subsets of controls. The datamodule pairs them via OT or randomly.
        # It's better to just take x_0 and predict x_1.
        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)
        # Ensure sizes match
        x_0 = x_0[: x_1.size(0)]

        cond_batch = self._move_condition_batch(batch["cond_batch"])
        x_pred = self.forward(x_0, cond_batch)
        loss = F.mse_loss(x_pred, x_1)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x_1.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions"""

        return self.forward(batch["x_ctrl"].to(self.device), self._move_condition_batch(batch["cond_batch"]))
