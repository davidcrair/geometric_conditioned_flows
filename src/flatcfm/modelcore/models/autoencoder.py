"""autoencoder lightning model"""

from __future__ import annotations

import torch

from flatcfm.models.autoencoder import LinearAutoEncoder, NegativeBinomialAutoEncoder, StandardAutoEncoder
from flatcfm.training.losses import CosineDistancePreservationLoss, DistancePreservationLoss, Log1pMSEAuxLoss, LossComposer, MSEReconLoss, NBReconLoss, OroJaRLoss, PredictorCrossEntropyLoss, PullbackIsotropyLoss
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from .base import BasePerturbationModel


class AutoencoderModel(BasePerturbationModel):
    """autoencoder model"""

    task_name = "ae"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = int(self.model_cfg.get("latent_dim", 128))
        self.family = str(self.model_cfg.get("family", "negative_binomial"))
        decoder_cfg = self.model_cfg.get("decoder", {})
        self.mean_head = str(decoder_cfg.get("mean_head", "per_cell_gene"))
        space_base_cfg = self.space_config.get("base", {})
        self.input_space_kind = str(space_base_cfg.get("kind", "normalized_log1p"))
        self.target_sum = float(space_base_cfg.get("target_sum", 1e4))
        n_cell_types = self.model_init_kwargs.get("n_cell_types")
        n_perturbations = self.model_init_kwargs.get("n_perturbations")

        if self.family == "standard":
            self.model = StandardAutoEncoder(
                n_genes=self.input_dim,
                latent_dim=self.latent_dim,
                hidden_dim=int(self.model_cfg.get("hidden_dim", 256)),
                n_layers=int(self.model_cfg.get("n_layers", 3)),
                dropout=float(self.model_cfg.get("dropout", 0.1)),
                input_space_kind=self.input_space_kind,
                target_sum=self.target_sum,
                output_activation=str(self.model_cfg.get("output_activation", "softplus")),
            )
            recon_loss = MSEReconLoss(
                gene_variance_normalize=bool(self.loss_cfg.get("mse_gene_variance_normalize", False)),
                floor_frac_of_mean=float(self.loss_cfg.get("mse_variance_floor_frac", 0.1)),
                pert_weighted=bool(self.loss_cfg.get("mse_pert_weighted", False)),
            )
        elif self.family == "linear":
            self.model = LinearAutoEncoder(
                n_genes=self.input_dim,
                latent_dim=self.latent_dim,
                input_space_kind=self.input_space_kind,
                target_sum=self.target_sum,
            )
            recon_loss = MSEReconLoss(
                gene_variance_normalize=bool(self.loss_cfg.get("mse_gene_variance_normalize", False)),
                floor_frac_of_mean=float(self.loss_cfg.get("mse_variance_floor_frac", 0.1)),
                pert_weighted=bool(self.loss_cfg.get("mse_pert_weighted", False)),
            )
        elif self.family == "negative_binomial":
            if self.mean_head != "per_cell_gene":
                raise ValueError(f"Unsupported decoder mean head for negative binomial autoencoder: {self.mean_head}")
            self.model = NegativeBinomialAutoEncoder(
                n_genes=self.input_dim,
                latent_dim=self.latent_dim,
                hidden_dim=int(self.model_cfg.get("hidden_dim", 256)),
                n_layers=int(self.model_cfg.get("n_layers", 3)),
                dropout=float(self.model_cfg.get("dropout", 0.1)),
                mean_head=self.mean_head,
                dispersion_mode=str(decoder_cfg.get("dispersion_head", "shared_gene")),
                input_space_kind=self.input_space_kind,
                target_sum=self.target_sum,
                n_cell_types=n_cell_types,
                n_perturbations=n_perturbations,
                rate_parameterization=str(decoder_cfg.get("rate_parameterization", "independent")),
            )
            recon_loss = NBReconLoss(
                gene_normalize=bool(self.loss_cfg.get("nb_gene_normalize", False)),
            )
        else:
            raise ValueError(f"Unsupported autoencoder family: {self.family}")
        distance_loss_type = str(self.loss_cfg.get("distance_loss_type", "mse"))
        if distance_loss_type == "cosine":
            distance_loss = CosineDistancePreservationLoss(
                zeta=float(self.loss_cfg.get("distance_zeta", 0.0)),
            )
        elif distance_loss_type == "mse":
            distance_loss = DistancePreservationLoss(
                zeta=float(self.loss_cfg.get("distance_zeta", 0.0)),
                alpha_min=float(self.loss_cfg.get("distance_alpha_min", 1.0)),
            )
        else:
            raise ValueError(f"unsupported distance_loss_type: {distance_loss_type}")

        self.loss_composer = LossComposer(
            {
                "recon": recon_loss,
                "distance": distance_loss,
                "pullback": PullbackIsotropyLoss(
                    alpha_min=float(self.loss_cfg.get("pullback_alpha_min", 1.0)),
                ),
                "ce_context": PredictorCrossEntropyLoss(
                    logits_key="logits_context", targets_key="cell_type_ids",
                ),
                "ce_state": PredictorCrossEntropyLoss(
                    logits_key="logits_state", targets_key="perturbation_ids",
                    label_smoothing=float(self.loss_cfg.get("ce_state_label_smoothing", 0.1)),
                ),
                "orojar": OroJaRLoss(),
                "log1p_mse": Log1pMSEAuxLoss(),
            },
            self.loss_cfg["weights"],
        )

    def on_fit_start(self) -> None:
        """lightning hook fires once at the start of trainer.fit

        for standard family MSE recon this installs two optional per-gene
        loss modifiers when their respective flags are enabled

          gene_variance_normalize: divides per-gene squared error by
            training gene variance (generally hurts downstream metrics
            because it down-weights high-variance perturbation responsive
            genes, kept for ablation purposes)
          pert_weighted: multiplies per-gene squared error by a binary
            mask that upweights the top N perturbation-responsive genes
            by a fixed multiplier this directly rewards the model for
            reconstructing perturbation-relevant structure
        """

        super().on_fit_start()
        if self.family not in ("standard", "linear"):
            return
        if "recon" not in self.loss_composer.loss_map:
            return
        recon_term = self.loss_composer.loss_map["recon"]

        want_varnorm = getattr(recon_term, "gene_variance_normalize", False)
        want_pertweight = getattr(recon_term, "pert_weighted", False)
        if not want_varnorm and not want_pertweight:
            return

        if want_varnorm and recon_term.gene_variance.numel() == 0:
            self._install_gene_variance(recon_term)

        if want_pertweight and recon_term.gene_weights.numel() == 0:
            self._install_pert_weighted_mask(recon_term)

    def _install_gene_variance(self, recon_term) -> None:
        """compute per-gene training variance and install on the recon loss"""

        rank_zero_warn(
            "Computing per-gene training variance for variance-normalized MSE loss..."
        )
        running_sum = torch.zeros(self.input_dim, device=self.device)
        running_sum_sq = torch.zeros(self.input_dim, device=self.device)
        n_cells = 0
        with torch.no_grad():
            train_loader = self.trainer.datamodule.train_dataloader()
            for batch in train_loader:
                x = batch["x_input"].to(self.device)
                running_sum += x.sum(dim=0)
                running_sum_sq += (x ** 2).sum(dim=0)
                n_cells += x.size(0)
        mean = running_sum / max(n_cells, 1)
        variance = running_sum_sq / max(n_cells, 1) - mean ** 2
        recon_term.set_gene_variance(variance.clamp(min=1e-6))
        rank_zero_warn(
            f"Set gene variance buffer: n_cells={n_cells}, "
            f"min={variance.min().item():.4e}, max={variance.max().item():.4e}, "
            f"mean={variance.mean().item():.4e}"
        )

    def _install_pert_weighted_mask(self, recon_term) -> None:
        """compute top-N perturbation-responsive gene indices and install

        ranks genes by std of their mean expression across training
        perturbation groups (defined by cell_type x perturbation x dose)
        then builds a binary weight vector of 1s with the top N set to
        pert_weight_multiplier
        """

        import numpy as np

        top_n = int(self.loss_cfg.get("mse_pert_top_n", 500))
        multiplier = float(self.loss_cfg.get("mse_pert_weight_multiplier", 10.0))

        dm = self.trainer.datamodule
        schema = dm.schema
        control_col = str(schema.control_column)
        control_val = str(schema.control_value)
        pert_col = str(schema.perturbation_source)
        covariate_keys = tuple(str(cov.source_column) for cov in schema.sample_covariates)

        held_out_mask = np.asarray(dm.masks["is_held_out"], dtype=bool)
        train_adata = dm.adata_full[~held_out_mask].copy()
        x_train, _, _ = dm.train_pipeline.transform(train_adata, device="cpu")
        x_train = np.asarray(x_train, dtype=np.float32)

        train_obs = train_adata.obs.copy().reset_index(drop=True)
        pert_mask = train_obs[control_col].astype(str).to_numpy() != control_val
        pert_obs = train_obs.loc[pert_mask].reset_index(drop=True)
        x_train_pert = x_train[pert_mask]

        group_cols = [*covariate_keys, pert_col, "dose"]
        group_means = []
        for _, grp in pert_obs.groupby(list(group_cols), observed=True):
            if len(grp) < 2:
                continue
            group_means.append(x_train_pert[grp.index.to_numpy()].mean(axis=0))
        if not group_means:
            rank_zero_warn(
                "No training perturbation groups found; skipping pert-weighted mask."
            )
            return
        group_means_arr = np.stack(group_means)
        responsiveness = group_means_arr.std(axis=0)

        top_n = min(top_n, responsiveness.shape[0])
        top_idx = np.argsort(-responsiveness)[:top_n]

        weights = torch.ones(self.input_dim, dtype=torch.float32, device=self.device)
        weights[torch.as_tensor(top_idx, device=self.device)] = multiplier
        recon_term.set_gene_weights(weights)

        rank_zero_warn(
            f"Installed pert-weighted mask: top_n={top_n} multiplier={multiplier} "
            f"resp_range=[{responsiveness[top_idx].min():.4f}, {responsiveness[top_idx].max():.4f}] "
            f"weight_ratio={multiplier:.1f}x for top-N vs 1.0x for others"
        )

    def configure_optimizers(self):
        """configure optimizers"""

        optimizer_name = str(self.task_cfg.get("optimizer", "adamw")).lower()
        lr = float(self.task_cfg.get("lr", 1e-4))
        weight_decay = float(self.task_cfg.get("weight_decay", 0.0))

        if "optimizer" not in self.task_cfg:
            rank_zero_warn(
                f"No optimizer specified in task_cfg; defaulting to AdamW (lr={lr}, weight_decay={weight_decay})."
            )

        if optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        raise ValueError(f"Unsupported optimizer for autoencoder model: {optimizer_name}")

    def set_loss_weights(self, loss_weights: dict) -> None:
        """set loss weights"""

        self.loss_composer.loss_weights = dict(loss_weights)

    def set_trainable_parts(self, encoder: bool, decoder: bool) -> None:
        """set trainable parts"""

        for param in self.model.encoder.parameters():
            param.requires_grad = encoder

        decoder_hidden = getattr(self.model, "decoder_hidden", None)
        if decoder_hidden is not None:
            for param in decoder_hidden.parameters():
                param.requires_grad = decoder

        decoder_module = getattr(self.model, "decoder", None)
        if decoder_module is not None and decoder_hidden is None:
            for param in decoder_module.parameters():
                param.requires_grad = decoder

        dec_log_rate = getattr(self.model, "dec_log_rate", None)
        if dec_log_rate is not None:
            for param in dec_log_rate.parameters():
                param.requires_grad = decoder

        dec_log_theta = getattr(self.model, "dec_log_theta", None)
        if dec_log_theta is not None:
            for param in dec_log_theta.parameters():
                param.requires_grad = decoder

        log_theta = getattr(self.model, "log_theta", None)
        if log_theta is not None:
            log_theta.requires_grad = decoder

        # alpha is the learned distance scale factor used by the distance
        # loss so it must be trainable whenever encoder OR decoder is active
        self.model.alpha.requires_grad = encoder or decoder

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """shared step"""

        x_raw = batch["x_raw"].to(self.device)
        x_input = batch["x_input"].to(self.device)
        lib_size = batch["lib_size"].to(self.device)
        subset_lib_size = batch.get("subset_lib_size")
        if subset_lib_size is not None:
            subset_lib_size = subset_lib_size.to(self.device)
        # for softmax rate parameterization, decode with L_subset so
        # mu = L_subset * softmax(logits) and sum(mu) = L_subset (matching raw subset counts)
        # for independent rates, keep L_total (existing behavior)
        rate_param = getattr(self.model, "rate_parameterization", "independent")
        decode_lib = subset_lib_size if (rate_param == "softmax" and subset_lib_size is not None) else lib_size
        cell_type_ids = batch.get("cell_type_ids")
        if cell_type_ids is not None:
            cell_type_ids = cell_type_ids.to(self.device)
        perturbation_ids = batch.get("perturbation_ids")
        if perturbation_ids is not None:
            perturbation_ids = perturbation_ids.to(self.device)
        z = self.model.encode(x_input)

        # orojar predictor heads
        has_predictors = (
            self.loss_composer.loss_weights.get("orojar", 0) > 0
            or self.loss_composer.loss_weights.get("ce_context", 0) > 0
            or self.loss_composer.loss_weights.get("ce_state", 0) > 0
        )
        if has_predictors and getattr(self.model, "context_predictor", None) is not None:
            # z needs grad tracking for autograd.grad in OroJaRLoss
            if not z.requires_grad:
                z = z.requires_grad_(True)
            logits_context = self.model.context_predictor(z)
            logits_state = self.model.state_predictor(z)
        else:
            logits_context = None
            logits_state = None

        # pdist for the distance loss runs on gpu here instead of in the
        # dataloader which used to do it on cpu and starve the gpu for ~100 ms
        # per step distance_embeddings is the raw potential matrix slice legacy
        # runs may still send a precomputed distances vector in which case we
        # forward it through unchanged
        distance_embeddings = batch.get("distance_embeddings")
        if distance_embeddings is not None:
            distance_embeddings = distance_embeddings.to(self.device, non_blocking=True)
        original_distances = batch.get("distances")
        if original_distances is not None:
            original_distances = original_distances.to(self.device, non_blocking=True)

        if self.family in ("standard", "linear"):
            recon = self.model.decode(z, decode_lib)
            loss, individual_losses = self.loss_composer(
                x_input=x_input,
                recon=recon,
                z=z,
                model=self.model,
                original_distances=original_distances,
                distance_embeddings=distance_embeddings,
                cell_type_ids=cell_type_ids,
                perturbation_ids=perturbation_ids,
                logits_context=logits_context,
                logits_state=logits_state,
                n_genes_total=self.input_dim,
                gene_subsample=self.loss_cfg.get("gene_subsample"),
            )
        else:
            mu, theta = self.model.decode(z, decode_lib)
            loss, individual_losses = self.loss_composer(
                x_raw=x_raw,
                x_input=x_input,
                mu=mu,
                theta=theta,
                z=z,
                model=self.model,
                original_distances=original_distances,
                distance_embeddings=distance_embeddings,
                cell_type_ids=cell_type_ids,
                perturbation_ids=perturbation_ids,
                logits_context=logits_context,
                logits_state=logits_state,
                total_lib_size=lib_size,
                target_sum=self.target_sum,
                n_genes_total=self.input_dim,
                gene_subsample=self.loss_cfg.get("gene_subsample"),
            )
        self._log_losses(stage, loss, individual_losses, batch_size=x_raw.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """encode and reconstruct"""

        x_input = batch["x_input"].to(self.device)
        lib_size = batch["lib_size"].to(self.device)
        input_lib_size = batch.get("input_lib_size")
        if input_lib_size is not None:
            input_lib_size = input_lib_size.to(self.device)
        # for softmax, decode with L_subset; normalize with L_total
        rate_param = getattr(self.model, "rate_parameterization", "independent")
        if rate_param == "softmax":
            subset_lib_size = batch.get("subset_lib_size")
            if subset_lib_size is not None:
                decode_lib = subset_lib_size.to(self.device)
            else:
                decode_lib = lib_size
        else:
            decode_lib = lib_size
        z = self.model.encode(x_input)
        return self.model.reconstruct_input(z, decode_lib, input_library_size=input_lib_size or lib_size)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """predict step"""

        del batch_idx, dataloader_idx
        with torch.no_grad():
            preds = self._predict(batch)
        return {"predictions": preds.detach().cpu()}

    def export_metadata(self) -> dict:
        """export metadata"""

        metadata = super().export_metadata()
        metadata["latent_dim"] = self.latent_dim
        metadata["family"] = self.family
        metadata["input_space_kind"] = self.input_space_kind
        metadata["target_sum"] = self.target_sum
        metadata["mean_head"] = self.mean_head
        metadata["dispersion_head"] = self.model_cfg.get("decoder", {}).get("dispersion_head", "shared_gene")
        metadata["rate_parameterization"] = getattr(self.model, "rate_parameterization", "independent")
        metadata["loss_weights"] = dict(self.loss_composer.loss_weights)
        return metadata
