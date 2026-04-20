"""
composable loss registry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp as func_jvp, vmap
from typing import Dict, Tuple
import ot
from geomloss import SamplesLoss


class LossTerm(nn.Module):
    """base class for loss component"""

    def forward(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("subclasses should implement this method")


class LossComposer(nn.Module):
    def __init__(self, loss_map: Dict[str, LossTerm], loss_weights: Dict[str, float]):
        super().__init__()
        self.loss_map = nn.ModuleDict(loss_map)
        self.loss_weights = loss_weights

    def forward(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        individual_losses = {}

        for name, weight in self.loss_weights.items():
            if weight == 0:
                individual_losses[name] = 0.0
                continue

            loss_value = self.loss_map[name](**kwargs)
            weighted_loss = weight * loss_value
            total_loss += weighted_loss
            individual_losses[name] = loss_value.item()

        return total_loss, individual_losses


class NBReconLoss(LossTerm):
    """negative binomial reconstruction loss"""

    def __init__(self, eps: float = 1e-8, gene_normalize: bool = False):
        super().__init__()
        self.eps = eps
        self.gene_normalize = gene_normalize

    def forward(self, x_raw: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute negative binomial reconstruction loss

        Args:
            x_raw: raw count data (batch_size, num_genes)
            mu: predicted mean (batch_size, num_genes)
            theta: predicted dispersion (batch_size, num_genes)

        Returns:
            negative binomial reconstruction loss
        """
        # add eps for numerical stability
        mu = mu + self.eps
        theta = theta + self.eps

        # compute negative binomial log likelihood
        log_likelihood = (
            torch.lgamma(theta + x_raw)
            - torch.lgamma(theta)
            - torch.lgamma(x_raw + 1)
            + theta * torch.log(theta / (theta + mu))
            + x_raw * torch.log(mu / (theta + mu))
        )

        if self.gene_normalize:
            # normalize per-gene log-likelihood by batch mean count so low and
            # high expression genes contribute equally to the gradient
            gene_scale = x_raw.detach().mean(dim=0).clamp(min=1.0)
            log_likelihood = log_likelihood / gene_scale.unsqueeze(0)

        # return negative log likelihood
        return -torch.mean(log_likelihood)


class Log1pMSEAuxLoss(LossTerm):
    """auxiliary mse loss in log1p normalized space for nb autoencoders

    computes mse between log1p(mu / L_total * target_sum) and x_input
    so the nb encoder is steered toward preserving perturbation direction
    in the evaluation space without changing the nb decoder
    """

    def forward(self, mu: torch.Tensor, x_input: torch.Tensor,
                total_lib_size: torch.Tensor, target_sum: float = 1e4, **kwargs) -> torch.Tensor:
        """compute mse in log1p space"""

        lib = total_lib_size.unsqueeze(-1).clamp(min=1.0)
        recon_log1p = torch.log1p(mu / lib * target_sum)
        return F.mse_loss(recon_log1p, x_input)


class MSEReconLoss(LossTerm):
    """mean squared reconstruction loss

    supports two forms of per-gene loss reweighting both set by the
    lightning module in on_fit_start from training data statistics

    gene_variance_normalize divides per-gene squared error by its training
    variance so every gene contributes equally regardless of its natural
    variance note that this down-weights high-variance perturbation-
    responsive genes and in practice hurts downstream metrics

    pert_weighted multiplies per-gene squared error by a binary or
    continuous mask that emphasizes genes which actually respond to
    perturbations on the training set this directly rewards the model
    for reconstructing perturbation-relevant structure
    """

    def __init__(
        self,
        gene_variance_normalize: bool = False,
        eps: float = 1e-6,
        floor_frac_of_mean: float = 0.1,
        pert_weighted: bool = False,
    ):
        super().__init__()
        self.gene_variance_normalize = bool(gene_variance_normalize)
        self.eps = float(eps)
        self.floor_frac_of_mean = float(floor_frac_of_mean)
        self.pert_weighted = bool(pert_weighted)
        self.register_buffer("gene_variance", torch.empty(0), persistent=True)
        self.register_buffer("gene_weights", torch.empty(0), persistent=True)

    def set_gene_variance(self, gene_variance: torch.Tensor) -> None:
        """set per gene training variance for normalization

        applies a floor at max(eps, floor_frac_of_mean * mean_variance) so
        dead genes do not dominate the loss via extreme weighting
        """

        gene_variance = gene_variance.detach().clone()
        mean_var = float(gene_variance.mean().item())
        floor = max(self.eps, self.floor_frac_of_mean * mean_var)
        self.gene_variance = gene_variance.clamp(min=floor)

    def set_gene_weights(self, gene_weights: torch.Tensor) -> None:
        """set per gene multiplicative weights for the squared error"""

        self.gene_weights = gene_weights.detach().clone()

    def forward(self, x_input: torch.Tensor, recon: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute mse reconstruction loss"""

        diff_sq = (recon - x_input) ** 2
        if self.gene_variance_normalize and self.gene_variance.numel() > 0:
            diff_sq = diff_sq / self.gene_variance.to(diff_sq.device).unsqueeze(0)
        if self.pert_weighted and self.gene_weights.numel() > 0:
            diff_sq = diff_sq * self.gene_weights.to(diff_sq.device).unsqueeze(0)
        return torch.mean(diff_sq)


class PullbackIsotropyLoss(LossTerm):
    """pullback isotropy loss for regularizing the decoder to be isotropic in the latent space

    uses batched vmap over cells and jvp over latent basis vectors to compute
    the negative binomial fisher information weighted pullback metric efficiently
    """

    def __init__(self, alpha_min: float = 1.0):
        """initialize pullback isotropy loss

        Args:
            alpha_min: minimum value for learned alpha scale factor
                prevents alpha from collapsing to zero which trivially
                satisfies the loss without actually flattening the geometry
        """

        super().__init__()
        self.alpha_min = float(alpha_min)

    def forward(self, model: nn.Module, z, n_genes_total=None, gene_subsample=None, **kwargs) -> torch.Tensor:
        """computes the flatvi flattening loss using batched jacobian-vector products

        Args:
            model: the autoencoder instance with decode_for_pullback and alpha
            z: latent vectors (batch_size, latent_dim)
            n_genes_total: total genes (used if subsampling) defaults to model.n_genes
            gene_subsample: number of genes to use for stochastic approximation
        """

        B, d_lat = z.shape
        device = z.device
        alpha = torch.clamp(model.alpha, min=self.alpha_min)

        if n_genes_total is None:
            n_genes_total = model.n_genes

        # gene selection for stochastic approximation
        if gene_subsample is not None and n_genes_total > gene_subsample:
            gene_idx = torch.randperm(n_genes_total, device=device)[:gene_subsample]
        else:
            gene_idx = None

        # batched FIM weights: one decode call for the whole batch
        with torch.no_grad():
            lib_ones = torch.ones(B, device=device)
            is_nb = hasattr(model, "family") and getattr(model, "family") == "negative_binomial"
            if is_nb:
                mu_all, theta_all = model.decode(z, lib_ones)
            else:
                mu_all = model.decode_for_pullback(z, lib_ones)
                theta_all = torch.ones_like(mu_all)

            if gene_idx is not None:
                mu_all = mu_all[:, gene_idx]
                theta_all = theta_all[:, gene_idx]

            sqrt_weights_all = torch.sqrt(
                theta_all / (mu_all * (mu_all + theta_all) + 1e-6)
            )  # (B, G)

        # single-sample decode for vmap
        def decode_single(z_single):
            lib = torch.ones(1, device=device)
            mu = model.decode_for_pullback(z_single.unsqueeze(0), lib).squeeze(0)
            if gene_idx is not None:
                return mu.index_select(0, gene_idx)
            return mu

        # compute JVPs for each latent basis vector, vmapped over cells
        def jvp_single(z_single, tangent_single):
            return func_jvp(decode_single, (z_single,), (tangent_single,))[1]

        U_cols = []
        for i in range(d_lat):
            tangent = torch.zeros(B, d_lat, device=device)
            tangent[:, i] = 1.0
            jvp_batch = vmap(jvp_single, randomness="different")(z, tangent)  # (B, G)
            U_cols.append(jvp_batch * sqrt_weights_all)

        # assemble pullback metric and compute frobenius loss
        U = torch.stack(U_cols, dim=1)       # (B, d_lat, G)
        G = U @ U.transpose(1, 2)            # (B, d_lat, d_lat)
        eye = torch.eye(d_lat, device=device)
        diff = G - alpha * eye.unsqueeze(0)
        return (diff ** 2).sum(dim=(1, 2)).mean()


class DistancePreservationLoss(LossTerm):
    """preserve pairwise distances in the latent space

    uses learned scale (model.alpha) so latent distances target
    alpha * d rather than forcing the latent space to match the
    absolute scale of the target geometry (eg phate)

    optional exponential weighting emphasizes local structure:
      w_ij = exp(-zeta * d_ij)
    so between-cluster pairs are downweighted
    """

    def __init__(self, zeta: float = 0.0, alpha_min: float = 1.0):
        """initialize distance preservation loss

        Args:
            zeta: exponential decay rate for distance weighting
                0 means uniform weighting
                >0 downweights distant pairs to focus on local structure
            alpha_min: minimum value for learned alpha scale factor
                forces the latent space to spread out at least this much
        """

        super().__init__()
        self.zeta = float(zeta)
        self.alpha_min = float(alpha_min)

    def forward(self, z: torch.Tensor, original_distances: torch.Tensor = None, model=None,
                cell_type_ids=None, distance_embeddings: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """compute distance preservation loss

        Args:
            z: latent vectors (batch_size, latent_dim)
            original_distances: precomputed upper triangle pairwise distance
                vector legacy path used when a 2d distance matrix was provided
                at dataset construction time
            distance_embeddings: (batch_size, d_embed) per cell target
                geometry vectors when provided pdist runs on gpu here and the
                loss iterates per cell type so only within cell type pairs
                are ever computed both eliminates the cpu pdist bottleneck
                and skips work for cross cell type pairs that same cell type
                masking would discard
            model: autoencoder with learnable alpha scale parameter
            cell_type_ids: integer cell type labels per cell in the batch
                when provided only same-cell-type pairs contribute to loss
        """

        raw_alpha = model.alpha if model is not None else 1.0
        scale = torch.clamp(raw_alpha, min=self.alpha_min)

        # per cell type pdist on gpu avoids computing cross cell type pairs
        # that would be masked out anyway
        if distance_embeddings is not None and cell_type_ids is not None:
            unique_cts = torch.unique(cell_type_ids)
            sum_weighted_residuals = z.new_zeros(())
            pair_count = 0
            for ct in unique_cts:
                mask = cell_type_ids == ct
                n_c = int(mask.sum().item())
                if n_c < 2:
                    continue
                z_ct = z[mask]
                emb_ct = distance_embeddings[mask]
                d_z = torch.pdist(z_ct)
                d_p = torch.pdist(emb_ct)
                r = (d_z - scale * d_p) ** 2
                if self.zeta > 0:
                    r = torch.exp(-self.zeta * d_p) * r
                sum_weighted_residuals = sum_weighted_residuals + r.sum()
                pair_count += r.numel()
            if pair_count == 0:
                return torch.tensor(0.0, device=z.device, requires_grad=True)
            return sum_weighted_residuals / pair_count

        # legacy path with precomputed distances vector still mask cross cell
        # type pairs after the fact
        latent_distances = torch.pdist(z)
        if original_distances is None and distance_embeddings is not None:
            original_distances = torch.pdist(distance_embeddings)
        scaled_targets = scale * original_distances
        residuals = (latent_distances - scaled_targets) ** 2
        if self.zeta > 0:
            weights = torch.exp(-self.zeta * original_distances)
            residuals = weights * residuals

        if cell_type_ids is not None:
            n = z.size(0)
            row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=z.device)
            same_type = cell_type_ids[row_idx] == cell_type_ids[col_idx]
            if same_type.any():
                return torch.mean(residuals[same_type])
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        return torch.mean(residuals)


class CosineDistancePreservationLoss(LossTerm):
    """scale-invariant localized distance preservation loss

    uses cosine similarity on zeta-weighted distance vectors instead of MSE
    so that the loss only constrains the relative structure of the latent
    space without penalizing its absolute scale

    the loss is:
      W = exp(-zeta * D_P)
      D_P_tilde = W * D_P
      D_Z_tilde = W * D_Z
      L = 1 - cos(D_P_tilde, D_Z_tilde)

    this decouples the geometric loss from reconstruction loss: the encoder
    is free to inflate the latent space to whatever variance the decoder
    needs while the cosine term preserves the local neighborhood structure
    """

    def __init__(self, zeta: float = 0.0):
        """initialize cosine distance preservation loss

        Args:
            zeta: exponential decay rate for distance weighting
                0 means uniform weighting (global structure preserved)
                >0 focuses on local neighborhood structure
        """

        super().__init__()
        self.zeta = float(zeta)

    def forward(self, z: torch.Tensor, original_distances: torch.Tensor,
                cell_type_ids=None, **kwargs) -> torch.Tensor:
        """compute scale-invariant localized distance preservation loss

        Args:
            z: latent vectors (batch_size, latent_dim)
            original_distances: pairwise distances in the target space
            cell_type_ids: integer cell type labels per cell in the batch
                when provided only same-cell-type pairs contribute to loss
        """

        latent_distances = torch.pdist(z)

        # select pairs to use
        if cell_type_ids is not None:
            n = z.size(0)
            row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=z.device)
            same_type = cell_type_ids[row_idx] == cell_type_ids[col_idx]
            if not same_type.any():
                return torch.tensor(0.0, device=z.device, requires_grad=True)
            d_p = original_distances[same_type]
            d_z = latent_distances[same_type]
        else:
            d_p = original_distances
            d_z = latent_distances

        # zeta weighting to focus on local structure
        if self.zeta > 0:
            weights = torch.exp(-self.zeta * d_p)
            d_p_weighted = weights * d_p
            d_z_weighted = weights * d_z
        else:
            d_p_weighted = d_p
            d_z_weighted = d_z

        # cosine similarity between weighted distance vectors
        dot = torch.dot(d_p_weighted, d_z_weighted)
        norm_p = torch.norm(d_p_weighted)
        norm_z = torch.norm(d_z_weighted)
        denom = norm_p * norm_z

        if denom < 1e-12:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        cosine_sim = dot / denom
        return 1.0 - cosine_sim


class PredictorCrossEntropyLoss(LossTerm):
    """cross-entropy loss for auxiliary predictor heads

    parameterized by kwarg names so multiple instances can coexist
    in the same LossComposer (each reads different kwargs)
    """

    def __init__(self, logits_key: str = "logits", targets_key: str = "targets",
                 label_smoothing: float = 0.0):
        super().__init__()
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.label_smoothing = label_smoothing

    def forward(self, **kwargs) -> torch.Tensor:
        """compute cross-entropy loss"""

        logits = kwargs.get(self.logits_key)
        targets = kwargs.get(self.targets_key)
        if logits is None or targets is None:
            return torch.tensor(0.0, requires_grad=True)
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)


class OroJaRLoss(LossTerm):
    """orthogonal jacobian regularization loss

    forces the jacobians of two predictor networks (context and state)
    to be orthogonal in latent space using stochastic trace estimation
    via rademacher random projections

    this creates a fiber bundle geometry where cell-type directions
    are perpendicular to perturbation-response directions
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor, logits_context: torch.Tensor,
                logits_state: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute orojar penalty

        Args:
            z: latent vectors (batch_size, latent_dim) must have requires_grad
            logits_context: cell-type predictor output (batch_size, n_cell_types)
            logits_state: perturbation predictor output (batch_size, n_perturbations)
        """

        if logits_context is None or logits_state is None:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        # requires computation graph for autograd.grad — skip during eval
        if not z.requires_grad or z.grad_fn is None:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # rademacher random projections
        v_context = torch.randn_like(logits_context).sign()
        v_state = torch.randn_like(logits_state).sign()

        # project to scalar for autograd
        scalar_context = torch.sum(logits_context * v_context)
        scalar_state = torch.sum(logits_state * v_state)

        # compute gradients wrt z
        grad_context = torch.autograd.grad(
            outputs=scalar_context,
            inputs=z,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_state = torch.autograd.grad(
            outputs=scalar_state,
            inputs=z,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # cosine similarity between gradient vectors per sample
        dot_product = torch.sum(grad_context * grad_state, dim=1)
        norm_context = torch.norm(grad_context, dim=1) + self.eps
        norm_state = torch.norm(grad_state, dim=1) + self.eps
        cosine_sim = dot_product / (norm_context * norm_state)

        return torch.mean(cosine_sim ** 2)


class FlowMatchingMSELoss(LossTerm):
    """loss term for flow matching with mean squared error"""

    def forward(self, pred_v, target_v, **kwargs) -> torch.Tensor:
        """compute flow matching loss

        Args:
            pred_v: predicted velocity vector (batch_size, latent_dim)
            target_v: target velocity vector (batch_size, latent_dim)

        Returns:
            mean squared error between predicted and target velocities
        """
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss


class FlowMatchingWeightedMSELoss(LossTerm):
    """variance-weighted flow matching mse loss

    weights each dimension by its pca explained variance ratio so the model
    focuses on signal-rich components rather than spending equal effort on
    noisy low-variance directions
    """

    def __init__(self, dimension_weights: list[float] | None = None):
        super().__init__()
        if dimension_weights is not None:
            w = torch.as_tensor(dimension_weights, dtype=torch.float32)
            # normalize so weights sum to number of dimensions (preserves loss scale)
            w = w * (len(w) / w.sum())
            self.register_buffer("weights", w)
        else:
            self.weights = None

    def forward(self, pred_v, target_v, **kwargs) -> torch.Tensor:
        """compute variance-weighted flow matching loss

        Args:
            pred_v: predicted velocity vector (batch_size, latent_dim)
            target_v: target velocity vector (batch_size, latent_dim)

        Returns:
            weighted mean squared error between predicted and target velocities
        """
        sq_err = (pred_v - target_v) ** 2
        if self.weights is not None:
            sq_err = sq_err * self.weights.to(sq_err.device)
        return sq_err.mean()


class SphereMSELoss(LossTerm):
    """mse loss for tangent vectors on the sphere

    uses mean-over-batch of sum-over-features instead of mean-over-all
    this avoids the loss being divided by D (the number of genes)
    which would make it vanishingly small for high-dimensional simplices
    """

    def forward(self, pred_v, target_v, **kwargs) -> torch.Tensor:
        """compute sphere-aware mse loss

        Args:
            pred_v: predicted tangent velocity (batch_size dim)
            target_v: target tangent velocity (batch_size dim)

        Returns:
            mean per-sample squared norm of velocity error
        """
        return torch.mean(torch.sum((pred_v - target_v) ** 2, dim=-1))


class OTLoss(LossTerm):
    """loss term for optimal transport distance between predicted and target distributions

    uses geomloss sinkhorn divergence (debiased) instead of raw entropic ot
    debias=True subtracts the self-transport terms so S(mu, mu) = 0 and
    the loss does not reward mode collapse from the entropic bias
    """

    def __init__(
        self,
        sinkhorn_reg: float = 0.1,
        sinkhorn_max_iter: int = 50,
        debias: bool = True,
    ):
        """initialize otloss

        Args:
            sinkhorn_reg: entropic regularization strength (epsilon in sinkhorn)
                          geomloss uses blur = sqrt(epsilon) under the hood
            sinkhorn_max_iter: retained for config back-compat geomloss uses
                               its own multiscale schedule instead
            debias: if true use sinkhorn divergence (debiased) if false use
                    raw entropic ot which has sinkhorn bias toward low-entropy
                    coupling and can cause mode collapse
        """
        super().__init__()
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        self.debias = bool(debias)
        # blur is geomloss naming for sqrt(epsilon) so the scale of the
        # regularization in cost units matches sinkhorn_reg
        blur = max(self.sinkhorn_reg, 1e-4) ** 0.5
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur,
            debias=self.debias,
            backend="tensorized",
        )

    def forward(self, x_pred, x_target, **kwargs) -> torch.Tensor:
        # geomloss expects point clouds (not a cost matrix) and uses its own
        # internal log-domain sinkhorn so the manual cost_matrix path is unused
        x_pred32 = x_pred.to(torch.float32)
        x_target32 = x_target.to(torch.float32)
        value = self.sinkhorn(x_pred32, x_target32)
        value = value.to(dtype=x_pred.dtype)
        if torch.isfinite(value):
            return value

        # fallback to differentiable nearest-neighbor transport surrogate if
        # sinkhorn diverges eg on pathological inputs
        M = torch.cdist(x_pred, x_target) ** 2
        M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=0.0)
        nearest_cost = M.min(dim=1).values.mean() + M.min(dim=0).values.mean()
        return 0.5 * nearest_cost


class DensityLoss(LossTerm):
    """encourages points to be close to target distribution
    uses hinge loss on k-nearest neighbor distances
    based on https://github.com/xingzhis/mioflow-lite
    """

    def __init__(self, top_k: int = 5, hinge_value: float = 1.0):
        super().__init__()
        self.top_k = int(top_k)
        self.hinge_value = float(hinge_value)

    def forward(self, x_pred, x_target, cost_matrix=None, **kwargs) -> torch.Tensor:
        if cost_matrix is not None:
            # cost_matrix is squared distances; DensityLoss needs unsquared
            c_dist = torch.sqrt(cost_matrix)
        else:
            c_dist = torch.cdist(x_pred, x_target)
        values, _ = torch.topk(c_dist, self.top_k, dim=1, largest=False, sorted=False)
        values = torch.clamp(values - self.hinge_value, min=0.0)
        return torch.mean(values)


class EnergyLoss(LossTerm):
    """penalizes $\\int ||v(x_t, t)||^2 dt$ to encourage simpler vector fields

    prefers kinetic_energy from augmented ode integration which uses
    the solver's internal steps for an exact integral
    falls back to finite differences on a saved trajectory for backward compat
    """

    def forward(
        self,
        kinetic_energy: torch.Tensor | None = None,
        x_trajectory: torch.Tensor | None = None,
        t_span: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """compute energy loss

        Args:
            kinetic_energy: (b) tensor of accumulated per-cell kinetic energy
                from augmented ode integration
            x_trajectory: (t b d) tensor of states (finite-diff fallback)
            t_span: (t) tensor of time points (finite-diff fallback)
        """

        if kinetic_energy is not None:
            return torch.mean(kinetic_energy)
        # finite-difference fallback
        dt = t_span[1:] - t_span[:-1]
        dx = x_trajectory[1:] - x_trajectory[:-1]
        v_approx = dx / dt[:, None, None]
        return torch.mean(torch.sum(v_approx ** 2, dim=-1))


class MeanFlowIdentityLoss(LossTerm):
    """meanflow identity loss with optional adaptive reweighting

    base loss:
      l_id = ||u_theta - (v - (t-r) * sg(du_dt))||^2

    optional adaptive weighting (paper-inspired):
      w = ||u_theta - v|| / ||u_target - v||
      l = w * ||u_theta - u_target||^2

    for the special case `r == t` `u_target == v` so the denominator is
    exactly zero in that branch this implementation falls back to `w = 1`
    """

    def __init__(
        self,
        adaptive_weighting: bool = True,
        adaptive_power: float = 1.0,
        adaptive_eps: float = 1e-6,
        adaptive_clip_min: float = 0.1,
        adaptive_clip_max: float = 10.0,
        equal_time_eps: float = 1e-8,
    ):
        """initialize meanflowidentityloss

        Args:
            adaptive_weighting: if true apply adaptive weighting to residuals
            adaptive_power: exponent applied to the adaptive ratio
            adaptive_eps: numerical epsilon for denominator stabilization
            adaptive_clip_min: minimum clipped adaptive weight
            adaptive_clip_max: maximum clipped adaptive weight
            equal_time_eps: threshold for identifying `r == t` rows
        """
        super().__init__()
        self.adaptive_weighting = adaptive_weighting
        self.adaptive_power = adaptive_power
        self.adaptive_eps = adaptive_eps
        self.adaptive_clip_min = adaptive_clip_min
        self.adaptive_clip_max = adaptive_clip_max
        self.equal_time_eps = equal_time_eps

    def forward(self, u_theta, v, du_dt, t, r, **kwargs) -> torch.Tensor:
        """compute meanflow identity loss

        Args:
            u_theta: predicted average velocity (batch_size, dim)
            v: ground truth velocity x_1 - x_0 (batch_size, dim)
            du_dt: time derivative of u_theta (batch_size, dim) detached
            t: end time (batch_size)
            r: start time (batch_size)

        Returns:
            meanflow identity loss (optionally adaptively weighted)
        """
        t_minus_r = (t - r).unsqueeze(-1)  # (batch_size, 1)
        u_target = v - t_minus_r * du_dt.detach()
        residual_sq = (u_theta - u_target) ** 2

        if not self.adaptive_weighting:
            return torch.mean(residual_sq)

        with torch.no_grad():
            weights = torch.ones_like(t)
            non_equal_mask = (t - r).abs() > self.equal_time_eps

            if torch.any(non_equal_mask):
                num = torch.norm((u_theta - v)[non_equal_mask], dim=-1)
                den = torch.norm((u_target - v)[non_equal_mask], dim=-1).clamp_min(self.adaptive_eps)
                ratio = (num / den).pow(self.adaptive_power)
                ratio = torch.clamp(ratio, min=self.adaptive_clip_min, max=self.adaptive_clip_max)
                weights[non_equal_mask] = ratio

        return torch.mean(weights.unsqueeze(-1) * residual_sq)


class EnergyDistanceLoss(LossTerm):
    """energy distance: 2*e[||x-y||] - e[||x-x'||] - e[||y-y'||]

    distributional loss comparing one-step predictions to target samples
    """

    def __init__(self, max_samples: int = 512):
        """initialize energydistanceloss

        Args:
            max_samples: max samples to use per batch (subsample to avoid oom)
        """
        super().__init__()
        self.max_samples = max_samples

    def forward(self, x_pred_one_step, x_target, **kwargs) -> torch.Tensor:
        """compute energy distance between predicted and target distributions

        Args:
            x_pred_one_step: one-step predicted samples (n_pred, dim)
            x_target: target perturbed samples (n_target, dim)

        Returns:
            energy distance scalar
        """
        # Subsample if needed
        n_pred = x_pred_one_step.size(0)
        n_target = x_target.size(0)

        if n_pred > self.max_samples:
            idx = torch.randperm(n_pred, device=x_pred_one_step.device)[:self.max_samples]
            x_pred_one_step = x_pred_one_step[idx]

        if n_target > self.max_samples:
            idx = torch.randperm(n_target, device=x_target.device)[:self.max_samples]
            x_target = x_target[idx]

        # E[||x - y||]
        cross = torch.cdist(x_pred_one_step, x_target)
        e_cross = cross.mean()

        # E[||x - x'||]
        self_pred = torch.cdist(x_pred_one_step, x_pred_one_step)
        e_self_pred = self_pred.mean()

        # E[||y - y'||]
        self_target = torch.cdist(x_target, x_target)
        e_self_target = self_target.mean()

        return 2.0 * e_cross - e_self_pred - e_self_target
