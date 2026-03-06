"""
composable loss registry
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from models import NBAutoEncoder
import ot


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

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

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

        # return negative log likelihood
        return -torch.mean(log_likelihood)


class PullbackIsotropyLoss(LossTerm):
    """pullback isotropy loss for regularizing the learned vector field to be isotropic in the latent space"""

    def forward(self, model: NBAutoEncoder, z, n_genes_total=None, gene_subsample=None, **kwargs) -> torch.Tensor:
        """computes the flatvi regularization loss using jacobian-vector products (jvp)

        Args:
            model: the nbautoenoder instance
            z: latent vectors (batch latent_dim)
            n_genes_total: total genes (used if subsampling) defaults to model.n_genes
            gene_subsample: number of genes to use for stochastic approximation
        """
        B, d_lat = z.shape
        device = z.device

        # Use model's alpha parameter
        alpha = model.alpha

        # Default to model's gene count if not provided
        if n_genes_total is None:
            n_genes_total = model.n_genes

        # 1. Gene Selection (Stochastic Approximation)
        if gene_subsample is not None and n_genes_total > gene_subsample:
            gene_idx = torch.randperm(n_genes_total, device=device)[:gene_subsample]
        else:
            gene_idx = None

        # Helper: Decode z -> mu (NB Mean) for JVP
        # Fix library size to 1.0 to capture intrinsic manifold geometry, not read depth.
        def get_decoded_mu(z_vec):
            z_in = z_vec.unsqueeze(0)  # (1, d_lat)
            # Pass dummy library size of 1.0
            lib_dummy = torch.ones(1, device=device)
            mu, _ = model.decode(z_in, lib_dummy)

            mu_squeezed = mu.squeeze(0)
            if gene_idx is not None:
                return mu_squeezed.index_select(0, gene_idx)
            return mu_squeezed

        eye = torch.eye(d_lat, device=device)
        basis = torch.eye(d_lat, device=device)
        losses = []

        for b in range(B):
            z_b = z[b]

            # 2. Compute FIM Weights (w_g)
            # We need values at the current point z_b to weight the gradients.
            with torch.no_grad():
                lib_dummy = torch.ones(1, device=device)
                mu_full, theta_full = model.decode(z_b.unsqueeze(0), lib_dummy)
                mu_b = mu_full.squeeze(0)
                theta_b = theta_full.squeeze(0)

                if gene_idx is not None:
                    mu_b = mu_b.index_select(0, gene_idx)
                    theta_b = theta_b.index_select(0, gene_idx)

                # Fisher Information Weights for Negative Binomial [cite: 180, 589]
                # w = theta / (mu * (mu + theta))
                weights = theta_b / (mu_b * (mu_b + theta_b) + 1e-6)
                sqrt_weights = torch.sqrt(weights)

            # 3. Compute Jacobian-Vector Products
            # Instead of instantiating the full (Genes x Latent) Jacobian, we compute
            # column vectors directly.
            U_cols = []
            for i in range(d_lat):
                v_i = basis[i]  # Standard basis vector e_i

                # d(mu)/d(z) * e_i
                _, jvp_mu = torch.autograd.functional.jvp(get_decoded_mu, z_b, v_i, create_graph=True)

                # Apply sqrt(Fisher Weights)
                weighted_jvp = jvp_mu * sqrt_weights
                U_cols.append(weighted_jvp)

            # 4. Assemble the Pullback Metric
            # U is (d_lat, n_genes_sub)
            U = torch.stack(U_cols, dim=0)

            # G = U @ U.T corresponds to J^T * diag(w) * J
            G = U @ U.T

            # 5. Isotropy Loss [cite: 160]
            # Force metric G to resemble alpha * Identity
            losses.append(((G - alpha * eye) ** 2).sum())

        return torch.stack(losses).mean()


class DistancePreservationLoss(LossTerm):
    """loss term to preserve pairwise distances in the latent space"""

    def forward(self, z: torch.Tensor, original_distances: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute distance preservation loss

        Args:
            z: latent vectors (batch_size, latent_dim)
            original_distances: pairwise distances in the original space (batch_size * (batch_size - 1) / 2)

        Returns:
            distance preservation loss
        """
        # compute pairwise distances in the latent space
        latent_distances = torch.pdist(z)

        # compute mean squared error between original and latent distances
        loss = torch.mean((latent_distances - original_distances) ** 2)
        return loss


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


class OTLoss(LossTerm):
    """loss term for optimal transport distance between predicted and target distributions
    based on https://github.com/xingzhis/mioflow-lite

    uses sinkhorn divergence (gpu-native) instead of exact emd (cpu-bound)
    """

    def __init__(self, sinkhorn_reg: float = 0.1, sinkhorn_max_iter: int = 50):
        """initialize otloss

        Args:
            sinkhorn_reg: entropic regularization strength for sinkhorn
            sinkhorn_max_iter: maximum sinkhorn iterations
        """
        super().__init__()
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_max_iter = sinkhorn_max_iter

    def forward(self, x_pred, x_target, cost_matrix=None, **kwargs) -> torch.Tensor:
        mu = torch.tensor(ot.unif(x_pred.size()[0]), dtype=x_pred.dtype, device=x_pred.device)
        nu = torch.tensor(ot.unif(x_target.size()[0]), dtype=x_target.dtype, device=x_target.device)
        if cost_matrix is not None:
            M = cost_matrix
        else:
            M = torch.cdist(x_pred, x_target) ** 2
        # median-normalize for numerical stability
        med = M.median()
        if med > 0:
            M = M / med
        return ot.bregman.sinkhorn2(
            mu, nu, M, reg=self.sinkhorn_reg, numItermax=self.sinkhorn_max_iter
        )


class DensityLoss(LossTerm):
    """encourages points to be close to target distribution
    uses hinge loss on k-nearest neighbor distances
    based on https://github.com/xingzhis/mioflow-lite
    """

    def forward(self, x_pred, x_target, cost_matrix=None, top_k=5, hinge_value=1.0, **kwargs) -> torch.Tensor:
        if cost_matrix is not None:
            # cost_matrix is squared distances; DensityLoss needs unsquared
            c_dist = torch.sqrt(cost_matrix)
        else:
            c_dist = torch.cdist(x_pred, x_target)
        values, _ = torch.topk(c_dist, top_k, dim=1, largest=False, sorted=False)
        values = torch.clamp(values - hinge_value, min=0.0)
        return torch.mean(values)


class EnergyLoss(LossTerm):
    """penalizes $\\int ||v(x_t, t)||^2$ to encourage simpler vector fields

    uses finite differences on the trajectory instead of extra forward passes:
    v ≈ (x_{t+1} - x_t) / dt
    """

    def forward(
        self,
        x_trajectory: torch.Tensor,
        t_span: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """compute energy loss along a trajectory using finite differences

        Args:
            x_trajectory: (t b d) tensor of states along the trajectory
            t_span: (t) tensor of time points
        """
        # dt between consecutive time steps: (T-1,)
        dt = t_span[1:] - t_span[:-1]
        # finite-difference velocities: (T-1, B, D)
        dx = x_trajectory[1:] - x_trajectory[:-1]
        v_approx = dx / dt[:, None, None]
        # mean ||v||^2 across time, batch, and dims
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
