"""
conditional flow, condition encoder, and neural ODE logic
"""

from torch import Tensor
import torch.nn as nn
from typing import Optional
import torch
from torchdiffeq import odeint
import torch.autograd.functional as F
import numpy as np
from data.types import ConditionBatch
from data.dataset import condition_batch_to_device


class ConditionEncoder(nn.Module):
    def __init__(
        self, covariate_dicts: dict, condition_dim: int = 64, embedding_dim: int = 64, projection_dim: int = 64
    ):
        super().__init__()

        self.pert_cov_keys = list(covariate_dicts.get("perturbation_covariates", {}).keys())
        self.sample_cov_keys = list(covariate_dicts.get("sample_covariates", {}).keys())

        num_perts = covariate_dicts.get("perturbation_num_categories", 2)
        self.emb_pert = nn.Embedding(num_embeddings=num_perts, embedding_dim=embedding_dim)

        self.emb_pert_cov = nn.ModuleDict(
            {
                key: nn.Embedding(
                    num_embeddings=covariate_dicts["perturbation_covariates"][key], embedding_dim=embedding_dim
                )
                for key in self.pert_cov_keys
            }
        )

        self.emb_sample_cov = nn.ModuleDict(
            {
                key: nn.Embedding(num_embeddings=covariate_dicts["sample_covariates"][key], embedding_dim=embedding_dim)
                for key in self.sample_cov_keys
            }
        )

        # pre-concatenation projection
        self.proj_perturbation = nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.ReLU())

        self.proj_pert_cov = nn.ModuleDict(
            {key: nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.ReLU()) for key in self.pert_cov_keys}
        )

        self.proj_sample_cov = nn.ModuleDict(
            {key: nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.ReLU()) for key in self.sample_cov_keys}
        )

        # aggregation mlp
        num_inputs = 1 + len(self.pert_cov_keys) + len(self.sample_cov_keys)
        concat_dim = num_inputs * projection_dim

        self.output_mlp = nn.Sequential(
            nn.Linear(concat_dim, condition_dim), nn.ReLU(), nn.Linear(condition_dim, condition_dim)
        )

        self.output_dim = condition_dim

    def forward(self, cond_batch: dict) -> Tensor:
        features_to_concat = []

        p_emb = self.emb_pert(cond_batch["perturbations"])
        p_proj = self.proj_perturbation(p_emb)
        features_to_concat.append(p_proj)

        for key in self.pert_cov_keys:
            indices = cond_batch[f"perturbation_covariates"][key]
            emb = self.emb_pert_cov[key](indices)
            proj = self.proj_pert_cov[key](emb)
            features_to_concat.append(proj)

        for key in self.sample_cov_keys:
            indices = cond_batch["sample_covariates"][key]
            emb = self.emb_sample_cov[key](indices)
            proj = self.proj_sample_cov[key](emb)
            features_to_concat.append(proj)

        concatenated = torch.cat(features_to_concat, dim=-1)
        return self.output_mlp(concatenated)


class GaussianFourierProjection(nn.Module):
    """gaussian fourier embeddings for noise levels/time"""

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class CondFlow(nn.Module):
    """referencing lipman et al 2024 "flow matching guide and code"""

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 4096,
        hidden_layers: int = 3,
        output_dim: int = 100,
        cond_encoder: Optional[ConditionEncoder] = None,
    ):
        super().__init__()

        if cond_encoder is not None:
            self.cond_encoder = cond_encoder
            self.cond_dim = cond_encoder.output_dim
        else:
            raise ValueError("cond_encoder must be provided.")

        self.time_embed = GaussianFourierProjection(embedding_size=64)

        # 1. Separate input/output layers from the hidden blocks
        self.input_layer = nn.Linear(input_dim + 64 + self.cond_dim, hidden_dim)

        # 2. Define hidden layers explicitly as a ModuleList
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ELU()

    def forward(self, x_t: Tensor, t: Tensor, cond_batch: ConditionBatch) -> Tensor:
        t_emb = self.time_embed(t)
        cond_emb = self.cond_encoder(cond_batch)
        # Concatenate inputs
        h = torch.cat([x_t, t_emb, cond_emb], dim=-1)

        # Input projection (no residual here, dimensions change)
        h = self.activation(self.input_layer(h))

        # Hidden layers WITH Residual Connections
        for layer in self.hidden_layers:
            # Save identity
            identity = h

            # Forward pass
            out = layer(h)
            out = self.activation(out)

            # Add residual (Skip Connection)
            h = out + identity

        # Output projection
        h = self.output_layer(h)
        return h

    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        cond_batch: ConditionBatch,
        dt: float,
    ) -> Tensor:
        velocity = self.forward(
            x_t,
            t,
            cond_batch,
        )
        x_next = x_t + velocity * dt
        return x_next


class CondFlowODE(nn.Module):
    """wrapper for condflow to make it compatible with torchdiffeq.odeint"""

    def __init__(self, flow_model: CondFlow, cond_batch: ConditionBatch, device: torch.device):
        super().__init__()
        self.flow = flow_model
        # Move cond_batch to device once
        self.cond_batch = condition_batch_to_device(cond_batch, device)
        self.device = device

    def forward(self, t, x):
        # t is a scalar from odeint
        t_batch = t.expand(x.shape[0]).to(device=x.device, dtype=x.dtype)
        return self.flow.forward(x, t_batch, self.cond_batch)


# Sampling with adaptive solver
def sample_ode(flow, x_init, cond_batch, method="rk4", time: float = 1.0):
    """methods: 'dopri5' (rk45 adaptive) 'rk4' 'midpoint' 'euler'"""
    ode_func = CondFlowODE(flow, cond_batch, device=x_init.device)
    t_span = torch.linspace(0, time, 50, device=x_init.device, dtype=x_init.dtype)
    trajectory = odeint(ode_func, x_init, t_span, method=method, rtol=1e-4, atol=1e-4)
    return trajectory[-1]  # Return final state
