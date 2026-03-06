"""
conditional meanflow model (gao et al "mean flows")

learns a mean velocity field u_θ(z_t, r, t) via a jvp-based identity loss
uses direct velocity parameterization (option a) with fm convention (t: 0→1)
"""

from torch import Tensor
import torch.nn as nn
from typing import Optional
import torch
from torchdiffeq import odeint
import numpy as np

from data.types import ConditionBatch
from data.dataset import condition_batch_to_device
from models.flow import ConditionEncoder, GaussianFourierProjection


class CondMeanFlow(nn.Module):
    """conditional meanflow: predicts average velocity u_θ(z_t, r, t, cond)

    uses standard (option a) parameterization for fm convention (t: 0→1)
    two time inputs (r, t) with separate gaussian fourier embeddings
    """

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 4096,
        hidden_layers: int = 3,
        output_dim: int = 100,
        cond_encoder: Optional[ConditionEncoder] = None,
    ):
        """initialize condmeanflow

        Args:
            input_dim: dimension of input z_t
            hidden_dim: width of hidden layers
            hidden_layers: number of residual hidden layers
            output_dim: dimension of output (average velocity)
            cond_encoder: condition encoder for perturbation context
        """
        super().__init__()

        if cond_encoder is not None:
            self.cond_encoder = cond_encoder
            self.cond_dim = cond_encoder.output_dim
        else:
            raise ValueError("cond_encoder must be provided.")

        self.r_embed = GaussianFourierProjection(embedding_size=64)
        self.t_embed = GaussianFourierProjection(embedding_size=64)

        # input_dim + 64 (r embed) + 64 (t embed) + cond_dim
        self.input_layer = nn.Linear(input_dim + 64 + 64 + self.cond_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ELU()

    def forward(self, z_t: Tensor, r: Tensor, t: Tensor, cond_batch: ConditionBatch) -> Tensor:
        """forward pass: predict average velocity u_θ

        Args:
            z_t: interpolated state (batch_size, input_dim)
            r: start time parameter (batch_size)
            t: end time parameter (batch_size)
            cond_batch: condition batch dict

        Returns:
            u_theta: average velocity (batch_size, output_dim)
        """
        r_emb = self.r_embed(r)
        t_emb = self.t_embed(t)
        cond_emb = self.cond_encoder(cond_batch)

        h = torch.cat([z_t, r_emb, t_emb, cond_emb], dim=-1)

        # Input projection
        h = self.activation(self.input_layer(h))

        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            identity = h
            out = layer(h)
            out = self.activation(out)
            h = out + identity

        # Output projection
        h = self.output_layer(h)
        return h

    def sample_one_step(self, x_0: Tensor, cond_batch: ConditionBatch) -> Tensor:
        """one-step sampling: x̂_1 = x_0 + u_θ(x_0, r=0, t=1, cond)

        Args:
            x_0: initial state / control cells (batch_size, input_dim)
            cond_batch: condition batch dict

        Returns:
            x_hat_1: predicted perturbed state (batch_size, output_dim)
        """
        batch_size = x_0.size(0)
        device = x_0.device
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        u_theta = self.forward(x_0, r, t, cond_batch)
        return x_0 + u_theta


class CondMeanFlowODE(nn.Module):
    """wrapper for torchdiffeq: dx/dt = u_θ(x, r=0, t, cond)"""

    def __init__(self, model: CondMeanFlow, cond_batch: ConditionBatch, device: torch.device):
        """initialize ode wrapper

        Args:
            model: condmeanflow model
            cond_batch: condition batch (moved to device once)
            device: torch device
        """
        super().__init__()
        self.model = model
        self.cond_batch = condition_batch_to_device(cond_batch, device)
        self.device = device

    def forward(self, t, x):
        """ode right-hand side

        Args:
            t: scalar time from odeint
            x: state tensor (batch_size, dim)

        Returns:
            velocity at (x r=0 t)
        """
        t_batch = t.expand(x.shape[0]).to(device=x.device, dtype=x.dtype)
        r_batch = torch.zeros_like(t_batch)
        return self.model.forward(x, r_batch, t_batch, self.cond_batch)


def sample_mean_flow(self, z_1: Tensor, cond_batch: ConditionBatch) -> Tensor:
    """one-step sampling mapping the prior to the data distribution

    Args:
        z_1: initial noise sampled from the prior (batch_size, input_dim)
        cond_batch: condition batch dict

    Returns:
        z_0: generated state in the data distribution
    """
    batch_size = z_1.size(0)
    device = z_1.device

    # Sampling integrates backward from t=1 to r=0
    r = torch.zeros(batch_size, device=device)
    t = torch.ones(batch_size, device=device)

    u_theta = self.forward(z_1, r, t, cond_batch)

    # The true 1-step generative formula: z_0 = z_1 - u(z_1, 0, 1)
    return z_1 - u_theta  # [cite: 193]
