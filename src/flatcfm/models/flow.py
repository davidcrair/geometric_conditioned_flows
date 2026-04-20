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
from flatcfm.data.types import ConditionBatch
from flatcfm.data.dataset import condition_batch_to_device


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        covariate_dicts: dict,
        condition_dim: int = 64,
        embedding_dim: int = 64,
        projection_dim: int = 64,
        pretrained_perturbation_embeddings: Tensor | None = None,
        embedding_dropout: float = 0.0,
        perturbation_dropout: float = 0.0,
        sample_covariate_dropout: float = 0.0,
    ):
        """conditioning encoder with optional regularizers

        embedding_dropout
            dropout applied to each embedding projection output before concat
            same idea as standard dropout but targeted at the conditioning
            path so backbone capacity stays untouched

        perturbation_dropout
            prob of zeroing the entire perturbation projection during training
            forces the downstream model to handle a null drug token which
            regularizes the learned (perturbation x cell_type) interactions
            that drive the held out cosine_log_fc collapse

        sample_covariate_dropout
            same token zeroing applied to each sample covariate (cell_type)
            the most direct attack on the "memorize response per cell line"
            failure mode for underrepresented context holdouts
        """

        super().__init__()

        self.pert_cov_keys = list(covariate_dicts.get("perturbation_covariates", {}).keys())
        self.sample_cov_keys = list(covariate_dicts.get("sample_covariates", {}).keys())
        self.embedding_dropout_p = float(embedding_dropout)
        self.perturbation_dropout_p = float(perturbation_dropout)
        self.sample_covariate_dropout_p = float(sample_covariate_dropout)
        self._emb_dropout = nn.Dropout(self.embedding_dropout_p) if self.embedding_dropout_p > 0 else nn.Identity()

        num_perts = covariate_dicts.get("perturbation_num_categories", 2)
        # when pretrained embeddings are provided use them as a frozen lookup
        # table so held out drugs with the same molecular structure share
        # representation with training drugs instead of collapsing to a
        # randomly initialized row of an nn.Embedding
        if pretrained_perturbation_embeddings is not None:
            if pretrained_perturbation_embeddings.shape[0] != num_perts:
                raise ValueError(
                    f"pretrained perturbation embeddings have {pretrained_perturbation_embeddings.shape[0]} rows "
                    f"but vocab has {num_perts} perturbations"
                )
            self.emb_pert = nn.Embedding.from_pretrained(
                pretrained_perturbation_embeddings.float(),
                freeze=True,
            )
            self._pert_input_dim = int(pretrained_perturbation_embeddings.shape[1])
        else:
            self.emb_pert = nn.Embedding(num_embeddings=num_perts, embedding_dim=embedding_dim)
            self._pert_input_dim = embedding_dim

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

        self.proj_perturbation = nn.Sequential(nn.Linear(self._pert_input_dim, projection_dim), nn.ReLU())

        self.proj_pert_cov = nn.ModuleDict(
            {key: nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.ReLU()) for key in self.pert_cov_keys}
        )

        self.proj_sample_cov = nn.ModuleDict(
            {key: nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.ReLU()) for key in self.sample_cov_keys}
        )

        num_inputs = 1 + len(self.pert_cov_keys) + len(self.sample_cov_keys)
        concat_dim = num_inputs * projection_dim

        self.output_mlp = nn.Sequential(
            nn.Linear(concat_dim, condition_dim), nn.ReLU(), nn.Linear(condition_dim, condition_dim)
        )

        self.output_dim = condition_dim

    def _maybe_zero_token(self, projection: Tensor, prob: float) -> Tensor:
        """zero the whole projection per sample with a given probability

        only active during training and only when prob > 0 applies the same
        mask to every feature dim of a sample so one token is either fully
        present or fully zero like classifier free guidance dropout
        """

        if not self.training or prob <= 0.0:
            return projection
        keep_mask = (torch.rand(projection.shape[0], 1, device=projection.device) >= prob).to(projection.dtype)
        return projection * keep_mask

    def forward(self, cond_batch: dict) -> Tensor:
        features_to_concat = []

        p_emb = self.emb_pert(cond_batch["perturbations"])
        p_proj = self.proj_perturbation(p_emb)
        p_proj = self._maybe_zero_token(p_proj, self.perturbation_dropout_p)
        p_proj = self._emb_dropout(p_proj)
        features_to_concat.append(p_proj)

        for key in self.pert_cov_keys:
            indices = cond_batch[f"perturbation_covariates"][key]
            emb = self.emb_pert_cov[key](indices)
            proj = self.proj_pert_cov[key](emb)
            proj = self._emb_dropout(proj)
            features_to_concat.append(proj)

        for key in self.sample_cov_keys:
            indices = cond_batch["sample_covariates"][key]
            emb = self.emb_sample_cov[key](indices)
            proj = self.proj_sample_cov[key](emb)
            proj = self._maybe_zero_token(proj, self.sample_covariate_dropout_p)
            proj = self._emb_dropout(proj)
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


class FiLMLayer(nn.Module):
    """feature-wise linear modulation: condition produces per-layer scale and shift"""

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.linear.bias.data[hidden_dim:] = 1.0

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        params = self.linear(cond)
        shift, scale = params.chunk(2, dim=-1)
        return h * scale + shift


class CondFlow(nn.Module):
    """referencing lipman et al 2024 "flow matching guide and code"""

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 4096,
        hidden_layers: int = 3,
        output_dim: int = 100,
        cond_encoder: Optional[ConditionEncoder] = None,
        conditioning: str = "concat",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conditioning = conditioning

        if cond_encoder is not None:
            self.cond_encoder = cond_encoder
            self.cond_dim = cond_encoder.output_dim
        else:
            raise ValueError("cond_encoder must be provided.")

        self.time_embed = GaussianFourierProjection(embedding_size=64)

        if conditioning == "film":
            self.input_layer = nn.Linear(input_dim + 64, hidden_dim)
            self.film_layers = nn.ModuleList(
                [FiLMLayer(self.cond_dim, hidden_dim) for _ in range(hidden_layers)]
            )
        else:
            self.input_layer = nn.Linear(input_dim + 64 + self.cond_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x_t: Tensor, t: Tensor, cond_batch: ConditionBatch) -> Tensor:
        t_emb = self.time_embed(t)
        cond_emb = self.cond_encoder(cond_batch)

        if self.conditioning == "film":
            h = torch.cat([x_t, t_emb], dim=-1)
            h = self.activation(self.input_layer(h))
            for layer, film in zip(self.hidden_layers, self.film_layers):
                identity = h
                out = layer(h)
                out = self.activation(out)
                out = film(out, cond_emb)
                out = self.dropout(out)
                h = out + identity
        else:
            h = torch.cat([x_t, t_emb, cond_emb], dim=-1)
            h = self.activation(self.input_layer(h))
            for layer in self.hidden_layers:
                identity = h
                out = layer(h)
                out = self.activation(out)
                out = self.dropout(out)
                h = out + identity

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
    """wrapper for condflow to make it compatible with torchdiffeq.odeint

    when augmented=True the state is (x, energy) where d(energy)/dt = ||v||^2
    so the solver integrates the kinetic energy at every internal step
    """

    def __init__(self, flow_model: CondFlow, cond_batch: ConditionBatch, device: torch.device,
                 augmented: bool = False):
        super().__init__()
        self.flow = flow_model
        self.cond_batch = condition_batch_to_device(cond_batch, device)
        self.device = device
        self.augmented = augmented

    def forward(self, t, state):
        x = state[0] if self.augmented else state
        t_batch = t.expand(x.shape[0]).to(device=x.device, dtype=x.dtype)
        v = self.flow.forward(x, t_batch, self.cond_batch)
        if self.augmented:
            return (v, torch.sum(v ** 2, dim=-1))
        return v


def sample_ode(flow, x_init, cond_batch, method="rk4", time: float = 1.0):
    """methods: 'dopri5' (rk45 adaptive) 'rk4' 'midpoint' 'euler'"""

    ode_func = CondFlowODE(flow, cond_batch, device=x_init.device)
    t_span = torch.linspace(0, time, 50, device=x_init.device, dtype=x_init.dtype)
    trajectory = odeint(ode_func, x_init, t_span, method=method, rtol=1e-4, atol=1e-4)
    return trajectory[-1]
