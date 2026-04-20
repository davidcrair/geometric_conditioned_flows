"""autoencoder model families"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dim: int, n_layers: int, dropout: float, output_dim: int) -> nn.Sequential:
    """build mlp"""

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
    for _ in range(max(n_layers - 1, 0)):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class StandardAutoEncoder(nn.Module):
    """standard autoencoder"""

    family = "standard"

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        input_space_kind: str = "normalized_log1p",
        target_sum: float = 1e4,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.input_space_kind = str(input_space_kind)
        self.target_sum = float(target_sum)
        self.output_activation = str(output_activation)
        self.encoder = _build_mlp(n_genes, hidden_dim, n_layers, dropout, latent_dim)
        self.decoder = _build_mlp(latent_dim, hidden_dim, n_layers, dropout, n_genes)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def encode(self, x_input: torch.Tensor) -> torch.Tensor:
        """encode input"""

        return self.encoder(x_input)

    def decode(self, z: torch.Tensor, library_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        """decode latent"""

        del library_size
        decoded = self.decoder(z)
        if self.input_space_kind == "raw_counts":
            return decoded
        # getattr fallback lets pickles saved before output_activation was
        # added to the class still round-trip
        activation = getattr(self, "output_activation", "softplus")
        if activation == "relu":
            return torch.nn.functional.relu(decoded)
        return torch.nn.functional.softplus(decoded)

    def decode_for_pullback(self, z: torch.Tensor, library_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        """decode for pullback"""

        return self.decode(z, library_size=library_size)

    def reconstruct_input(
        self,
        z: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        input_library_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """reconstruct input space"""

        del input_library_size
        return self.decode(z, library_size=library_size)

    def reconstruct_counts(self, z: torch.Tensor, library_size: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """reconstruct counts"""

        del sample
        recon = self.decode(z, library_size=library_size)
        if self.input_space_kind == "raw_counts":
            return recon
        lib = library_size.unsqueeze(-1).clamp(min=1.0)
        return torch.expm1(recon).clamp(min=0.0) * lib / self.target_sum


class LinearAutoEncoder(nn.Module):
    """plain linear autoencoder

    two nn.Linear layers end-to-end with no nonlinearity no layernorm no
    dropout exists as a diagnostic baseline because linear autoencoders
    trained with MSE have a unique minimum modulo rotation that coincides
    with the top-k PCA subspace if this model matches PCA on held-out
    metrics then the non-linear AE plateau is driven by the nonlinearity
    not by capacity or loss weighting

    uses the same interface as StandardAutoEncoder so it drops into the
    modelcore without touching downstream code
    """

    family = "linear"

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        input_space_kind: str = "normalized_log1p",
        target_sum: float = 1e4,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.input_space_kind = str(input_space_kind)
        self.target_sum = float(target_sum)
        self.encoder = nn.Linear(self.n_genes, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.n_genes)
        # alpha parameter only exists to satisfy distance loss and
        # set_trainable_parts which access self.model.alpha unconditionally
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def encode(self, x_input: torch.Tensor) -> torch.Tensor:
        """encode input"""

        return self.encoder(x_input)

    def decode(self, z: torch.Tensor, library_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        """decode latent"""

        del library_size
        return self.decoder(z)

    def decode_for_pullback(self, z: torch.Tensor, library_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        """decode for pullback"""

        return self.decode(z, library_size=library_size)

    def reconstruct_input(
        self,
        z: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        input_library_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """reconstruct input space"""

        del input_library_size
        return self.decode(z, library_size=library_size)

    def reconstruct_counts(self, z: torch.Tensor, library_size: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """reconstruct counts"""

        del sample
        recon = self.decode(z, library_size=library_size)
        if self.input_space_kind == "raw_counts":
            return recon
        lib = library_size.unsqueeze(-1).clamp(min=1.0)
        return torch.expm1(recon).clamp(min=0.0) * lib / self.target_sum


class NegativeBinomialAutoEncoder(nn.Module):
    """negative binomial autoencoder"""

    family = "negative_binomial"

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        mean_head: str = "per_cell_gene",
        dispersion_mode: str = "shared_gene",
        input_space_kind: str = "normalized_log1p",
        target_sum: float = 1e4,
        n_cell_types: Optional[int] = None,
        n_perturbations: Optional[int] = None,
        rate_parameterization: str = "independent",
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.mean_head = str(mean_head)
        self.dispersion_mode = str(dispersion_mode)
        self.input_space_kind = str(input_space_kind)
        self.target_sum = float(target_sum)
        self.rate_parameterization = str(rate_parameterization)

        self.encoder = _build_mlp(n_genes, hidden_dim, n_layers, dropout, latent_dim)
        self.decoder_hidden = _build_mlp(latent_dim, hidden_dim, n_layers, dropout, hidden_dim)
        if self.mean_head != "per_cell_gene":
            raise ValueError(f"Unsupported mean head: {mean_head}")
        self.dec_log_rate = nn.Linear(hidden_dim, n_genes)
        nn.init.constant_(self.dec_log_rate.bias, -math.log(max(self.n_genes, 1)))
        if self.dispersion_mode == "shared_gene":
            self.log_theta = nn.Parameter(torch.zeros(n_genes))
            self.dec_log_theta = None
        elif self.dispersion_mode == "per_cell_gene":
            self.log_theta = None
            self.dec_log_theta = nn.Linear(hidden_dim, n_genes)
        else:
            raise ValueError(f"Unsupported dispersion mode: {dispersion_mode}")
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # auxiliary predictor heads for orojar
        if n_cell_types is not None and n_cell_types > 0:
            self.context_predictor = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(64, n_cell_types),
            )
        else:
            self.context_predictor = None
        if n_perturbations is not None and n_perturbations > 0:
            self.state_predictor = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(64, n_perturbations),
            )
        else:
            self.state_predictor = None

    def encode(self, x_input: torch.Tensor) -> torch.Tensor:
        """encode input"""

        return self.encoder(x_input)

    def _theta(self, hidden: torch.Tensor) -> torch.Tensor:
        """compute theta"""

        if self.dispersion_mode == "shared_gene":
            return self.log_theta.exp().clamp(min=1e-4, max=1e4)
        return torch.nn.functional.softplus(self.dec_log_theta(hidden)).clamp(min=1e-4, max=1e4)

    def decode(self, z: torch.Tensor, library_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """decode latent"""

        hidden = self.decoder_hidden(z)
        log_rate = self.dec_log_rate(hidden).clamp(min=-20.0, max=20.0)
        lib = library_size.unsqueeze(-1).clamp(min=0.0)
        if self.rate_parameterization == "softmax":
            mu = lib * torch.softmax(log_rate, dim=-1)
        else:
            mu = lib * torch.exp(log_rate)
        theta = self._theta(hidden)
        return mu, theta

    def decode_for_pullback(self, z: torch.Tensor, library_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        """decode for pullback"""

        if library_size is None:
            library_size = torch.ones(z.size(0), device=z.device)
        mu, _ = self.decode(z, library_size)
        return mu

    def reconstruct_input(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor,
        input_library_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """reconstruct input space"""

        mu, _ = self.decode(z, library_size)
        if self.input_space_kind == "raw_counts":
            return mu
        if input_library_size is None:
            input_library_size = library_size
        lib = input_library_size.unsqueeze(-1).clamp(min=1.0)
        return torch.log1p(mu / lib * self.target_sum)

    def reconstruct_counts(self, z: torch.Tensor, library_size: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """reconstruct counts"""

        mu, theta = self.decode(z, library_size)
        if not sample:
            return mu
        probs = mu / (mu + theta + 1e-8)
        return torch.distributions.NegativeBinomial(total_count=theta, probs=probs).sample()

    def nb_log_likelihood(self, x_raw: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """negative binomial log likelihood"""

        eps = 1e-8
        log_theta_mu = torch.log(theta + mu + eps)
        ll = (
            torch.lgamma(x_raw + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x_raw + 1)
            + theta * (torch.log(theta + eps) - log_theta_mu)
            + x_raw * (torch.log(mu + eps) - log_theta_mu)
        )
        return ll.sum(dim=-1)

    def loss(self, x_input: torch.Tensor, x_raw: torch.Tensor, library_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """loss"""

        z = self.encode(x_input)
        mu, theta = self.decode(z, library_size)
        recon_loss = -self.nb_log_likelihood(x_raw, mu, theta).mean()
        return recon_loss, z

    def forward(self, x_input: torch.Tensor, x_raw: torch.Tensor, library_size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """forward"""

        return self.loss(x_input, x_raw, library_size)

    def reconstruct_log_norm(self, raw_adata, sample: bool = True, device=None):
        """reconstruct log norm"""

        self.eval()
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device

            x = raw_adata.X
            x_dense = x.toarray() if hasattr(x, "toarray") else np.array(x)
            x_raw = torch.as_tensor(x_dense, dtype=torch.float32, device=device)
            library_size = x_raw.sum(dim=1)
            if self.input_space_kind == "raw_counts":
                x_input = x_raw
            else:
                x_input = torch.log1p(x_raw / library_size.unsqueeze(-1).clamp(min=1) * self.target_sum)

            z = self.encode(x_input)
            counts = self.reconstruct_counts(z, library_size, sample=sample)
            sample_lib = counts.sum(dim=-1, keepdim=True).clamp(min=1)
            return torch.log1p(counts / sample_lib * self.target_sum)


NBAutoEncoder = NegativeBinomialAutoEncoder
