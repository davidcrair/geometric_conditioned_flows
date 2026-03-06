"""
negative binomial autoencoder model and components
"""

from torch import nn
import torch
import numpy as np
from typing import Optional


class NBAutoEncoder(nn.Module):
    """autoencoder with negative binomial decoder for scrna-seq"""

    def __init__(
        self, n_genes: int, latent_dim: int = 64, hidden_dim: int = 256, n_layers: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        self.n_genes = n_genes

        # encoder with LayerNorm + residual
        enc_layers = [nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            enc_layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        enc_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder
        dec_layers = [nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            dec_layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        self.decoder_hidden = nn.Sequential(*dec_layers)

        self.dec_log_rate = nn.Linear(hidden_dim, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def encode(self, x_log_norm):
        return self.encoder(x_log_norm)

    def decode(self, z, library_size):
        """returns nb mean and dispersion"""
        h = self.decoder_hidden(z)
        # We use softmax to map h to a frequency distribution over genes
        rate = torch.softmax(self.dec_log_rate(h), dim=-1)
        # mu is the expected raw counts, ensuring sum(mu) == library_size
        mu = library_size.unsqueeze(-1) * rate
        theta = self.log_theta.exp().clamp(min=1e-4, max=1e4)
        return mu, theta

    def nb_log_likelihood(self, x_raw, mu, theta):
        """negative binomial log-likelihood"""
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

    def loss(self, x_log_norm, x_raw, library_size):
        z = self.encode(x_log_norm)
        mu, theta = self.decode(z, library_size)
        recon_loss = -self.nb_log_likelihood(x_raw, mu, theta).mean()
        return recon_loss, z

    def forward(self, x_log_norm, x_raw, library_size):
        return self.loss(x_log_norm, x_raw, library_size)

    def reconstruct_log_norm(self, raw_adata, sample=True, device=None):
        """reconstruct log-normalized data from the autoencoder

        Args:
            raw_adata: anndata object containing the data to reconstruct raw_adata.X should not be log-normalized
            sample: if true sample from the nb distribution otherwise return the mean
        """
        self.eval()
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device

            X = raw_adata.X
            X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X)
            x = torch.as_tensor(X_dense, dtype=torch.float32, device=device)
            cell_library_size = x.sum(dim=1)
            x_log_norm = torch.log1p(x / cell_library_size.unsqueeze(-1).clamp(min=1) * 1e4)

            z = self.encode(x_log_norm)
            mu, theta = self.decode(z, cell_library_size)

            if sample:
                # Sample from NB: total_count = theta, probs = mu / (mu + theta)
                # PyTorch NB mean = total_count * probs / (1 - probs) = theta * (mu/(mu+theta)) / (theta/(mu+theta)) = mu
                p = mu / (mu + theta + 1e-8)
                nb_sample = torch.distributions.NegativeBinomial(total_count=theta, probs=p).sample()
                # normalize by sample's own library size (matches scanpy normalize_total)
                sample_lib = nb_sample.sum(dim=-1, keepdim=True).clamp(min=1)
                return torch.log1p(nb_sample / sample_lib * 1e4)
            else:
                return torch.log1p(mu / cell_library_size.unsqueeze(-1).clamp(min=1) * 1e4)
