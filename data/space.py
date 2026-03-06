"""
space manager for bi-directional space transformation
"""

import torch
import numpy as np
import scanpy as sc
from typing import Optional, Union, List
from models.autoencoder import NBAutoEncoder


class SpaceManager:
    """handles transitions between raw counts and the training space

    modes:
        - "raw": no transformation
        - "log1p": total count normalization and log(1 + x) transformation on hvgs
        - "ae_latent": encode log1p data into ae latent space
    """

    def __init__(
        self,
        mode: str = "log1p",
        n_hvgs: Optional[int] = 2000,
        ae_model: Optional[NBAutoEncoder] = None,
        target_sum: float = 1e4,
    ):
        if mode not in ["raw", "log1p", "ae_latent"]:
            raise ValueError(f"Unsupported mode: {mode}. Must be one of 'raw', 'log1p', or 'ae_latent'.")
        self.mode = mode
        self.n_hvgs = n_hvgs
        self.ae_model = ae_model
        self.target_sum = target_sum

        self.hvg_names: List[str] = []
        self.is_fitted = False

    def fit(self, adata: sc.AnnData):
        """identifies hvgs and prepares space

        Args:
            adata: anndata object to fit the space manager on

        Raises:
            ValueError: if n_hvgs exceeds the number of features in adata
        """
        n_vars = adata.n_vars

        # "raw" mode needs no HVG selection; skip regardless of n_hvgs
        run_hvg = self.n_hvgs is not None and self.mode != "raw"

        if run_hvg:
            if self.n_hvgs > n_vars:
                raise ValueError(
                    f"n_hvgs={self.n_hvgs} exceeds the number of features "
                    f"({n_vars}). Reduce n_hvgs or pass n_hvgs=None to use all features."
                )
            temp_adata = adata.copy()
            sc.pp.normalize_total(temp_adata, target_sum=self.target_sum)
            sc.pp.log1p(temp_adata)
            sc.pp.highly_variable_genes(temp_adata, n_top_genes=self.n_hvgs, flavor="seurat_v3", subset=False)
            self.hvg_names = adata.var_names[temp_adata.var["highly_variable"]].tolist()
        else:
            self.hvg_names = adata.var_names.tolist()

        self.is_fitted = True

    def to_latent(self, adata: sc.AnnData, device: str = "cpu") -> torch.Tensor:
        """
        maps anndata (raw counts) to training space
        """

        if not self.is_fitted:
            raise ValueError("SpaceManager must be fitted before transformation")

        # subset to hvg
        curr_adata = adata[:, self.hvg_names].copy()

        x = curr_adata.X
        # handle sparse matrix
        if hasattr(x, "toarray"):
            x = x.toarray()

        sc.pp.normalize_total(curr_adata, target_sum=self.target_sum)
        sc.pp.log1p(curr_adata)
        x_dense = curr_adata.X.toarray() if hasattr(curr_adata.X, "toarray") else curr_adata.X
        x_log = torch.tensor(x_dense, dtype=torch.float32).to(device)

        if self.mode == "raw":
            x_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            return torch.tensor(x_raw, dtype=torch.float32).to(device)

        if self.mode == "log1p":
            return x_log

        if self.mode == "ae_latent":
            if self.ae_model is None:
                raise ValueError("AE model must be provided for ae_latent mode")
            with torch.no_grad():
                latent = self.ae_model.encode(x_log)
            return latent

    def to_raw(self, latent_tensor: torch.Tensor, library_size: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        maps from training space back to raw counts
        """

        if self.mode == "raw":
            return latent_tensor

        if self.mode == "ae_latent":
            if self.ae_model is None:
                raise ValueError("AE model must be provided for ae_latent mode")
            self.ae_model.eval()
            with torch.no_grad():
                # decode back to NB params
                mu, theta = self.ae_model.decode(latent_tensor, library_size)

                if sample:
                    # Sample from NB: r = theta, p = theta / (theta + mu)
                    p = mu / (mu + theta + 1e-8)
                    r = theta
                    dist = torch.distributions.NegativeBinomial(total_count=r, probs=p)
                    return dist.sample()
                else:
                    return mu

        if self.mode == "log1p":
            x_norm = torch.expm1(latent_tensor)
            raw = x_norm * library_size.unsqueeze(1) / self.target_sum
            return raw
