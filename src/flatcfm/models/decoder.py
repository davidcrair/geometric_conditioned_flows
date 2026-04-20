"""conditioned decoder module for perturbation prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedDecoder(nn.Module):
    """decoder MLP that predicts expression from [x_control; condition_onehot]

    shared between flatcfm's DecoderOnlyModel lightning wrapper and the
    perturbench adapter so both use the same architecture
    """

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int = 256):
        """initialize conditioned decoder

        Args:
            input_dim: gene expression dimension
            cond_dim: total one-hot condition dimension
            hidden_dim: hidden layer width
        """

        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_control: torch.Tensor, cond_onehot: torch.Tensor) -> torch.Tensor:
        """predict perturbed expression

        Args:
            x_control: control expression (batch_size, input_dim)
            cond_onehot: concatenated one-hot condition vector (batch_size, cond_dim)
        """

        x_in = torch.cat([x_control, cond_onehot], dim=1)
        return self.decoder(x_in)

    @staticmethod
    def compute_cond_dim(covariate_dicts: dict) -> int:
        """compute total one-hot condition dimension from covariate_dicts"""

        cond_dim = int(covariate_dicts["perturbation_num_categories"])
        for name in sorted(covariate_dicts.get("sample_covariates", {}).keys()):
            cond_dim += int(covariate_dicts["sample_covariates"][name])
        for name in sorted(covariate_dicts.get("perturbation_covariates", {}).keys()):
            cond_dim += int(covariate_dicts["perturbation_covariates"][name])
        return cond_dim

    @staticmethod
    def build_onehot(cond_batch: dict, covariate_dicts: dict) -> torch.Tensor:
        """build concatenated one-hot condition vector from integer-ID cond_batch

        Args:
            cond_batch: dict with perturbations (B,) and covariate index tensors
            covariate_dicts: dict with category cardinalities
        """

        perturbations = cond_batch["perturbations"]
        device = perturbations.device
        parts = []

        n_pert = int(covariate_dicts["perturbation_num_categories"])
        parts.append(F.one_hot(perturbations.long(), n_pert).float())

        for name in sorted(covariate_dicts.get("sample_covariates", {}).keys()):
            n_cat = int(covariate_dicts["sample_covariates"][name])
            vals = cond_batch["sample_covariates"][name].long().to(device)
            parts.append(F.one_hot(vals, n_cat).float())

        for name in sorted(covariate_dicts.get("perturbation_covariates", {}).keys()):
            n_cat = int(covariate_dicts["perturbation_covariates"][name])
            vals = cond_batch["perturbation_covariates"][name].long().to(device)
            parts.append(F.one_hot(vals, n_cat).float())

        return torch.cat(parts, dim=1)
