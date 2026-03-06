"""
interface and implementations of baseline models
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from data.types import ConditionBatch


class BaselineModel(nn.Module):
    """base class for baseline models subclasses will inherit from this"""

    def __init__(self):
        super().__init__()

    def fit(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: ConditionBatch):
        """fit the baseline model to the training data

        Args:
            x_control: tensor of control cells
            x_perturbed: tensor of perturbed cells
            cond_perturbed: condition batch for x_perturbed
        """
        raise NotImplementedError("subclasses should implement this method")

    def sample(self, x_control: torch.Tensor, cond_batch: ConditionBatch) -> torch.Tensor:
        """predict perturbed cells

        Args:
            x_control: tensor of control cells
            cond_batch: condition batch for the cells to predict
        """
        raise NotImplementedError("subclasses should implement this method")


class NoEffectBaseline(BaselineModel):
    """baseline model that predicts no effect of perturbation i.e. the perturbed cells are the same as the control cells"""

    def fit(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: ConditionBatch):
        pass

    def sample(self, x_control: torch.Tensor, cond_batch: ConditionBatch) -> torch.Tensor:
        return x_control


class ContextMeanBaseline(BaselineModel):
    """baseline model that predicts the mean of all matching the condition
    uses sample covariates to group cells but ignores perturbation covariates and perturbation identity
    """

    def __init__(self):
        super().__init__()
        self.condition_means: Dict[str, torch.Tensor] = {}

    def _make_key(self, cond_batch: ConditionBatch, idx: int) -> str:
        parts = []
        for key in cond_batch["sample_covariates"]:
            parts.append(f"{key}={cond_batch['sample_covariates'][key][idx].item()}")
        return "|".join(parts)

    def fit(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: ConditionBatch):
        n_perturbed = x_perturbed.shape[0]
        n_control = x_control.shape[0]
        keys = [self._make_key(cond_perturbed, i) for i in range(n_perturbed)]
        for key in set(keys):
            mask = torch.tensor([k == key for k in keys])
            self.condition_means[key] = torch.cat([x_control, x_perturbed[mask]], dim=0).mean(dim=0)

    def sample(self, x_control: torch.Tensor, cond_batch: ConditionBatch) -> torch.Tensor:
        out = torch.zeros_like(x_control)
        n = x_control.shape[0]
        keys = [self._make_key(cond_batch, i) for i in range(n)]
        for key in set(keys):
            if key not in self.condition_means:
                raise ValueError(f"sample covariate combination {key!r} not seen during fitting")
            mask = torch.tensor([k == key for k in keys])
            out[mask] = self.condition_means[key]
        return out


class PerturbMeanBaseline(BaselineModel):
    """predicts the mean of the perturbed cells grouped by perturbation and covariate key

    groups cells by perturbation perturbation covariates and sample covariates and predicts the mean of the perturbed cells in each group
    """

    def __init__(self, pert_cov_keys: Optional[list] = None, sample_cov_keys: Optional[list] = None):
        super().__init__()
        self.pert_cov_keys = pert_cov_keys if pert_cov_keys is not None else []
        self.sample_cov_keys = sample_cov_keys if sample_cov_keys is not None else []
        self.perturb_means: Dict[str, torch.Tensor] = {}

    def _make_key(self, cond_batch: ConditionBatch, idx: int) -> str:
        parts = [f"perturbation={cond_batch['perturbations'][idx].item()}"]
        for key in self.pert_cov_keys:
            parts.append(f"{key}={cond_batch['perturbation_covariates'][key][idx].item()}")
        for key in self.sample_cov_keys:
            parts.append(f"{key}={cond_batch['sample_covariates'][key][idx].item()}")
        return "|".join(parts)

    def fit(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: ConditionBatch):
        n = x_perturbed.shape[0]
        keys = [self._make_key(cond_perturbed, i) for i in range(n)]
        for key in set(keys):
            mask = torch.tensor([k == key for k in keys])
            self.perturb_means[key] = x_perturbed[mask].mean(dim=0)

    def sample(self, x_control: torch.Tensor, cond_batch: ConditionBatch) -> torch.Tensor:
        out = torch.zeros_like(x_control)
        n = x_control.shape[0]
        keys = [self._make_key(cond_batch, i) for i in range(n)]
        for key in set(keys):
            if key not in self.perturb_means:
                print(f"Warning: perturbation/covariate combination {key!r} not seen during fitting, using zeros")
                # fallback to global mean of all perturbations
                fallback = torch.stack(list(self.perturb_means.values())).mean(dim=0)
                mask = torch.tensor([k == key for k in keys])
                out[mask] = fallback
            else:
                mask = torch.tensor([k == key for k in keys])
                out[mask] = self.perturb_means[key]
        return out


class DecoderOnlyBaseline(BaselineModel):
    """baseline model that uses a decoder to predict the perturbed cells from the control cells and the condition"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, n_epochs: int = 100, lr: float = 1e-3):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr

        self.decoder = None
        self._pert_categories = None
        self._sample_cov_categories = {}
        self._pert_cov_categories = {}
        self._cond_dim = 0

    def _build_onehot(self, cond: dict) -> torch.Tensor:
        """builds a one-hot condition vector from a condition dict

        Args:
            cond: dict with keys:
                - 'perturbations': longtensor of shape (n)
                - 'sample_covariates': dict mapping name -> longtensor of shape (n)
                - 'perturbation_covariates': dict mapping name -> longtensor of shape (n)

        Returns:
            float tensor of shape (n cond_dim) with concatenated one-hot encodings
        """
        parts = []

        # Perturbation one-hot
        pert_ids = cond["perturbations"]
        n = pert_ids.shape[0]
        n_perts = len(self._pert_categories)
        pert_onehot = torch.zeros(n, n_perts)
        for i, cat in enumerate(self._pert_categories):
            pert_onehot[:, i] = (pert_ids == cat).float()
        parts.append(pert_onehot)

        # Sample covariate one-hots
        for name, categories in self._sample_cov_categories.items():
            vals = cond["sample_covariates"].get(name, torch.zeros(n, dtype=torch.long))
            onehot = torch.zeros(n, len(categories))
            for i, cat in enumerate(categories):
                onehot[:, i] = (vals == cat).float()
            parts.append(onehot)

        # Perturbation covariate one-hots
        for name, categories in self._pert_cov_categories.items():
            vals = cond["perturbation_covariates"].get(name, torch.zeros(n, dtype=torch.long))
            onehot = torch.zeros(n, len(categories))
            for i, cat in enumerate(categories):
                onehot[:, i] = (vals == cat).float()
            parts.append(onehot)

        return torch.cat(parts, dim=1)

    def fit(
        self,
        x_control: torch.Tensor,
        x_perturbed: torch.Tensor,
        cond_perturbed: dict,
    ) -> None:
        """fits the decoder mlp on control -> perturbed mapping

        Args:
            x_control: float tensor of shape (n_ctrl input_dim)
            x_perturbed: float tensor of shape (n_pert input_dim)
            cond_perturbed: condition dict for perturbed cells with keys:
                - 'perturbations': longtensor of shape (n_pert)
                - 'sample_covariates': dict mapping name -> longtensor
                - 'perturbation_covariates': dict mapping name -> longtensor
        """
        # Infer categories from training data
        self._pert_categories = cond_perturbed["perturbations"].unique().tolist()
        self._sample_cov_categories = {k: v.unique().tolist() for k, v in cond_perturbed["sample_covariates"].items()}
        self._pert_cov_categories = {
            k: v.unique().tolist() for k, v in cond_perturbed["perturbation_covariates"].items()
        }

        # Compute total conditioning dim
        self._cond_dim = (
            len(self._pert_categories)
            + sum(len(v) for v in self._sample_cov_categories.values())
            + sum(len(v) for v in self._pert_cov_categories.values())
        )

        # Build MLP: input + cond -> hidden -> output
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim + self._cond_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        n_ctrl = x_control.shape[0]
        n_pert = x_perturbed.shape[0]
        cond_onehot_pert = self._build_onehot(cond_perturbed)  # (n_pert, cond_dim)

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            # Subsample perturbed to match control batch size
            idx = torch.randperm(n_pert)[:n_ctrl]
            x_pert_batch = x_perturbed[idx]
            cond_batch = cond_onehot_pert[idx]
            x_in = torch.cat([x_control, cond_batch], dim=1)
            x_pred = self.decoder(x_in)
            loss = torch.nn.functional.mse_loss(x_pred, x_pert_batch)
            loss.backward()
            optimizer.step()

    def sample(self, x_control: torch.Tensor, cond: dict) -> torch.Tensor:
        """generates predicted perturbed cells from control cells

        Args:
            x_control: float tensor of shape (n input_dim)
            cond: condition dict with keys:
                - 'perturbations': longtensor of shape (n)
                - 'sample_covariates': dict mapping name -> longtensor
                - 'perturbation_covariates': dict mapping name -> longtensor

        Returns:
            float tensor of shape (n output_dim) of predicted perturbed cells
        """
        self.decoder.eval()
        with torch.no_grad():
            cond_onehot = self._build_onehot(cond)  # (n, cond_dim)
            x_in = torch.cat([x_control, cond_onehot], dim=1)
            return self.decoder(x_in)


class AdditiveBaseline(BaselineModel):
    """baseline model that predicts the perturbed cells by adding the mean difference between perturbed and control cells to the control cells"""

    def __init__(self):
        super().__init__()
        self.delta = None

    def fit(self, x_control: torch.Tensor, x_perturbed: torch.Tensor, cond_perturbed: ConditionBatch):
        self.delta = x_perturbed.mean(dim=0) - x_control.mean(dim=0)

    def sample(self, x_control: torch.Tensor, cond_batch: ConditionBatch) -> torch.Tensor:
        if self.delta is None:
            raise ValueError("model must be fitted before sampling")
        return x_control + self.delta
