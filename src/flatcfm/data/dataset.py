"""
pytorch datasets: CondFMDataset
"""

from typing import TypedDict, Dict, Optional

import anndata as ad
import torch
from scipy import sparse
from torch.utils.data import BatchSampler


from .types import ConditionBatch


def slice_condition_batch(batch: ConditionBatch, idx: torch.Tensor) -> ConditionBatch:
    """slices a conditionbatch typeddict using a tensor of indices"""
    return {
        "perturbations": batch["perturbations"][idx],
        "perturbation_covariates": {k: v[idx] for k, v in batch["perturbation_covariates"].items()},
        "sample_covariates": {k: v[idx] for k, v in batch["sample_covariates"].items()},
    }


def condition_batch_to_device(batch: ConditionBatch, device: torch.device) -> ConditionBatch:
    """moves all tensors in a conditionbatch typeddict to the specified device"""
    return {
        "perturbations": batch["perturbations"].to(device),
        "perturbation_covariates": {k: v.to(device) for k, v in batch["perturbation_covariates"].items()},
        "sample_covariates": {k: v.to(device) for k, v in batch["sample_covariates"].items()},
    }


class CondFMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
        condition_batch: ConditionBatch,
        control_col: str = "vehicle",
        control_value=1,
        n_pcs: int = 50,
        use_norm: bool = True,
        use_pca: bool = True,
        device: str = "cpu",
    ):
        del device  # Keep API compatibility; tensors remain on CPU by design.
        self.adata = adata

        if condition_batch["perturbations"].shape[0] != adata.n_obs:
            raise ValueError(
                f"ConditionBatch perturbations length {condition_batch['perturbations'].shape[0]} != adata.n_obs {adata.n_obs}"
            )

        if use_pca:
            if "X_pca" not in adata.obsm:
                raise ValueError("Run PCA first so adata.obsm['X_pca'] exists.")
            x_pca = adata.obsm["X_pca"]
            if n_pcs > x_pca.shape[1]:
                n_pcs = x_pca.shape[1]
            x = torch.as_tensor(x_pca[:, :n_pcs].copy(), dtype=torch.float32)
        else:
            if "highly_variable" in adata.var:
                hvg_mask = adata.var["highly_variable"].to_numpy()
                x_raw = adata[:, hvg_mask].X
            else:
                # Default to all features if HVGs haven't been computed
                x_raw = adata.X

            if sparse.issparse(x_raw):
                x = torch.as_tensor(x_raw.toarray(), dtype=torch.float32)
            else:
                x = torch.as_tensor(x_raw.copy(), dtype=torch.float32)

        mean = x.mean(0)
        std = x.std(0).clamp_min(1e-8)
        self.data = (x - mean) / std if use_norm else x
        self.mean = mean
        self.std = std
        self.use_norm = use_norm
        self.use_pca = use_pca

        control_mask_np = adata.obs[control_col].to_numpy() == control_value
        self.control_mask = torch.as_tensor(control_mask_np, dtype=torch.bool)
        self.pert_mask = ~self.control_mask

        self.control_global_idx = torch.where(self.control_mask)[0]
        self.pert_global_idx = torch.where(self.pert_mask)[0]

        self.control_data = self.data[self.control_global_idx]
        self.perturbed_data = self.data[self.pert_global_idx]

        self.condition_batch_full = condition_batch
        self.pert_condition_batch = slice_condition_batch(self.condition_batch_full, self.pert_global_idx)

        self.global_to_local_pert = torch.full((adata.n_obs,), -1, dtype=torch.long)
        self.global_to_local_pert[self.pert_global_idx] = torch.arange(self.pert_global_idx.numel())

        pert_condition_ids = self.pert_condition_batch["perturbations"]
        unique_conditions = torch.unique(pert_condition_ids)
        self.condition_to_pert_local = {
            int(condition_id.item()): torch.where(pert_condition_ids == condition_id)[0]
            for condition_id in unique_conditions
        }

    def __len__(self):
        return self.perturbed_data.size(0)

    def __getitem__(self, idx):
        pert_local_idx = int(idx)
        condition_id = int(self.pert_condition_batch["perturbations"][pert_local_idx].item())
        return {
            "pert_local_idx": pert_local_idx,
            "condition_id": condition_id,
        }

    def num_conditions(self):
        return len(self.condition_to_pert_local)

    def sample_condition(self, obs_col: str, obs_value, n: int, generator: Optional[torch.Generator] = None):
        mask = torch.as_tensor(self.adata.obs[obs_col].to_numpy() == obs_value, dtype=torch.bool) & self.pert_mask
        pool_global = torch.where(mask)[0]
        if pool_global.numel() == 0:
            raise ValueError(f"No perturbed cells for {obs_col}={obs_value!r}")

        pick_global = pool_global[torch.randint(0, pool_global.numel(), (n,), generator=generator)]
        pick_local = self.global_to_local_pert[pick_global]
        return {
            "cond_batch": slice_condition_batch(self.condition_batch_full, pick_global),
            "x_target": self.perturbed_data[pick_local],
            "global_idx": pick_global,
            "local_idx": pick_local,
        }


class ConditionFirstBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: CondFMDataset,
        batch_size: int,
        steps_per_epoch: int,
        generator: Optional[torch.Generator] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        if dataset.num_conditions() == 0:
            raise ValueError("No perturbed conditions found for condition-first batching.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.generator = generator
        self.condition_ids = sorted(dataset.condition_to_pert_local.keys())

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            cond_idx = torch.randint(0, len(self.condition_ids), (1,), generator=self.generator).item()
            cond_id = self.condition_ids[cond_idx]
            pool = self.dataset.condition_to_pert_local[cond_id]

            if pool.numel() >= self.batch_size:
                selection = torch.randperm(pool.numel(), generator=self.generator)[: self.batch_size]
                batch_idx = pool[selection]
            else:
                selection = torch.randint(0, pool.numel(), (self.batch_size,), generator=self.generator)
                batch_idx = pool[selection]

            yield batch_idx.tolist()

    def __len__(self):
        return self.steps_per_epoch


def make_train_collate(ds: CondFMDataset):
    def collate(batch_items):
        pert_local_idx = torch.as_tensor([item["pert_local_idx"] for item in batch_items], dtype=torch.long)
        x_1 = ds.perturbed_data.index_select(0, pert_local_idx)
        cond_batch = slice_condition_batch(ds.pert_condition_batch, pert_local_idx)
        return {
            "pert_local_idx": pert_local_idx,
            "x_1": x_1,
            "cond_batch": cond_batch,
        }

    return collate
