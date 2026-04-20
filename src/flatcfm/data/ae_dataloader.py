"""ae dataloader helpers"""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AEBatchDataset(Dataset):
    """dataset for ae batches"""

    def __init__(
        self,
        adata: ad.AnnData,
        distances: Optional[np.ndarray] = None,
        cell_type_ids: Optional[np.ndarray] = None,
        perturbation_ids: Optional[np.ndarray] = None,
        batch_size: int = 128,
        shuffle: bool = True,
        x_raw: Optional[np.ndarray] = None,
        x_input: Optional[np.ndarray] = None,
        library_size: Optional[np.ndarray] = None,
        input_library_size: Optional[np.ndarray] = None,
        input_space_kind: str = "normalized_log1p",
        target_sum: float = 1e4,
    ):
        x_value = adata.X if x_raw is None else np.asarray(x_raw, dtype=np.float32)
        if hasattr(x_value, "toarray"):
            x_value = x_value.toarray()

        self.x_raw = torch.tensor(np.asarray(x_value, dtype=np.float32), dtype=torch.float32)
        self.subset_lib_size = self.x_raw.sum(1)
        if library_size is None:
            self.lib_size = self.subset_lib_size.clone()
        else:
            self.lib_size = torch.tensor(np.asarray(library_size, dtype=np.float32), dtype=torch.float32)
        if input_library_size is None:
            self.input_lib_size = self.lib_size.clone()
        else:
            self.input_lib_size = torch.tensor(np.asarray(input_library_size, dtype=np.float32), dtype=torch.float32)

        if input_space_kind == "raw_counts":
            self.x_log_norm = self.x_raw.clone()
        else:
            self.x_log_norm = torch.log1p(self.x_raw / self.input_lib_size.unsqueeze(1).clamp(min=1) * target_sum)

        if x_input is None:
            self.x_input = self.x_raw if input_space_kind == "raw_counts" else self.x_log_norm
        else:
            self.x_input = torch.tensor(np.asarray(x_input, dtype=np.float32), dtype=torch.float32)

        n_rows = len(self.x_raw)
        if distances is not None:
            distances = np.asarray(distances, dtype=np.float32)
            if distances.ndim == 2 and distances.shape[0] == n_rows and distances.shape[1] == n_rows:
                self.dist_matrix = torch.tensor(distances, dtype=torch.float32)
                self.dist_embeddings = None
            else:
                self.dist_matrix = None
                self.dist_embeddings = torch.tensor(distances, dtype=torch.float32)
        else:
            self.dist_matrix = None
            self.dist_embeddings = None

        if cell_type_ids is not None:
            self.cell_type_ids = torch.tensor(np.asarray(cell_type_ids, dtype=np.int64), dtype=torch.long)
        else:
            self.cell_type_ids = None

        if perturbation_ids is not None:
            self.perturbation_ids = torch.tensor(np.asarray(perturbation_ids, dtype=np.int64), dtype=torch.long)
        else:
            self.perturbation_ids = None

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.perm = torch.randperm(n_rows) if self.shuffle else torch.arange(n_rows)

    def on_epoch_end(self):
        """reshuffle indices"""

        if self.shuffle:
            self.perm = torch.randperm(len(self.x_raw))

    def __len__(self):
        return (len(self.x_raw) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.x_raw))
        batch_idxs = self.perm[start_idx:end_idx]

        batch = {
            "x_raw": self.x_raw[batch_idxs],
            "x_log_norm": self.x_log_norm[batch_idxs],
            "x_input": self.x_input[batch_idxs],
            "lib_size": self.lib_size[batch_idxs],
            "subset_lib_size": self.subset_lib_size[batch_idxs],
            "input_lib_size": self.input_lib_size[batch_idxs],
        }

        if self.dist_matrix is not None:
            dist_mat = self.dist_matrix[batch_idxs][:, batch_idxs]
            triu_idx = np.triu_indices(dist_mat.size(0), k=1)
            batch["distances"] = dist_mat[triu_idx]
        elif self.dist_embeddings is not None:
            # return the raw embedding slice pdist gets computed on gpu in the
            # training step see autoencoder module _shared_step
            batch["distance_embeddings"] = self.dist_embeddings[batch_idxs]

        if self.cell_type_ids is not None:
            batch["cell_type_ids"] = self.cell_type_ids[batch_idxs]

        if self.perturbation_ids is not None:
            batch["perturbation_ids"] = self.perturbation_ids[batch_idxs]

        return batch


def make_ae_dataloader(
    adata: ad.AnnData,
    distances: Optional[np.ndarray] = None,
    cell_type_ids: Optional[np.ndarray] = None,
    perturbation_ids: Optional[np.ndarray] = None,
    batch_size: int = 128,
    shuffle: bool = True,
    x_raw: Optional[np.ndarray] = None,
    x_input: Optional[np.ndarray] = None,
    library_size: Optional[np.ndarray] = None,
    input_library_size: Optional[np.ndarray] = None,
    input_space_kind: str = "normalized_log1p",
    target_sum: float = 1e4,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """build ae dataloader"""

    dataset = AEBatchDataset(
        adata,
        distances=distances,
        cell_type_ids=cell_type_ids,
        perturbation_ids=perturbation_ids,
        batch_size=batch_size,
        shuffle=shuffle,
        x_raw=x_raw,
        x_input=x_input,
        library_size=library_size,
        input_library_size=input_library_size,
        input_space_kind=input_space_kind,
        target_sum=target_sum,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )
