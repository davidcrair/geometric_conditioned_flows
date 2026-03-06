"""
autoencoder training loop
based on https://github.com/xingzhis/mioflow-lite
"""

import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import anndata as ad
import numpy as np
import torch.optim as optim
from models import NBAutoEncoder
from training import LossComposer
from collections import defaultdict
from torch import amp

torch.set_float32_matmul_precision("high")


class AEBatchDataset(Dataset):
    """dataset for ae training with optional distance preservation

    Args:
        adata: anndata with raw counts in .X
        distances: pairwise distance matrix (n×n) or per-cell embedding
            vectors (n×d) from which pairwise distances are computed
            on-the-fly per batch the latter avoids storing an n×n matrix
        batch_size: batch size
        shuffle: whether to shuffle each epoch
    """

    def __init__(
        self, adata: ad.AnnData, distances: Optional[np.ndarray] = None, batch_size: int = 128, shuffle: bool = True
    ):
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.x_raw = torch.tensor(X, dtype=torch.float32)
        self.lib_size = self.x_raw.sum(1)
        # log norm
        self.x_log_norm = torch.log1p(self.x_raw / self.lib_size.unsqueeze(1).clamp(min=1) * 1e4)

        n = len(self.x_raw)
        if distances is not None:
            distances = np.asarray(distances, dtype=np.float32)
            if distances.ndim == 2 and distances.shape[0] == n and distances.shape[1] == n:
                # n×n precomputed distance matrix
                self.dist_matrix = torch.tensor(distances, dtype=torch.float32)
                self.dist_embeddings = None
            else:
                # n×d embedding — compute pdist per batch on the fly
                self.dist_matrix = None
                self.dist_embeddings = torch.tensor(distances, dtype=torch.float32)
        else:
            self.dist_matrix = None
            self.dist_embeddings = None

        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self.perm = torch.randperm(n)
        else:
            self.perm = torch.arange(n)

    def on_epoch_end(self):
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
            "lib_size": self.lib_size[batch_idxs],
        }

        if self.dist_matrix is not None:
            dist_mat = self.dist_matrix[batch_idxs][:, batch_idxs]
            triu_idx = np.triu_indices(dist_mat.size(0), k=1)
            batch["distances"] = dist_mat[triu_idx]
        elif self.dist_embeddings is not None:
            batch_emb = self.dist_embeddings[batch_idxs]
            batch["distances"] = torch.pdist(batch_emb)

        return batch


def make_ae_dataloader(
    adata: ad.AnnData, distances: Optional[np.ndarray] = None, batch_size: int = 128, shuffle: bool = True
):
    dataset = AEBatchDataset(adata, distances=distances, batch_size=batch_size, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
    return dataloader


def set_requires_grad(model, encoder: bool = True, decoder: bool = True):
    for param in model.encoder.parameters():
        param.requires_grad = encoder

    for param in model.decoder_hidden.parameters():
        param.requires_grad = decoder

    for param in model.dec_log_rate.parameters():
        param.requires_grad = decoder

    model.log_theta.requires_grad = decoder
    model.alpha.requires_grad = decoder


def train_ae(
    model: NBAutoEncoder,
    train_loader: DataLoader,
    loss_composer: LossComposer,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
    gene_subsample: Optional[int] = None,
) -> dict:
    """
    train nbautoencdoer with flexible loss composition
    """

    model = model.to(device)

    set_requires_grad(model, encoder=not freeze_encoder, decoder=not freeze_decoder)

    # only optimize parameters that require grad
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "individual_train_losses": defaultdict(list),
        "individual_val_losses": defaultdict(list),
    }

    print("training nb autoencoder...")
    print(f"encoder frozen: {freeze_encoder}, decoder frozen: {freeze_decoder}")
    print(f"loss terms: {list(loss_composer.loss_weights.keys())}")

    epoch_pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_individual = {}

        for batch in train_loader:
            x_raw = batch["x_raw"].to(device)
            x_log_norm = batch["x_log_norm"].to(device)
            lib_size = batch["lib_size"].to(device)
            original_distances = batch["distances"].to(device) if "distances" in batch else None

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model.encode(x_log_norm)
                mu, theta = model.decode(z, lib_size)

                loss, ind_losses = loss_composer(
                    x_raw=x_raw,
                    mu=mu,
                    theta=theta,
                    z=z,
                    original_distances=original_distances,
                    model=model,
                    gene_subsample=gene_subsample,
                )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            for k, v in ind_losses.items():
                train_individual[k] = train_individual.get(k, 0.0) + v

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        for k in train_individual:
            avg_ind_loss = train_individual[k] / len(train_loader)
            history["individual_train_losses"][k].append(avg_ind_loss)

        if hasattr(train_loader.dataset, "on_epoch_end"):
            train_loader.dataset.on_epoch_end()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_individual = {}

            with torch.no_grad():
                for batch in val_loader:
                    x_raw = batch["x_raw"].to(device)
                    x_log_norm = batch["x_log_norm"].to(device)
                    lib_size = batch["lib_size"].to(device)
                    original_distances = batch["distances"].to(device) if "distances" in batch else None

                    with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        z = model.encode(x_log_norm)
                        mu, theta = model.decode(z, lib_size)

                        loss, ind_losses = loss_composer(
                            x_raw=x_raw,
                            mu=mu,
                            theta=theta,
                            z=z,
                            original_distances=original_distances,
                            model=model,
                            gene_subsample=gene_subsample,
                        )

                    val_loss += loss.item()

                    for k, v in ind_losses.items():
                        val_individual[k] = val_individual.get(k, 0.0) + v

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            for k in val_individual:
                avg_ind_loss = val_individual[k] / len(val_loader)
                history["individual_val_losses"][k].append(avg_ind_loss)

        postfix = {
            "train_loss": f"{avg_train_loss:.4f}",
            "val_loss": f"{avg_val_loss:.4f}" if val_loader is not None else "N/A",
        }
        for k, v in history["individual_train_losses"].items():
            postfix[f"train_{k}"] = f"{v[-1]:.4f}"
        epoch_pbar.set_postfix(postfix)

        if hasattr(train_loader.dataset, "on_epoch_end"):
            train_loader.dataset.on_epoch_end()

    else:
        postfix = {"train_loss": f"{avg_train_loss:.4f}"}
        for k, v in history["individual_train_losses"].items():
            postfix[f"train_{k}"] = f"{v[-1]:.4f}"
        epoch_pbar.set_postfix(postfix)

    set_requires_grad(model, encoder=True, decoder=True)
    return history


def train_ae_two_phase(
    model,
    train_loader: DataLoader,
    loss_composer: LossComposer,
    encoder_epochs: int = 50,
    decoder_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    phase1_weights: Optional[dict] = None,
    phase2_weights: Optional[dict] = None,
    gene_subsample: Optional[int] = None,
) -> dict:
    """
    two phase training for nbautoencoder
    phase1 trains encoder (decoder frozen) for distance-based losses
    phase2 trains decoder (encoder frozen) for reconstruction-based losses
    """

    print("starting phase 1: training encoder with distance-based losses...")

    # save original weights
    original_weights = loss_composer.loss_weights.copy()

    # update wieghts for phase 1 if provided
    if phase1_weights is not None:
        loss_composer.loss_weights.update(phase1_weights)

    phase1_history = train_ae(
        model=model,
        train_loader=train_loader,
        loss_composer=loss_composer,
        learning_rate=learning_rate,
        epochs=encoder_epochs,
        device=device,
        val_loader=val_loader,
        freeze_encoder=False,
        freeze_decoder=True,
        gene_subsample=gene_subsample,
    )

    print("\n\nstarting phase 2: training decoder with reconstruction-based losses...")

    # update weights for phase 2 if provided
    if phase2_weights is not None:
        loss_composer.loss_weights = original_weights.copy()  # reset to original before updating
        loss_composer.loss_weights.update(phase2_weights)

    phase2_history = train_ae(
        model=model,
        train_loader=train_loader,
        loss_composer=loss_composer,
        learning_rate=learning_rate,
        epochs=decoder_epochs,
        device=device,
        val_loader=val_loader,
        freeze_encoder=True,
        freeze_decoder=False,
        gene_subsample=gene_subsample,
    )

    # restore original weights
    loss_composer.loss_weights = original_weights

    combined_history = {
        "phase1": phase1_history,
        "phase2": phase2_history,
    }

    return combined_history
