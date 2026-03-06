"""
training loop for conditional flow matching model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Optional, Dict
from torch import amp

from data.dataset import CondFMDataset, make_train_collate, ConditionFirstBatchSampler, condition_batch_to_device
from models.flow import CondFlow
from training.losses import LossComposer
from collections import defaultdict

torch.set_float32_matmul_precision("high")


def train_flow_matching(
    model: CondFlow,
    dataset: CondFMDataset,
    loss_composer: LossComposer,
    optimizer: optim.Optimizer,
    epochs: int = 100,
    batch_size: int = 1024,
    device: str = "cpu",
    val_dataset: Optional[CondFMDataset] = None,
    save_path: Optional[str] = None,
    flow_noise: float = 0.0,
    use_sampler: bool = True,
    steps_per_epoch: int = 100,
) -> dict:
    """
    training loop for conditional flow matching model
    """

    model.to(device)
    device_obj = torch.device(device)

    train_collate = make_train_collate(dataset)

    if use_sampler:
        sampler = ConditionFirstBatchSampler(dataset, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        train_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=train_collate)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate)

    # x_0
    control_data = dataset.control_data.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "individual_train_losses": defaultdict(list),
        "individual_val_losses": defaultdict(list),
    }

    print("training flow matching model on device:", device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_individual = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x_1 = batch["x_1"].to(device)
            cond_batch = condition_batch_to_device(batch["cond_batch"], device_obj)
            curr_batch_size = x_1.size(0)

            optimizer.zero_grad()

            # sample time uniform
            t = torch.rand(curr_batch_size, device=device)

            # sample x_0 (source) from control cells
            idx_0 = torch.randint(0, control_data.size(0), (curr_batch_size,), device=device)
            x_0 = control_data[idx_0]

            # compute linear interpolation x_t and targ. velocity v_t
            # x_t = (1-t)x_0 + t*x_1
            # v_t = d/dt x_t = x_1 - x_0
            t_expanded = t.view(-1, 1)
            x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
            target_v = x_1 - x_0

            # add flow noise if enabled
            if flow_noise > 0.0:
                noise = torch.randn_like(x_t) * flow_noise
                x_t = x_t + noise

            # predict velocity with flow model
            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_v = model(x_t, t, cond_batch)

                # compute FM loss
                loss, ind_losses = loss_composer(
                    pred_v=pred_v, target_v=target_v, x_t=x_t, t=t, cond_batch=cond_batch, model=model
                )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in ind_losses.items():
                epoch_individual[k] = epoch_individual.get(k, 0.0) + v

        avg_epoch_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_epoch_loss)

        for k in epoch_individual:
            avg_ind_loss = epoch_individual[k] / len(train_loader)
            history["individual_train_losses"][k].append(avg_ind_loss)

        if val_dataset is not None:
            model.eval()
            val_loss = 0.0
            val_collate = make_train_collate(val_dataset)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collate)
            val_control_pool = val_dataset.control_data.to(device)

            with torch.no_grad():
                for batch in val_loader:
                    x_1 = batch["x_1"].to(device)
                    cond_batch = condition_batch_to_device(batch["cond_batch"], device_obj)
                    curr_batch_size = x_1.size(0)

                    # sample time uniform
                    t = torch.rand(curr_batch_size, device=device)

                    # sample x_0 (source) from control cells
                    idx_0 = torch.randint(0, val_control_pool.size(0), (curr_batch_size,), device=device)
                    x_0 = val_control_pool[idx_0]

                    # compute linear interpolation x_t and targ. velocity v_t
                    t_expanded = t.view(-1, 1)
                    x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
                    target_v = x_1 - x_0

                    with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # predict velocity with flow model
                        pred_v = model(x_t, t, cond_batch)

                        # compute FM loss
                        loss, ind_losses = loss_composer(
                            pred_v=pred_v, target_v=target_v, x_t=x_t, t=t, cond_batch=cond_batch, model=model
                        )
                    val_loss += loss.item()

                    for k, v in ind_losses.items():
                        history["individual_val_losses"][k].append(v)

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)
            print(f"epoch {epoch + 1}/{epochs} - train Loss: {avg_epoch_loss:.4f} - val loss: {avg_val_loss:.4f}")
        else:
            print(f"epoch {epoch + 1}/{epochs} - train Loss: {avg_epoch_loss:.4f}")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "flow_matching_model.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return history
