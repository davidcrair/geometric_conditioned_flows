"""
training loop for neural ODE model with distribution matching losses
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Optional, Dict
from torchdiffeq import odeint, odeint_adjoint
from torch import amp

from data.dataset import CondFMDataset, make_train_collate, ConditionFirstBatchSampler, condition_batch_to_device
from models.flow import CondFlow, CondFlowODE
from training.losses import LossComposer
from collections import defaultdict

torch.set_float32_matmul_precision("high")


def train_neural_ode(
    model: CondFlow,
    dataset: CondFMDataset,
    loss_composer: LossComposer,
    optimizer: optim.Optimizer,
    epochs: int = 100,
    batch_size: int = 1024,
    device: str = "cpu",
    val_dataset: Optional[CondFMDataset] = None,
    save_path: Optional[str] = None,
    use_sampler: bool = True,
    steps_per_epoch: int = 100,
    ode_method: str = "dopri5",
    adjoint: bool = False,
    n_energy_steps: int = 10,
) -> dict:
    """
    training loop for neural ODE model using distribution matching losses (OT, Density, Energy)
    """

    model.to(device)
    device_obj = torch.device(device)

    train_collate = make_train_collate(dataset)

    if use_sampler:
        sampler = ConditionFirstBatchSampler(dataset, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        train_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=train_collate)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate)

    # control data (x_0 source)
    control_data = dataset.control_data.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "individual_train_losses": defaultdict(list),
        "individual_val_losses": defaultdict(list),
    }

    ode_func_type = odeint_adjoint if adjoint else odeint

    # t_span for integration and energy loss
    t_span = torch.linspace(0.0, 1.0, n_energy_steps, device=device)

    print(f"training neural ODE model on device: {device} (adjoint={adjoint}, method={ode_method})")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_individual = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x_1 = batch["x_1"].to(device)
            cond_batch = batch["cond_batch"]  # Stay on CPU for now, CondFlowODE handles device
            curr_batch_size = x_1.size(0)

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # sample x_0 (source) from control cells
                idx_0 = torch.randint(0, control_data.size(0), (curr_batch_size,), device=device)
                x_0 = control_data[idx_0]

                # wrap model for ODE integration
                ode_func = CondFlowODE(model, cond_batch, device_obj)

                # integrate from t=0 to t=1
                # we need the full trajectory if energy loss is used
                trajectory = ode_func_type(ode_func, x_0, t_span, method=ode_method, rtol=1e-3, atol=1e-3)

                x_pred = trajectory[-1]

                # pre-compute shared cost matrix for OTLoss and DensityLoss
                cost_matrix = torch.cdist(x_pred, x_1) ** 2

                # compute losses
                loss, ind_losses = loss_composer(
                    x_pred=x_pred,
                    x_target=x_1,
                    x_trajectory=trajectory,
                    t_span=t_span,
                    cond_batch=ode_func.cond_batch,  # use the device-moved cond_batch
                    model=model,
                    cost_matrix=cost_matrix,
                    # compatibility with other loss terms if any
                    pred_v=None,
                    target_v=None,
                    x_t=None,
                    t=None,
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
                    cond_batch = batch["cond_batch"]
                    curr_batch_size = x_1.size(0)

                    idx_0 = torch.randint(0, val_control_pool.size(0), (curr_batch_size,), device=device)
                    x_0 = val_control_pool[idx_0]

                    with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ode_func = CondFlowODE(model, cond_batch, device_obj)
                        trajectory = ode_func_type(ode_func, x_0, t_span, method=ode_method, rtol=1e-3, atol=1e-3)
                        x_pred = trajectory[-1]

                        cost_matrix = torch.cdist(x_pred, x_1) ** 2

                        loss, ind_losses = loss_composer(
                            x_pred=x_pred,
                            x_target=x_1,
                            x_trajectory=trajectory,
                            t_span=t_span,
                            cond_batch=ode_func.cond_batch,
                            model=model,
                            cost_matrix=cost_matrix,
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
        model_save_path = os.path.join(save_path, "neural_ode_model.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return history
