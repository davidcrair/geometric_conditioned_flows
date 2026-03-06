import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from torch import amp

from data.dataset import CondFMDataset, make_train_collate, ConditionFirstBatchSampler, condition_batch_to_device
from models.mean_flow import CondMeanFlow
from training.losses import LossComposer

torch.set_float32_matmul_precision("high")


def _sample_time_pair(
    batch_size: int,
    device: torch.device,
    t_min: float,
    use_sorted_time_sampling: bool,
    mismatch_ratio_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """sample `(r, t)` with optional paper-style mismatch scheduling

    Args:
        batch_size: number of samples to draw
        device: device for generated tensors
        t_min: lower bound for sampled times
        use_sorted_time_sampling: if true draw two times and sort into `r <= t`
        mismatch_ratio_m: controls fraction of equal-time rows
            equal-time probability is `m / (m + 1)`

    Returns:
        tuple `(r, t)` with shape `(batch_size,)` each
    """
    if use_sorted_time_sampling:
        a = torch.rand(batch_size, device=device) * (1.0 - t_min) + t_min
        b = torch.rand(batch_size, device=device) * (1.0 - t_min) + t_min
        r = torch.minimum(a, b)
        t = torch.maximum(a, b)
    else:
        t = torch.rand(batch_size, device=device) * (1.0 - t_min) + t_min
        r = torch.rand(batch_size, device=device) * t

    if mismatch_ratio_m > 0:
        equal_prob = mismatch_ratio_m / (mismatch_ratio_m + 1.0)
        equal_mask = torch.rand(batch_size, device=device) < equal_prob
        r = torch.where(equal_mask, t, r)

    return r, t


def _ot_pair_controls(x_1: torch.Tensor, x_0_candidates: torch.Tensor) -> torch.Tensor:
    """compute ot-coupled `(x_0, x_1)` pairs for a minibatch

    uses exact linear assignment (discrete ot with uniform masses) to find a
    low-cost permutation between sampled controls and targets

    Args:
        x_1: target samples with shape `(batch_size, dim)`
        x_0_candidates: candidate control samples `(batch_size, dim)`

    Returns:
        reordered `x_0_candidates` aligned to `x_1`
    """
    cost = torch.cdist(x_1, x_0_candidates) ** 2
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    del row_ind  # rows are sorted and map 1:1 to x_1 order
    col_idx = torch.as_tensor(col_ind, dtype=torch.long, device=x_1.device)
    return x_0_candidates.index_select(0, col_idx)


def train_mean_flow(
    model: CondMeanFlow,
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
    t_min: float = 1e-3,
    use_distributional_loss: bool = False,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    use_ot_coupling: bool = True,
    use_sorted_time_sampling: bool = True,
    mismatch_ratio_m: int = 50,
) -> dict:
    """training loop for meanflow with jvp-based identity loss

    Args:
        model: meanflow model to train
        dataset: conditional dataset with control/perturbed splits
        loss_composer: composable loss object
        optimizer: optimizer for model parameters
        epochs: number of training epochs
        batch_size: batch size
        device: torch device string
        val_dataset: optional validation dataset
        save_path: optional model save directory
        use_sampler: if true use condition-first sampler
        steps_per_epoch: number of sampled batches per epoch
        t_min: minimum sampled time
        use_distributional_loss: optional one-step distributional loss toggle
        scheduler: optional lr scheduler
        use_ot_coupling: if true ot-couple sampled controls to targets per batch
        use_sorted_time_sampling: if true sample two times and sort into `r <= t`
        mismatch_ratio_m: equal-time ratio control probability of `r == t` is
            `m / (m + 1)` where `m` is this value

    Returns:
        training history dictionary
    """
    model.to(device)
    device_obj = torch.device(device)

    train_collate = make_train_collate(dataset)

    if use_sampler:
        sampler = ConditionFirstBatchSampler(dataset, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        train_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=train_collate)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate)

    control_data = dataset.control_data.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "individual_train_losses": defaultdict(list),
        "individual_val_losses": defaultdict(list),
    }

    print("Training MeanFlow model on device:", device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_individual = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x_1 = batch["x_1"].to(device)
            cond_batch = condition_batch_to_device(batch["cond_batch"], device_obj)
            curr_batch_size = x_1.size(0)

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 1. Sample starting control variables
                idx_0 = torch.randint(0, control_data.size(0), (curr_batch_size,), device=device)
                x_0 = control_data[idx_0]
                if use_ot_coupling:
                    x_0 = _ot_pair_controls(x_1=x_1, x_0_candidates=x_0)

                # 2. Sample temporal variables (r, t)
                r, t = _sample_time_pair(
                    batch_size=curr_batch_size,
                    device=device_obj,
                    t_min=t_min,
                    use_sorted_time_sampling=use_sorted_time_sampling,
                    mismatch_ratio_m=mismatch_ratio_m,
                )

                # 3. Interpolate using the terminal time t (not r)
                t_expanded = t.view(-1, 1)
                z_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1

                # The instantaneous velocity v
                v = x_1 - x_0  # [cite: 83]

                # 4. Forward-mode Autodiff (JVP)
                # Differentiate only with respect to the state z_t and time variables r, t.
                def u_fn(z_arg, r_arg, t_arg):
                    return model(z_arg, r_arg, t_arg, cond_batch)

                # The MeanFlow total derivative strictly requires the spatial tangent to be v
                tangents = (v, torch.zeros_like(r), torch.ones_like(t))  # [cite: 162]
                u_theta, du_dt = torch.func.jvp(u_fn, (z_t, r, t), tangents)

                # 5. Optional Distributional / 1-Step Validation
                x_pred_one_step = None
                x_target = None
                if use_distributional_loss:
                    with torch.no_grad():
                        # Evaluate 1-step prediction from prior (t=1 to r=0)
                        x_pred_one_step = model.sample_one_step(x_0, cond_batch)
                    x_target = x_1

                # 6. Compute losses
                # NOTE: Ensure `loss_composer` applies a stop_gradient (.detach())
                # to the target calculation u_tgt = v - (t-r)*du_dt to prevent higher-order gradients.
                loss, ind_losses = loss_composer(
                    u_theta=u_theta,
                    v=v,
                    du_dt=du_dt,
                    t=t,
                    r=r,
                    x_pred_one_step=x_pred_one_step,
                    x_target=x_target,
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for k, val in ind_losses.items():
                epoch_individual[k] += val

        # Aggregate training metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_epoch_loss)

        for k in epoch_individual:
            avg_ind_loss = epoch_individual[k] / len(train_loader)
            history["individual_train_losses"][k].append(avg_ind_loss)

        # 7. Validation Loop
        if val_dataset is not None:
            model.eval()
            val_loss = 0.0
            val_individual = defaultdict(float)

            val_collate = make_train_collate(val_dataset)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collate)
            val_control_pool = val_dataset.control_data.to(device)

            with torch.no_grad():
                for batch in val_loader:
                    x_1 = batch["x_1"].to(device)
                    cond_batch = condition_batch_to_device(batch["cond_batch"], device_obj)
                    curr_batch_size = x_1.size(0)

                    idx_0 = torch.randint(0, val_control_pool.size(0), (curr_batch_size,), device=device)
                    x_0 = val_control_pool[idx_0]
                    if use_ot_coupling:
                        x_0 = _ot_pair_controls(x_1=x_1, x_0_candidates=x_0)

                    r, t = _sample_time_pair(
                        batch_size=curr_batch_size,
                        device=device_obj,
                        t_min=t_min,
                        use_sorted_time_sampling=use_sorted_time_sampling,
                        mismatch_ratio_m=mismatch_ratio_m,
                    )

                    t_expanded = t.view(-1, 1)
                    z_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
                    v = x_1 - x_0

                    with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # Forward pass only
                        u_theta = model(z_t, r, t, cond_batch)

                        # Simplified validation metric (proxy for identity constraint)
                        loss_val = torch.mean((u_theta - v) ** 2)

                    val_loss += loss_val.item()

                    # Track dummy individual losses if needed to maintain structure
                    val_individual["l2_proxy"] += loss_val.item()

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            for k in val_individual:
                history["individual_val_losses"][k].append(val_individual[k] / len(val_loader))

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss:.4f} - Val Proxy Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "mean_flow_model.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return history
