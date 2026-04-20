"""flow matching lightning model"""

from __future__ import annotations

import ot
import torch

from flatcfm.models.flow import CondFlow, sample_ode
from flatcfm.training.losses import FlowMatchingMSELoss, FlowMatchingWeightedMSELoss, LossComposer

from .base import BasePerturbationModel


def _ot_pair_sinkhorn(
    x_1: torch.Tensor,
    x_0_pool: torch.Tensor,
    reg: float = 0.05,
) -> torch.Tensor:
    """ot pair controls using sinkhorn on gpu

    Args:
        x_1: perturbed cells (B, D)
        x_0_pool: control cell pool (B*pool_multiplier, D)
        reg: sinkhorn entropic regularization

    Returns:
        x_0 paired to x_1
    """
    B = x_1.size(0)
    device = x_1.device
    M = torch.cdist(x_1, x_0_pool) ** 2
    scale = M.detach().median().clamp_min(1e-8)
    M = (M / scale).clamp(max=1e3)
    a = torch.full((B,), 1.0 / B, dtype=torch.float32, device=device)
    b = torch.full((x_0_pool.size(0),), 1.0 / x_0_pool.size(0), dtype=torch.float32, device=device)
    P = ot.bregman.sinkhorn(
        a, b, M.to(torch.float32), reg=max(reg, 1e-4),
        numItermax=100, stopThr=1e-3,
    )

    # fix invalid rows from sinkhorn non-convergence: replace with uniform
    row_sums = P.sum(dim=1)
    bad_rows = (row_sums <= 0) | torch.isnan(row_sums)
    if bad_rows.any():
        P[bad_rows] = 1.0 / P.size(1)

    col_idx = torch.multinomial(P, num_samples=1).squeeze(1)

    return x_0_pool.index_select(0, col_idx)


class FlowMatchingModel(BasePerturbationModel):
    """flow matching model"""

    task_name = "fm"

    def __init__(self, **kwargs):
        dimension_weights = kwargs.pop("dimension_weights", None)
        super().__init__(**kwargs)
        self.model = CondFlow(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
            hidden_dim=int(self.model_cfg.get("hidden_dim", 256)),
            hidden_layers=int(self.model_cfg.get("hidden_layers", 4)),
            cond_encoder=self._build_condition_encoder(),
            conditioning=str(self.model_cfg.get("conditioning", "concat")),
            dropout=float(self.model_cfg.get("dropout", 0.0)),
        )
        use_weighted = bool(self.loss_cfg.get("weights", {}).get("fm_weighted_mse", 0))
        if use_weighted:
            loss_map = {"fm_weighted_mse": FlowMatchingWeightedMSELoss(dimension_weights)}
        else:
            loss_map = {"fm_mse": FlowMatchingMSELoss()}
        self.loss_composer = LossComposer(loss_map, self.loss_cfg["weights"])

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        """forward"""

        return self.model(x_t, t, cond_batch)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """shared step"""

        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)
        if bool(self.task_cfg.get("use_ot_coupling", False)) and stage == "train":
            x_0 = _ot_pair_sinkhorn(x_1, x_0, reg=float(self.task_cfg.get("ot_reg", 0.05)))
        else:
            x_0 = x_0[: x_1.size(0)]
        cond_batch = self._move_condition_batch(batch["cond_batch"])
        t = torch.rand(x_1.size(0), device=self.device)
        x_t = (1.0 - t.view(-1, 1)) * x_0 + t.view(-1, 1) * x_1
        if float(self.task_cfg.get("flow_noise", 0.0)) > 0:
            x_t = x_t + torch.randn_like(x_t) * float(self.task_cfg.get("flow_noise", 0.0))
        target_v = x_1 - x_0
        pred_v = self.forward(x_t, t, cond_batch)
        loss, individual_losses = self.loss_composer(
            pred_v=pred_v,
            target_v=target_v,
            x_t=x_t,
            t=t,
            cond_batch=cond_batch,
            model=self.model,
        )
        self._log_losses(stage, loss, individual_losses, batch_size=x_1.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions via ode sampling"""

        return sample_ode(
            self.model,
            batch["x_ctrl"].to(self.device),
            self._move_condition_batch(batch["cond_batch"]),
            method=str(self.predict_cfg.get("ode_method", "rk4")),
            time=float(self.predict_cfg.get("time", 1.0)),
        )
