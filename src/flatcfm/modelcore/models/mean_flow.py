"""mean flow lightning model"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from flatcfm.models.mean_flow import CondMeanFlow
from flatcfm.training.losses import LossComposer, MeanFlowIdentityLoss

from .base import BasePerturbationModel


def _sample_time_pair(
    batch_size: int,
    device: torch.device,
    t_min: float,
    use_sorted_time_sampling: bool,
    mismatch_ratio_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """sample time pair"""

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
    """ot pair controls"""

    cost = torch.cdist(x_1, x_0_candidates) ** 2
    _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    col_idx = torch.as_tensor(col_ind, dtype=torch.long, device=x_1.device)
    return x_0_candidates.index_select(0, col_idx)


class MeanFlowModel(BasePerturbationModel):
    """mean flow model"""

    task_name = "mean_flow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = CondMeanFlow(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
            hidden_dim=int(self.model_cfg.get("hidden_dim", 256)),
            hidden_layers=int(self.model_cfg.get("hidden_layers", 4)),
            cond_encoder=self._build_condition_encoder(),
        )
        self.loss_composer = LossComposer(
            {
                "mf_identity": MeanFlowIdentityLoss(
                    adaptive_weighting=bool(self.loss_cfg.get("adaptive_weighting", False))
                )
            },
            self.loss_cfg["weights"],
        )

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """shared step"""

        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)
        cond_batch = self._move_condition_batch(batch["cond_batch"])
        if bool(self.task_cfg.get("use_ot_coupling", True)):
            x_0 = _ot_pair_controls(x_1=x_1, x_0_candidates=x_0)

        r, t = _sample_time_pair(
            batch_size=x_1.size(0),
            device=self.device,
            t_min=float(self.task_cfg.get("t_min", 1e-3)),
            use_sorted_time_sampling=bool(self.task_cfg.get("use_sorted_time_sampling", True)),
            mismatch_ratio_m=int(self.task_cfg.get("mismatch_ratio_m", 50)),
        )
        z_t = (1.0 - t.view(-1, 1)) * x_0 + t.view(-1, 1) * x_1
        v = x_1 - x_0

        def u_fn(z_arg, r_arg, t_arg):
            return self.model(z_arg, r_arg, t_arg, cond_batch)

        tangents = (v, torch.zeros_like(r), torch.ones_like(t))
        u_theta, du_dt = torch.func.jvp(u_fn, (z_t, r, t), tangents)
        loss, individual_losses = self.loss_composer(
            u_theta=u_theta,
            v=v,
            du_dt=du_dt,
            t=t,
            r=r,
        )
        self._log_losses(stage, loss, individual_losses, batch_size=x_1.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions via one-step mean flow"""

        return self.model.sample_one_step(
            batch["x_ctrl"].to(self.device),
            self._move_condition_batch(batch["cond_batch"]),
        )
