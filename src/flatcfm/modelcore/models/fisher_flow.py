"""fisher flow lightning model

flow matching on the positive orthant of the unit sphere
internally maps input data to the probability simplex then to the
sphere via sqrt trains with slerp interpolation and geodesic
tangent velocity targets

supports both log1p-normalized and raw counts input spaces
"""

from __future__ import annotations

import ot
import torch

from flatcfm.models.flow import CondFlow
from flatcfm.training.losses import LossComposer, SphereMSELoss

from .base import BasePerturbationModel


# -- space conversion helpers -------------------------------------------------


def _to_simplex(x: torch.Tensor) -> torch.Tensor:
    """convert non-negative data to probability simplex via row normalization

    Args:
        x: non-negative data (B D) eg raw counts

    Returns:
        proportions on simplex (B D) summing to 1
    """

    x = x.clamp(min=0) + 1e-8
    return x / x.sum(dim=-1, keepdim=True)


def _log1p_to_simplex(x: torch.Tensor) -> torch.Tensor:
    """convert log1p-normalized data to probability simplex

    Args:
        x: log1p-normalized gene expression (B D)

    Returns:
        proportions on simplex (B D) summing to 1
    """

    return _to_simplex(torch.expm1(x))


def _simplex_to_log1p(p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """convert probability simplex back to log1p space

    Args:
        p: proportions on simplex (B D)
        scale: per-cell total from expm1(x_ctrl).sum(-1) (B)

    Returns:
        log1p-normalized data (B D) comparable to input space
    """

    return torch.log1p(p * scale.view(-1, 1))


def _simplex_to_raw(p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """convert probability simplex back to raw counts space

    Args:
        p: proportions on simplex (B D)
        scale: per-cell total counts (B)

    Returns:
        raw counts (B D)
    """

    return p * scale.view(-1, 1)


# -- sphere geometry helpers --------------------------------------------------


def _to_orthant(p: torch.Tensor) -> torch.Tensor:
    """map simplex -> positive orthant of sphere via sqrt"""

    return torch.sqrt(p.clamp(min=1e-8))


def _from_orthant(y: torch.Tensor) -> torch.Tensor:
    """map positive orthant -> simplex via square then renormalize"""

    p = y ** 2
    return p / p.sum(dim=-1, keepdim=True).clamp(min=1e-10)


def _normalize_sphere(y: torch.Tensor) -> torch.Tensor:
    """project onto unit sphere"""

    return y / y.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _slerp(y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """spherical linear interpolation

    Args:
        y0: start points on sphere (B D)
        y1: end points on sphere (B D)
        t: interpolation parameter (B 1)

    Returns:
        interpolated points on sphere (B D)
    """

    eps = 1e-6
    cos_omega = (y0 * y1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega).clamp(min=eps)
    w0 = torch.sin((1 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega
    return w0 * y0 + w1 * y1


def _slerp_tangent_velocity(
    y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor, zt: torch.Tensor
) -> torch.Tensor:
    """compute target velocity as tangent to slerp geodesic

    v_target = d/dt slerp(y0 y1 t) projected to tangent space at zt

    Args:
        y0: start points on sphere (B D)
        y1: end points on sphere (B D)
        t: time parameter (B 1)
        zt: interpolated point on sphere (B D)

    Returns:
        tangent velocity at zt (B D)
    """

    eps = 1e-6
    cos_omega = (y0 * y1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega).clamp(min=eps)
    v_target = omega * (
        -torch.cos((1 - t) * omega) / sin_omega * y0
        + torch.cos(t * omega) / sin_omega * y1
    )
    v_target = v_target - (v_target * zt).sum(dim=-1, keepdim=True) * zt
    return v_target


def _project_tangent(v: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """project vector v onto tangent space of sphere at point zt"""

    return v - (v * zt).sum(dim=-1, keepdim=True) * zt


# -- ot coupling on sphere ---------------------------------------------------


def _ot_pair_sinkhorn_sphere(
    x_1: torch.Tensor,
    x_0_pool: torch.Tensor,
    reg: float = 0.05,
) -> torch.Tensor:
    """ot pair controls using sinkhorn with sphere arc-length cost

    Args:
        x_1: perturbed cells on sphere (B D)
        x_0_pool: control cell pool on sphere (B*pool_multiplier D)
        reg: sinkhorn entropic regularization

    Returns:
        x_0 paired to x_1 on sphere
    """

    B = x_1.size(0)
    device = x_1.device
    cos_sim = (x_1 @ x_0_pool.T).clamp(-1 + 1e-6, 1 - 1e-6)
    M = torch.acos(cos_sim)
    scale = M.detach().median().clamp_min(1e-8)
    M = (M / scale).clamp(max=1e3)
    a = torch.full((B,), 1.0 / B, dtype=torch.float32, device=device)
    b = torch.full(
        (x_0_pool.size(0),), 1.0 / x_0_pool.size(0),
        dtype=torch.float32, device=device,
    )
    P = ot.bregman.sinkhorn(
        a, b, M.to(torch.float32), reg=max(reg, 1e-4),
        numItermax=100, stopThr=1e-3,
    )
    col_idx = torch.multinomial(P, num_samples=1).squeeze(1)
    return x_0_pool.index_select(0, col_idx)


# -- fisher flow model -------------------------------------------------------


class FisherFlowModel(BasePerturbationModel):
    """fisher flow model on the positive orthant of the unit sphere

    supports both log1p-normalized and raw counts input spaces
    internally converts to probability simplex then to sphere for
    slerp interpolation and geodesic tangent velocity training
    """

    task_name = "fisher_flow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = CondFlow(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
            hidden_dim=int(self.model_cfg.get("hidden_dim", 256)),
            hidden_layers=int(self.model_cfg.get("hidden_layers", 4)),
            cond_encoder=self._build_condition_encoder(),
        )
        self.loss_composer = LossComposer(
            {"sphere_mse": SphereMSELoss()},
            self.loss_cfg["weights"],
        )
        base_kind = self.space_config.get("base", {}).get("kind", "normalized_log1p")
        self._is_raw_counts = base_kind == "raw_counts"

    def _input_to_simplex(self, x: torch.Tensor) -> torch.Tensor:
        """convert input data to simplex based on space config"""

        if self._is_raw_counts:
            return _to_simplex(x)
        return _log1p_to_simplex(x)

    def _simplex_to_input(self, p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """convert simplex back to input space based on space config"""

        if self._is_raw_counts:
            return _simplex_to_raw(p, scale)
        return _simplex_to_log1p(p, scale)

    def _input_scale(self, x: torch.Tensor) -> torch.Tensor:
        """compute per-cell scale factor for inverse mapping"""

        if self._is_raw_counts:
            return x.clamp(min=0).sum(dim=-1)
        return torch.expm1(x).clamp(min=0).sum(dim=-1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_batch: dict) -> torch.Tensor:
        """forward pass with tangent space projection"""

        v_raw = self.model(x_t, t, cond_batch)
        return _project_tangent(v_raw, x_t)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """training/validation step with fisher flow geometry

        1 convert input data to simplex
        2 map simplex to sphere via sqrt
        3 optionally ot-couple using arc-length cost
        4 slerp interpolate on sphere
        5 compute target velocity as tangent to geodesic
        6 predict velocity with tangent projection
        7 sphere mse loss (sum over features mean over batch)
        """

        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)

        p_0 = self._input_to_simplex(x_0)
        p_1 = self._input_to_simplex(x_1)

        y_0 = _normalize_sphere(_to_orthant(p_0))
        y_1 = _normalize_sphere(_to_orthant(p_1))

        if bool(self.task_cfg.get("use_ot_coupling", False)) and stage == "train":
            y_0 = _ot_pair_sinkhorn_sphere(
                y_1, y_0,
                reg=float(self.task_cfg.get("ot_reg", 0.05)),
            )
        else:
            y_0 = y_0[: y_1.size(0)]

        cond_batch = self._move_condition_batch(batch["cond_batch"])

        t = torch.rand(y_1.size(0), device=self.device)
        t_col = t.view(-1, 1)
        zt = _slerp(y_0, y_1, t_col)
        zt = _normalize_sphere(zt)

        target_v = _slerp_tangent_velocity(y_0, y_1, t_col, zt)
        pred_v = self.forward(zt, t, cond_batch)

        loss, individual_losses = self.loss_composer(
            pred_v=pred_v,
            target_v=target_v.detach(),
            x_t=zt,
            t=t,
            cond_batch=cond_batch,
            model=self.model,
        )
        self._log_losses(stage, loss, individual_losses, batch_size=y_1.size(0))
        return loss

    def _predict(self, batch: dict) -> torch.Tensor:
        """generate predictions via euler integration on sphere

        converts control data to sphere euler-steps the learned velocity
        field with re-normalization then maps back through simplex to
        input space preserving the control cells expression scale
        """

        x_ctrl = batch["x_ctrl"].to(self.device)
        cond_batch = self._move_condition_batch(batch["cond_batch"])
        n_steps = int(self.predict_cfg.get("n_steps", 50))

        scale = self._input_scale(x_ctrl)

        p_ctrl = self._input_to_simplex(x_ctrl)
        zt = _normalize_sphere(_to_orthant(p_ctrl))
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = torch.full((zt.size(0),), step * dt, device=self.device)
            v = self.forward(zt, t_val, cond_batch)
            zt = _normalize_sphere(zt + dt * v)

        p_pred = _from_orthant(zt)
        return self._simplex_to_input(p_pred, scale)
