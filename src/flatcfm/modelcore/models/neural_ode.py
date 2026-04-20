"""neural ode lightning model"""

from __future__ import annotations

import torch
from torchdiffeq import odeint, odeint_adjoint

from flatcfm.models.flow import CondFlow, CondFlowODE, sample_ode
from flatcfm.training.losses import DensityLoss, EnergyLoss, LossComposer, OTLoss

from .base import BasePerturbationModel


class NeuralODEModel(BasePerturbationModel):
    """neural ode model"""

    task_name = "ode"

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
            {
                "ot": OTLoss(
                    sinkhorn_reg=float(self.loss_cfg.get("sinkhorn_reg", 0.1)),
                    sinkhorn_max_iter=int(self.loss_cfg.get("sinkhorn_max_iter", 50)),
                ),
                "density": DensityLoss(),
                "energy": EnergyLoss(),
            },
            self.loss_cfg["weights"],
        )

    def _odeint(self):
        """select odeint fn"""

        return odeint_adjoint if bool(self.task_cfg.get("adjoint", False)) else odeint

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """shared step"""

        x_0 = batch["x_0"].to(self.device)
        x_1 = batch["x_1"].to(self.device)
        cond_batch = self._move_condition_batch(batch["cond_batch"])
        energy_weight = float(self.loss_composer.loss_weights.get("energy", 0.0))
        augmented = energy_weight > 0
        n_steps = int(self.task_cfg.get("n_energy_steps", 10))
        step_size = 1.0 / max(n_steps, 1)
        t_span = torch.tensor([0.0, 1.0], device=self.device)
        ode_func = CondFlowODE(self.model, cond_batch, torch.device(self.device), augmented=augmented)
        init_state = (x_0, torch.zeros(x_0.shape[0], device=self.device)) if augmented else x_0
        trajectory = self._odeint()(
            ode_func,
            init_state,
            t_span,
            method=str(self.task_cfg.get("ode_method", "midpoint")),
            rtol=1e-3,
            atol=1e-3,
            options={"step_size": step_size},
        )
        if augmented:
            x_pred = trajectory[0][-1]
            kinetic_energy = trajectory[1][-1]
        else:
            x_pred = trajectory[-1]
            kinetic_energy = None
        cost_matrix = torch.cdist(x_pred, x_1) ** 2
        loss, individual_losses = self.loss_composer(
            x_pred=x_pred,
            x_target=x_1,
            kinetic_energy=kinetic_energy,
            cond_batch=ode_func.cond_batch,
            model=self.model,
            cost_matrix=cost_matrix,
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
