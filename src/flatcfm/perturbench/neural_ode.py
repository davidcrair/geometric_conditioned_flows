"""perturbench adapter for flatcfm neural ode"""

import torch

from flatcfm.models.flow import CondFlow, CondFlowODE, ConditionEncoder
from flatcfm.training.losses import DensityLoss, EnergyLoss, LossComposer, OTLoss
from torchdiffeq import odeint, odeint_adjoint

from .base import FlatCFMAdapter
from .flow_matching import _euler_sample


class FlatCFMNeuralODE(FlatCFMAdapter):
    """flatcfm neural ode model running inside perturbench

    trains with OT + density + energy losses via forward ODE integration
    uses the same CondFlow velocity field as flow matching
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        condition_dim: int = 128,
        embedding_dim: int = 64,
        projection_dim: int = 64,
        ode_method: str = "midpoint",
        n_energy_steps: int = 10,
        sinkhorn_reg: float = 0.1,
        sinkhorn_max_iter: int = 50,
        debias: bool = True,
        density_top_k: int = 5,
        density_hinge_value: float = 15.0,
        ot_weight: float = 1.0,
        density_weight: float = 0.0,
        energy_weight: float = 0.1,
        ode_time: float = 1.0,
        adjoint: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ode_method = ode_method
        self.ode_time = ode_time
        self._odeint_fn = odeint_adjoint if adjoint else odeint
        self.n_energy_steps = n_energy_steps

        cond_encoder = ConditionEncoder(
            covariate_dicts=self._covariate_dicts,
            condition_dim=condition_dim,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
        )
        self.model = CondFlow(
            input_dim=self.n_genes,
            output_dim=self.n_genes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            cond_encoder=cond_encoder,
        )
        self.loss_composer = LossComposer(
            {
                "ot": OTLoss(
                    sinkhorn_reg=sinkhorn_reg,
                    sinkhorn_max_iter=sinkhorn_max_iter,
                    debias=debias,
                ),
                "density": DensityLoss(top_k=density_top_k, hinge_value=density_hinge_value),
                "energy": EnergyLoss(),
            },
            {"ot": ot_weight, "density": density_weight, "energy": energy_weight},
        )

    def _onehot_to_cond_batch(
        self, pert: torch.Tensor, covs: dict[str, torch.Tensor] | None
    ) -> dict:
        """convert perturbench one-hot tensors to flatcfm integer-ID cond_batch"""

        cond_batch = {
            "perturbations": pert.argmax(dim=1),
            "perturbation_covariates": {},
            "sample_covariates": {},
        }
        if covs is not None:
            for k in sorted(covs.keys()):
                cond_batch["sample_covariates"][k] = covs[k].squeeze().argmax(dim=1)
        return cond_batch

    def _compute_loss(self, observed, control, pert, covs):
        cond_batch = self._onehot_to_cond_batch(pert, covs)
        x_0, x_1 = control, observed
        energy_weight = float(self.loss_composer.loss_weights.get("energy", 0.0))
        augmented = energy_weight > 0
        step_size = 1.0 / max(self.n_energy_steps, 1)
        t_span = torch.tensor([0.0, 1.0], device=self.device)
        ode_func = CondFlowODE(self.model, cond_batch, self.device, augmented=augmented)
        init_state = (x_0, torch.zeros(x_0.shape[0], device=self.device)) if augmented else x_0
        trajectory = self._odeint_fn(
            ode_func, init_state, t_span,
            method=self.ode_method, rtol=1e-3, atol=1e-3,
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
        return loss

    @torch.no_grad()
    def _generate_prediction(self, control, pert, covs, chunk_size: int = 512):
        """generate predictions via chunked euler integration"""

        cond_batch = self._onehot_to_cond_batch(pert, covs)
        n = control.size(0)
        if n <= chunk_size:
            return _euler_sample(
                self.model, control, cond_batch,
                n_steps=50, time=self.ode_time,
            )
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_cond = {
                "perturbations": cond_batch["perturbations"][start:end],
                "perturbation_covariates": {
                    k: v[start:end] for k, v in cond_batch["perturbation_covariates"].items()
                },
                "sample_covariates": {
                    k: v[start:end] for k, v in cond_batch["sample_covariates"].items()
                },
            }
            chunks.append(_euler_sample(
                self.model, control[start:end], chunk_cond,
                n_steps=50, time=self.ode_time,
            ))
        return torch.cat(chunks, dim=0)
