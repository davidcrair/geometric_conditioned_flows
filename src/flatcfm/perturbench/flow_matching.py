"""perturbench adapter for flatcfm flow matching"""

import torch
import torch.nn.functional as F

from flatcfm.models.flow import CondFlow, ConditionEncoder
from .base import FlatCFMAdapter


def _euler_sample(flow, x_init, cond_batch, n_steps: int = 50, time: float = 1.0):
    """memory-efficient euler integration that only keeps current state"""

    dt = time / n_steps
    x = x_init
    for i in range(n_steps):
        t = torch.full((x.size(0),), i * dt, device=x.device, dtype=x.dtype)
        v = flow(x, t, cond_batch)
        x = x + v * dt
    return x


class FlatCFMFlowMatching(FlatCFMAdapter):
    """flatcfm flow matching model running inside perturbench

    uses ConditionEncoder (learned embeddings) and CondFlow velocity field
    with ODE integration for prediction
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        condition_dim: int = 128,
        embedding_dim: int = 64,
        projection_dim: int = 64,
        ode_method: str = "rk4",
        ode_time: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ode_method = ode_method
        self.ode_time = ode_time

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

    def _onehot_to_cond_batch(
        self, pert: torch.Tensor, covs: dict[str, torch.Tensor] | None
    ) -> dict:
        """convert perturbench one-hot tensors to flatcfm integer-ID cond_batch

        ConditionEncoder uses nn.Embedding which needs integer IDs
        """

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
        t = torch.rand(x_1.size(0), device=self.device)
        x_t = (1.0 - t.view(-1, 1)) * x_0 + t.view(-1, 1) * x_1
        target_v = x_1 - x_0
        pred_v = self.model(x_t, t, cond_batch)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def _generate_prediction(self, control, pert, covs, chunk_size: int = 512):
        """generate predictions via chunked ODE integration to avoid OOM"""

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
