"""perturbench adapter for flatcfm decoder baseline"""

import torch
import torch.nn.functional as F

from flatcfm.models.decoder import ConditionedDecoder
from .base import FlatCFMAdapter


class FlatCFMDecoder(FlatCFMAdapter):
    """flatcfm decoder baseline running inside perturbench

    uses the shared ConditionedDecoder module:
    [x_control; pert_onehot; cov_onehot] -> MLP -> predicted expression
    """

    def __init__(self, hidden_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        cond_dim = self.n_perts + self._n_total_covariates()
        self.decoder_module = ConditionedDecoder(self.n_genes, cond_dim, hidden_dim)

    def _build_cond_onehot(
        self, pert: torch.Tensor, covs: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """concatenate perturbench one-hot tensors into single condition vector"""

        parts = [pert]
        if covs is not None:
            for k in sorted(covs.keys()):
                parts.append(covs[k].squeeze())
        return torch.cat(parts, dim=1)

    def _compute_loss(self, observed, control, pert, covs):
        cond = self._build_cond_onehot(pert, covs)
        return F.mse_loss(self.decoder_module(control, cond), observed)

    def _generate_prediction(self, control, pert, covs):
        cond = self._build_cond_onehot(pert, covs)
        return self.decoder_module(control, cond)
