import torch
from typing import TypedDict, Dict

class ConditionBatch(TypedDict):
    perturbations: torch.Tensor
    perturbation_covariates: Dict[str, torch.Tensor]
    sample_covariates: Dict[str, torch.Tensor]
