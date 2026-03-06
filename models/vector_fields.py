from torch import nn
from typing import Dict
from models import ConditionEncoder
from data import ConditionBatch
import torch


class CondODEFunc(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, cond_encoder: ConditionEncoder):
        super().__init__()
        self.cond_encoder = cond_encoder
        self.model = None
        raise NotImplementedError("not fully implemented")

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_batch: ConditionBatch) -> torch.Tensor:
        # encode conditions
        cond_emb = self.cond_encoder(cond_batch)
        # expand t and concnatenate
        input = torch.cat([x_t, t.expand(x_t.shape[0], 1), cond_emb], dim=-1)
        # return dx/dt
        return self.model(input)
