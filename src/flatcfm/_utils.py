"""shared low-level utilities"""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import OmegaConf


def to_plain_dict(cfg: Any) -> dict:
    """convert config to plain dict"""

    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)


def dense_array(x: Any) -> np.ndarray:
    """convert matrix to dense numpy array"""

    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)
