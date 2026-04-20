"""flatcfm package"""

import sys as _sys


def _register_legacy_shims():
    """shim for backward compatibility with pickled artifacts

    deferred so that lightweight imports (eg the hydra plugin) do not
    trigger the full dependency chain
    """

    from . import data, training, models

    _sys.modules["data"] = data
    _sys.modules["data.dataset"] = data.dataset
    _sys.modules["data.simulations"] = data.simulations
    _sys.modules["data.splitters"] = data.splitters
    _sys.modules["data.types"] = data.types
    _sys.modules["data.space"] = data.space
    _sys.modules["training"] = training
    _sys.modules["training.losses"] = training.losses
    _sys.modules["models"] = models
    _sys.modules["models.autoencoder"] = models.autoencoder
    _sys.modules["models.flow"] = models.flow
    _sys.modules["models.mean_flow"] = models.mean_flow


# skip heavy imports when only lightweight submodules are needed
# (eg hydra plugin discovery)
if not _sys.modules.get("flatcfm._skip_shims"):
    try:
        _register_legacy_shims()
    except ImportError:
        pass

__all__ = [
    "analysis",
    "data",
    "modelcore",
    "models",
    "training",
]
