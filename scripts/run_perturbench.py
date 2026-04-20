"""wrapper to run perturbench training with weights_only=False for torch >= 2.6"""

import lightning.fabric.plugins.io.torch_io as tio

_orig_load = tio.TorchCheckpointIO.load_checkpoint


def _load_unsafe(self, path, map_location=None, **kwargs):
    return _orig_load(self, path, map_location=map_location, weights_only=False)


tio.TorchCheckpointIO.load_checkpoint = _load_unsafe

from perturbench.modelcore.train import main  # noqa: E402

main()
