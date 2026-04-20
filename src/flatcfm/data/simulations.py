"""
toy dataset generators
"""

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.datasets import make_moons


def make_gaussian_to_moons(n_samples=2000) -> ad.AnnData:
    """generate a toy dataset (anndata object) where the control is gaussian and perturbed is two moons where each is a different perturbation"""

    rng = np.random.default_rng(0)

    # Gaussian control
    x_ctrl = rng.standard_normal((n_samples, 2)) * 0.5

    # Moons perturbations
    # make_moons with N returns N samples total.
    # We want 2 perturbations, each being one moon.
    # So we generate n_samples * 2 moon samples total.
    x_moons, moon_labels = make_moons(n_samples=n_samples * 2, noise=0.05, random_state=0)

    x_all = np.concatenate([x_ctrl, x_moons], axis=0)
    n_ctrl = x_ctrl.shape[0]
    n_moons = x_moons.shape[0]
    n_total = n_ctrl + n_moons

    obs = pd.DataFrame(
        {
            # vehicle=1.0 is control (Gaussian), vehicle=0.0 is perturbed (Moons)
            "vehicle": np.array([1.0] * n_ctrl + [0.0] * n_moons, dtype=float),
            # perturbation=0 is control, perturbation=1 and 2 are moons
            "perturbation": np.array([0.0] * n_ctrl + (moon_labels + 1).tolist(), dtype=float),
            "cell_type": ["Toy Cell"] * n_total,
        },
        index=[str(i) for i in range(n_total)],
    )

    adata = ad.AnnData(X=x_all.astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()

    return adata
