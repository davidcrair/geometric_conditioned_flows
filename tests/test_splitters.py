import unittest

import anndata as ad
import numpy as np
import pandas as pd

from flatcfm.data.splitters import (
    SplitConfig,
    apply_holdout_masks,
    build_holdout_manifest,
    select_stratified_cell_names,
    select_subsample_cell_names,
    validate_no_leakage,
)


def _make_toy_adata(n_per_group: int = 6) -> ad.AnnData:
    product_names = ["drugA", "drugB", "drugC", "drugD"]
    cell_types = ["K562", "A549"]

    obs_rows = []
    idx = 0
    for cell_type in cell_types:
        for product in product_names:
            for _ in range(n_per_group):
                obs_rows.append(
                    {
                        "obs_name": f"cell_{idx}",
                        "product_name": product,
                        "product_dose": f"{product}_10",
                        "cell_type": cell_type,
                        "vehicle": 0,
                        "replicate": "R1",
                    }
                )
                idx += 1
        for _ in range(n_per_group):
            obs_rows.append(
                {
                    "obs_name": f"cell_{idx}",
                    "product_name": "vehicle",
                    "product_dose": "vehicle_0",
                    "cell_type": cell_type,
                    "vehicle": 1,
                    "replicate": "R1",
                }
            )
            idx += 1

    obs_df = pd.DataFrame(obs_rows).set_index("obs_name")
    x = np.zeros((len(obs_df), 8), dtype=np.float32)
    return ad.AnnData(X=x, obs=obs_df)


class SplittersTest(unittest.TestCase):
    def test_holdout_manifest_is_deterministic(self) -> None:
        adata = _make_toy_adata()
        cfg = SplitConfig(seed=42, test_cell_type="K562", holdout_fraction=0.5)
        m1 = build_holdout_manifest(adata, cfg)
        m2 = build_holdout_manifest(adata, cfg)
        self.assertEqual(m1["selected_holdout_product_names"], m2["selected_holdout_product_names"])

    def test_subsample_is_deterministic(self) -> None:
        adata = _make_toy_adata()
        s1 = select_subsample_cell_names(adata, n_cells=15, seed=7)
        s2 = select_subsample_cell_names(adata, n_cells=15, seed=7)
        self.assertEqual(s1, s2)

    def test_stratified_subsample_preserves_groups(self) -> None:
        adata = _make_toy_adata(n_per_group=5)
        chosen = select_stratified_cell_names(
            adata,
            n_cells=20,
            seed=42,
            group_cols=("cell_type", "vehicle"),
        )
        obs = adata[chosen].obs
        groups = set(obs[["cell_type", "vehicle"]].astype(str).agg("|".join, axis=1).tolist())
        expected = set(adata.obs[["cell_type", "vehicle"]].astype(str).agg("|".join, axis=1).unique().tolist())
        self.assertTrue(expected.issubset(groups))

    def test_apply_holdout_masks_marks_selected_products(self) -> None:
        adata = _make_toy_adata()
        cfg = SplitConfig(seed=42, test_cell_type="K562", holdout_fraction=0.5)
        manifest = build_holdout_manifest(adata, cfg)
        masks = apply_holdout_masks(adata, manifest)

        held_products = set(adata.obs.loc[masks["is_held_out"], "product_name"].unique().tolist())
        selected = set(manifest["selected_holdout_product_names"])
        self.assertEqual(selected, held_products)

        # Ensure controls are never marked held out in strict mode.
        self.assertFalse(np.any(masks["is_held_out"] & (adata.obs["vehicle"].to_numpy() == 1)))

    def test_validate_no_leakage_pass_and_fail(self) -> None:
        adata = _make_toy_adata()
        cfg = SplitConfig(seed=42, test_cell_type="K562", holdout_fraction=0.5)
        manifest = build_holdout_manifest(adata, cfg)
        masks = apply_holdout_masks(adata, manifest)

        validate_no_leakage(adata, masks)

        bad_masks = dict(masks)
        bad_masks["is_train"] = np.ones(adata.n_obs, dtype=bool)
        with self.assertRaises(ValueError):
            validate_no_leakage(adata, bad_masks)


if __name__ == "__main__":
    unittest.main()
