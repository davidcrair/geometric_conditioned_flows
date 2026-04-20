import unittest

import anndata as ad
import numpy as np
import pandas as pd

from flatcfm.analysis.perturbench_adapter import PerturbBenchTaskMetadata, to_perturbench_predictions


class FlatCFMAdapterTest(unittest.TestCase):
    def test_to_perturbench_predictions_renames_and_aligns(self):
        pred = ad.AnnData(
            X=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            obs=pd.DataFrame({"perturbation": ["drugA_10"], "cell_type": ["K562"]}),
        )
        pred.var_names = ["gene2", "gene1", "gene3"]

        ref = ad.AnnData(
            X=np.array([[0.0, 0.0]], dtype=np.float32),
            obs=pd.DataFrame(index=["ctrl"]),
        )
        ref.var_names = ["gene1", "gene2"]

        task = PerturbBenchTaskMetadata(
            perturbation_key="condition",
            covariate_keys=("cell_type",),
            obs_map={"perturbation": "condition"},
        )
        out = to_perturbench_predictions(pred, ref, task, model_name="flatcfm")
        self.assertIn("flatcfm", out)
        converted = out["flatcfm"]
        self.assertEqual(list(converted.var_names), ["gene1", "gene2"])
        self.assertIn("condition", converted.obs.columns)
        self.assertIn("cell_type", converted.obs.columns)


if __name__ == "__main__":
    unittest.main()
