import unittest

import numpy as np

from flatcfm.analysis.benchmarks.metrics import (
    compute_deg_jaccard,
    compute_deg_overlap,
    precompute_deg_info,
    precompute_true_deg_info,
)


class DegOverlapTest(unittest.TestCase):
    def setUp(self):
        self.x_ctrl = np.array(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.10, -0.10, 0.05, 0.00],
                [-0.10, 0.10, -0.05, 0.02],
                [0.05, 0.00, 0.02, -0.02],
                [-0.05, 0.05, 0.01, 0.00],
                [0.00, -0.05, -0.02, 0.01],
            ],
            dtype=float,
        )

        self.x_true = np.array(
            [
                [4.00, 3.00, 0.00, 0.00],
                [4.10, 2.90, 0.05, 0.00],
                [3.90, 3.10, -0.05, 0.02],
                [4.05, 3.00, 0.02, -0.02],
                [3.95, 3.05, 0.01, 0.00],
                [4.00, 2.95, -0.02, 0.01],
            ],
            dtype=float,
        )

        self.x_pred_none = self.x_ctrl.copy()

        self.x_pred_partial = np.array(
            [
                [4.50, 0.00, 3.20, 0.00],
                [4.60, -0.10, 3.10, 0.00],
                [4.40, 0.10, 3.30, 0.02],
                [4.55, 0.00, 3.20, -0.02],
                [4.45, 0.05, 3.15, 0.00],
                [4.50, -0.05, 3.25, 0.01],
            ],
            dtype=float,
        )

    def test_no_effect_has_zero_deg_overlap(self):
        true_deg_info = precompute_true_deg_info(self.x_true, self.x_ctrl)

        overlaps = compute_deg_overlap(
            x_pred=self.x_pred_none,
            x_ctrl=self.x_ctrl,
            true_deg_info=true_deg_info,
            ks=[1, None],
        )

        self.assertEqual(true_deg_info["n_sig"], 2)
        self.assertEqual(overlaps["DEG@1"], 0.0)
        self.assertEqual(overlaps["DEG@N"], 0.0)

    def test_deg_overlap_uses_filtered_predicted_deg_set(self):
        true_deg_info = precompute_true_deg_info(self.x_true, self.x_ctrl)
        pred_deg_info = precompute_deg_info(self.x_pred_partial, self.x_ctrl)

        overlaps = compute_deg_overlap(
            x_pred=self.x_pred_partial,
            x_ctrl=self.x_ctrl,
            true_deg_info=true_deg_info,
            pred_deg_info=pred_deg_info,
            ks=[1, 2, None],
        )

        self.assertEqual(true_deg_info["n_sig"], 2)
        self.assertEqual(pred_deg_info["n_sig"], 2)
        self.assertEqual(set(true_deg_info["ranked_indices"][:2]), {0, 1})
        self.assertEqual(set(pred_deg_info["ranked_indices"][:2]), {0, 2})
        self.assertEqual(overlaps["DEG@1"], 1.0)
        self.assertEqual(overlaps["DEG@2"], 0.5)
        self.assertEqual(overlaps["DEG@N"], 0.5)

    def test_deg_jaccard_reports_partial_overlap(self):
        true_deg_info = precompute_true_deg_info(self.x_true, self.x_ctrl)
        scores = compute_deg_jaccard(
            x_pred=self.x_pred_partial,
            x_ctrl=self.x_ctrl,
            true_deg_info=true_deg_info,
            ks=(1, 2),
        )

        self.assertEqual(scores["deg_jaccard@1"], 1.0)
        self.assertAlmostEqual(scores["deg_jaccard@2"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
