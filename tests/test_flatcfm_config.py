import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from flatcfm.modelcore.models import AutoencoderModel


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "flatcfm" / "configs"


def _projection_kinds(cfg) -> list[str]:
    return [item.kind for item in cfg.space.projections]


class FlatCFMConfigTest(unittest.TestCase):
    def test_compose_experiments(self):
        expected = {
            "experiment=sciplex/fm_log1p": ("fm", [], "normalized_log1p", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=sciplex/fm_log1p_all_genes": ("fm", [], "normalized_log1p", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=sciplex/fm_raw_counts": ("fm", [], "raw_counts", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=sciplex/fm_pca_all_genes": ("fm", ["pca"], "normalized_log1p", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=sciplex/fm_ae_latent": ("fm", ["ae_latent"], "normalized_log1p", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=sciplex/fm_ae_latent_recon": ("fm", ["ae_latent"], "normalized_log1p", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=sciplex/fm_ae_latent_phate": ("fm", ["ae_latent"], "normalized_log1p", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=sciplex/ode_log1p": ("ode", [], "normalized_log1p", "flatcfm.modelcore.models.NeuralODEModel"),
            "experiment=sciplex/mean_flow_ae_latent": ("mean_flow", ["ae_latent"], "normalized_log1p", "flatcfm.modelcore.models.MeanFlowModel"),
            "experiment=sciplex/ae_log1p_recon": ("ae", [], "normalized_log1p", "flatcfm.modelcore.models.AutoencoderModel"),
            "experiment=sciplex/ae_log1p_all_genes_recon": ("ae", [], "normalized_log1p", "flatcfm.modelcore.models.AutoencoderModel"),
            "experiment=sciplex/ae_log1p_phate": ("ae", [], "normalized_log1p", "flatcfm.modelcore.models.AutoencoderModel"),
            "experiment=sciplex/ae_log1p_phate_two_phase": ("ae", [], "normalized_log1p", "flatcfm.modelcore.models.AutoencoderModel"),
            "experiment=toy/fm_identity": ("fm", [], "raw_counts", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=toy/ode_identity": ("ode", [], "raw_counts", "flatcfm.modelcore.models.NeuralODEModel"),
            "experiment=toy/lifted": ("fm", ["orthogonal_lift"], "raw_counts", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=toy/fm_lifted_pca": ("fm", ["orthogonal_lift", "pca"], "raw_counts", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=toy/ae_lifted_gr": ("ae", ["orthogonal_lift"], "raw_counts", "flatcfm.modelcore.models.AutoencoderModel"),
            "experiment=toy/fm_lifted_ae_latent": ("fm", ["orthogonal_lift", "ae_latent"], "raw_counts", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=toy/fm_nonlinear_rff": ("fm", ["nonlinear_rff_lift"], "raw_counts", "flatcfm.modelcore.models.FlowMatchingModel"),
            "experiment=toy/fm_nonlinear_rff_pca": (
                "fm",
                ["nonlinear_rff_lift", "pca"],
                "raw_counts",
                "flatcfm.modelcore.models.FlowMatchingModel",
            ),
        }
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            for override, (task_name, projection_kinds, base_kind, target) in expected.items():
                cfg = compose(config_name="train", overrides=[override])
                self.assertEqual(cfg.task.name, task_name)
                self.assertEqual(_projection_kinds(cfg), projection_kinds)
                self.assertEqual(cfg.space.base.kind, base_kind)
                self.assertEqual(cfg.evaluation_space.fit_scope, "full_dataset")
                self.assertEqual(cfg.model._target_, target)

    def test_explicit_ae_artifact_tags(self):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            recon_ae = compose(config_name="train", overrides=["experiment=sciplex/ae_log1p_recon"])
            phate_ae = compose(config_name="train", overrides=["experiment=sciplex/ae_log1p_phate"])
            recon_fm = compose(config_name="train", overrides=["experiment=sciplex/fm_ae_latent_recon"])
            phate_fm = compose(config_name="train", overrides=["experiment=sciplex/fm_ae_latent_phate"])

        self.assertEqual(recon_ae.space.ae_export_artifact_tag, "sciplex_ae_log1p_recon")
        self.assertEqual(phate_ae.space.ae_export_artifact_tag, "sciplex_ae_log1p_phate")
        self.assertEqual(recon_fm.space.projections[0].artifact_tag, "sciplex_ae_log1p_recon")
        self.assertEqual(phate_fm.space.projections[0].artifact_tag, "sciplex_ae_log1p_phate")

    def test_compose_lifted_overrides(self):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=toy/lifted",
                    "task=ae",
                    "model=ae",
                    "loss=ae",
                    "ae_geometry=ambient_euclidean",
                    "space.projections.0.ambient_dim=512",
                ],
            )
        self.assertEqual(cfg.task.name, "ae")
        self.assertEqual(cfg.ae_geometry.mode, "ambient_euclidean")
        self.assertEqual(cfg.space.projections[0].kind, "orthogonal_lift")
        self.assertEqual(cfg.space.projections[0].ambient_dim, 512)

    def test_default_nb_mean_head_instantiates_autoencoder_model(self):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="train", overrides=["experiment=sciplex/ae_log1p_recon"])

        model = AutoencoderModel(
            model_cfg=cfg.model,
            task_cfg=cfg.task,
            loss_cfg={"weights": {"recon": 1.0, "distance": 0.0, "pullback": 0.0}},
            predict_cfg=cfg.predict,
            input_dim=3,
            covariate_dicts={
                "perturbation_num_categories": 1,
                "perturbation_covariates": {},
                "sample_covariates": {},
            },
            feature_names=["gene1", "gene2", "gene3"],
            schema={"output_obs_map": {}},
            space_mode="normalized_log1p",
            space_config=cfg.space,
            evaluation_space_config=cfg.evaluation_space,
        )

        self.assertEqual(model.mean_head, "per_cell_gene")
        self.assertEqual(model.model.mean_head, "per_cell_gene")

    def test_invalid_nb_mean_head_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported decoder mean head"):
            AutoencoderModel(
                model_cfg={
                    "family": "negative_binomial",
                    "latent_dim": 4,
                    "hidden_dim": 8,
                    "n_layers": 1,
                    "dropout": 0.0,
                    "decoder": {
                        "mean_head": "compositional",
                        "dispersion_head": "shared_gene",
                    },
                },
                task_cfg={"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.0},
                loss_cfg={"weights": {"recon": 1.0, "distance": 0.0, "pullback": 0.0}},
                predict_cfg={},
                input_dim=3,
                covariate_dicts={
                    "perturbation_num_categories": 1,
                    "perturbation_covariates": {},
                    "sample_covariates": {},
                },
                feature_names=["gene1", "gene2", "gene3"],
                schema={"output_obs_map": {}},
                space_mode="normalized_log1p",
                space_config={"base": {"kind": "normalized_log1p", "target_sum": 1e4}},
                evaluation_space_config={"base": {"kind": "normalized_log1p", "target_sum": 1e4}},
            )


if __name__ == "__main__":
    unittest.main()
