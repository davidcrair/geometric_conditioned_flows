"""base adapter bridging flatcfm models to perturbench's PerturbationModel interface"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from perturbench.modelcore.models.base import PerturbationModel
from perturbench.data.types import Batch

# perturbench checkpoints pickle sklearn/transform objects in training_record
# which torch.load(weights_only=True) rejects (torch >= 2.6)
# override to use weights_only=False for checkpoint loading
import lightning.fabric.plugins.io.torch_io as _torch_io

_orig_load = _torch_io.TorchCheckpointIO.load_checkpoint


def _load_checkpoint_unsafe(self, path, map_location=None, **kwargs):
    return _orig_load(self, path, map_location=map_location, weights_only=False)


_torch_io.TorchCheckpointIO.load_checkpoint = _load_checkpoint_unsafe


class FlatCFMAdapter(PerturbationModel, ABC):
    """base class for flatcfm models running inside perturbench

    handles batch translation between perturbench's Batch namedtuple
    and flatcfm's tensor conventions and builds flatcfm-style
    covariate_dicts from perturbench's train_context

    if pca_n_components is set the model trains in PCA space and
    predictions are inverse-transformed back to gene space for
    perturbench evaluation
    """

    def __init__(self, datamodule=None, lr=None, wd=None,
                 n_genes=None, n_perts=None,
                 pca_n_components=None, model_name=None, **pb_kwargs):
        # n_genes and n_perts come from hydra config (set to null) but
        # PerturbationModel extracts them from datamodule so we drop them
        super().__init__(datamodule=datamodule, lr=lr, wd=wd, **pb_kwargs)
        self._custom_model_name = model_name
        self.save_hyperparameters(ignore=["datamodule"])

        # fit PCA on training data if requested
        self._pca_n_components = pca_n_components
        if pca_n_components is not None and datamodule is not None:
            self._fit_pca(datamodule)
            # override n_genes so subclass constructors build models
            # with PCA dimensionality
            self.n_genes = pca_n_components
        else:
            self.register_buffer("_pca_components", None)
            self.register_buffer("_pca_mean", None)

        if datamodule is not None:
            self._covariate_dicts = self._build_covariate_dicts(
                datamodule.train_context
            )

    def _fit_pca(self, datamodule):
        """fit PCA on training gene expression and store as buffers"""

        from sklearn.decomposition import PCA

        X = datamodule.train_dataset.gene_expression
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        pca = PCA(n_components=self._pca_n_components)
        pca.fit(X)
        self.register_buffer(
            "_pca_components",
            torch.from_numpy(pca.components_.astype(np.float32)),
        )
        self.register_buffer(
            "_pca_mean",
            torch.from_numpy(pca.mean_.astype(np.float32)),
        )

    @property
    def _has_pca(self) -> bool:
        return self._pca_components is not None

    def _to_pca(self, x: torch.Tensor) -> torch.Tensor:
        """project gene-space tensor to PCA space"""

        return (x - self._pca_mean) @ self._pca_components.T

    def _from_pca(self, x: torch.Tensor) -> torch.Tensor:
        """inverse-project PCA-space tensor back to gene space"""

        return x @ self._pca_components + self._pca_mean

    def _build_covariate_dicts(self, train_context: dict) -> dict:
        """translate perturbench train_context to flatcfm covariate_dicts format

        perturbench provides:
          train_context["perturbation_uniques"]: list of perturbation strings
          train_context["covariate_uniques"]: dict of {name: set of values}

        flatcfm expects:
          covariate_dicts["perturbation_num_categories"]: int
          covariate_dicts["sample_covariates"]: {name: n_categories}
          covariate_dicts["perturbation_covariates"]: {}
        """

        covariate_dicts = {
            "perturbation_num_categories": self.n_perts,
            "sample_covariates": {},
            "perturbation_covariates": {},
        }
        for cov_name, cov_uniques in train_context.get("covariate_uniques", {}).items():
            covariate_dicts["sample_covariates"][cov_name] = len(cov_uniques)
        return covariate_dicts

    def _n_total_covariates(self) -> int:
        """total one-hot covariate width from train_context"""

        return sum(
            len(v)
            for v in self.training_record["train_context"]["covariate_uniques"].values()
        )

    def training_step(self, batch: Batch, batch_idx: int):
        observed, control, pert, covs, _ = self.unpack_batch(batch)
        if self._has_pca:
            observed = self._to_pca(observed)
            control = self._to_pca(control)
        loss = self._compute_loss(observed, control, pert, covs)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, batch_size=observed.size(0)
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        observed, control, pert, covs, _ = self.unpack_batch(batch)
        if self._has_pca:
            observed = self._to_pca(observed)
            control = self._to_pca(control)
        loss = self._compute_loss(observed, control, pert, covs)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=observed.size(0),
        )
        return loss

    def predict(self, counterfactual_batch: Batch) -> torch.Tensor:
        control = counterfactual_batch.gene_expression.squeeze().to(self.device)
        pert = counterfactual_batch.perturbations.squeeze().to(self.device)
        covs = {k: v.to(self.device) for k, v in counterfactual_batch.covariates.items()}
        if self._has_pca:
            control = self._to_pca(control)
        pred = self._generate_prediction(control, pert, covs)
        if self._has_pca:
            pred = self._from_pca(pred)
        return pred.float()

    def _get_model_name(self) -> str:
        """model name used in perturbench evaluation summaries"""

        if self._custom_model_name is not None:
            return self._custom_model_name
        return str(self.__class__).split(".")[-1].replace("'>", "")

    def test_step(self, data_tuple, batch_idx):
        """override to inject custom model name into evaluation"""

        import anndata as ad
        from perturbench.analysis.benchmarks.evaluation import Evaluation

        counterfactual_batch, counterfactual_obs, reference_adata = data_tuple
        train_context = self.training_record["train_context"]
        model_name = self._get_model_name()

        predicted_adata = self.predict_step(
            (counterfactual_batch, counterfactual_obs),
            batch_idx,
        )

        control_adata = reference_adata[
            reference_adata.obs[train_context["perturbation_key"]]
            == train_context["perturbation_control_value"]
        ]
        assert control_adata.shape[0] > 0

        predicted_adata = predicted_adata[
            predicted_adata.obs[train_context["perturbation_key"]]
            != train_context["perturbation_control_value"]
        ]
        predicted_adata = ad.concat([predicted_adata, control_adata])
        predicted_adata.obs_names_make_unique()

        ev = Evaluation(
            model_adatas=[predicted_adata],
            model_names=[model_name],
            ref_adata=reference_adata,
            pert_col=train_context["perturbation_key"],
            cov_cols=train_context["covariate_keys"],
            ctrl=train_context["perturbation_control_value"],
        )
        for aggr in self.unique_aggregations:
            ev.aggregate(aggr_method=aggr)
        del ev.adatas
        ev.adatas = None
        self.evaluation_list.append(ev)

        import gc
        gc.collect()

    @abstractmethod
    def _compute_loss(
        self,
        observed: torch.Tensor,
        control: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def _generate_prediction(
        self,
        control: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ...
