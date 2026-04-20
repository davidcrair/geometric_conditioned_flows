"""hydra lightning prediction entrypoint"""

from __future__ import annotations

from pathlib import Path
import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from flatcfm.analysis.prediction_export import build_prediction_adata, save_prediction_artifacts
from flatcfm.analysis.perturbench_adapter import PerturbBenchTaskMetadata, to_perturbench_predictions
from flatcfm.modelcore.utils import instantiate_datamodule, instantiate_model, save_json


def _trainer_cfg(cfg: DictConfig) -> dict:
    """build trainer cfg"""

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop("num_workers", None)
    trainer_cfg.pop("pin_memory", None)
    return trainer_cfg


def _resolve_run_dir(predict_cfg: DictConfig) -> Path:
    """resolve run dir from explicit path or experiment name

    supports predict.run_dir=<path> or predict.experiment_name=<name>
    when experiment_name is given uses the most recent run directory
    """

    if predict_cfg.get("run_dir") is not None:
        return Path(predict_cfg.run_dir).expanduser().resolve()
    experiment_name = predict_cfg.get("experiment_name")
    if experiment_name is not None:
        run_root = Path("artifacts/runs") / str(experiment_name)
        run_dirs = sorted(path for path in run_root.glob("*") if path.is_dir())
        if not run_dirs:
            raise FileNotFoundError(f"no runs found under {run_root}")
        return run_dirs[-1].resolve()
    raise ValueError("predict.run_dir or predict.experiment_name must be set")


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig) -> dict:
    """main"""

    OmegaConf.set_struct(cfg.predict.target_filters, False)
    run_dir = _resolve_run_dir(cfg.predict)
    train_cfg = OmegaConf.load(run_dir / "run_config.yaml")
    with (run_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        run_metadata = json.load(handle)

    merged_cfg = OmegaConf.merge(train_cfg, {"predict": cfg.predict, "evaluation": cfg.evaluation})
    datamodule = instantiate_datamodule(merged_cfg)
    datamodule.setup("predict")
    model = instantiate_model(merged_cfg, datamodule)
    trainer = instantiate(
        _trainer_cfg(merged_cfg),
        callbacks=[],
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(run_dir),
    )

    checkpoint_path = merged_cfg.predict.get("checkpoint_path") or run_metadata.get("checkpoint_path")
    outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    pred_matrix, obs, feature_names, prediction_metadata = datamodule.export_prediction_outputs(outputs)
    pred_adata = build_prediction_adata(
        pred_matrix,
        obs=obs,
        var_names=feature_names,
        uns={
            "run_dir": str(run_dir),
            "task_name": merged_cfg.task.name,
            "prediction_name": str(merged_cfg.predict.get("name", "held_out")),
            "prediction_metadata": prediction_metadata,
        },
    )

    output_dir = run_dir / "predictions" / str(merged_cfg.predict.get("name", "held_out"))
    prediction_request = obs.reset_index(drop=True)
    artifacts = save_prediction_artifacts(
        output_dir=output_dir,
        predictions=pred_adata,
        run_config=OmegaConf.to_container(merged_cfg, resolve=True),
        prediction_request=prediction_request,
        prediction_metadata=prediction_metadata,
    )

    backend = str(merged_cfg.evaluation.get("backend", "native"))
    perturbench_predictions = None
    if backend in {"perturbench", "both"}:
        task_metadata = PerturbBenchTaskMetadata(
            perturbation_key=merged_cfg.condition.perturbation.name,
            covariate_keys=tuple(field.name for field in merged_cfg.condition.sample_covariates),
            control_value=merged_cfg.condition.control_value,
            obs_map=OmegaConf.to_container(merged_cfg.condition.output_obs_map, resolve=True),
        )
        ref_adata = datamodule.get_reference_adata(feature_names)
        perturbench_predictions = to_perturbench_predictions(
            pred_adata=pred_adata,
            ref_adata=ref_adata,
            task_metadata=task_metadata,
            model_name=str(merged_cfg.predict.get("model_name", "flatcfm")),
        )
        save_json(output_dir / "perturbench_manifest.json", {"model_names": list(perturbench_predictions.keys())})

    return {
        "output_dir": str(output_dir),
        "artifacts": {
            "predictions_path": str(artifacts.predictions_path),
            "prediction_request_path": str(artifacts.prediction_request_path),
            "prediction_metadata_path": str(artifacts.prediction_metadata_path),
        },
        "perturbench_predictions": perturbench_predictions,
    }


if __name__ == "__main__":
    main()
