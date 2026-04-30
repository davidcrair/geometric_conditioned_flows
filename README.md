# GLFM: Geometry-Regularized Latent Flow Models for Single-Cell Perturbation Prediction
fka flatcfm (still used in this repo name and some config names for historical reasons)

Flow matching on flat latent spaces for single-cell perturbation modeling. Predicts heterogeneous single-cell transcriptional responses to drug perturbations in underrepresented contexts.

Senior thesis project. Repo structure inspired by [perturbench](https://github.com/altoslabs/perturbench).

## Setup

Requires Python >= 3.12. Uses `uv` for package management.

```bash
uv sync
```

## Quick start

```bash
# train a flow matching model on sciplex data in log1p HVG space
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/fm_log1p \
  experiment_name=sciplex_fm_hvg \
  trainer.precision=bf16-mixed

# generate predictions (auto-picks latest run)
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict \
  predict.experiment_name=sciplex_fm_hvg

# evaluate
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg
```

All outputs go to `artifacts/runs/{experiment_name}/{timestamp}/`.

## Data

Default sciplex splits (`strict_k562`, `strict_mcf7`, `strict_a549`) draw a canonical 100k-cell subsample from the full pertpy `sciplex3_raw()` dataset, with a separate 50k-cell subsample reserved for AE training. The K562 split holds out 50% of K562 perturbations as the test set. The `perturbench_k562` split uses 150k cells (100k for AE) to match the perturbench protocol. Subsample selection is seeded so the same cells are reused across runs.

## Space hierarchy

```
raw space       all genes, integer counts
    | base transform (log1p normalize + select 2000 HVGs)
base space      2000 HVGs, normalized -- all metrics computed here
    | projections (pca, ae_latent, orthogonal_lift, etc.)
training space  where the model operates (50 PCA dims, 128 AE latent, etc.)
```

Predictions are always exported in **base space** so models trained in different projection spaces (PCA vs AE latent) can be directly compared.

## Configuration (Hydra)

The project uses [Hydra](https://hydra.cc/) for configuration. Configs live in `src/flatcfm/configs/` and compose hierarchically.

### Config groups

| Group | Location | Purpose |
|---|---|---|
| `experiment` | `experiment/sciplex/`, `experiment/toy/` | Presets that override multiple groups at once |
| `space` | `space/` | Feature space and projections for training |
| `evaluation_space` | `evaluation_space/` | Space for evaluation (default: same as training) |
| `data` | `data/` | Data source (`sciplex`, `toy`) |
| `task` | `task/` | Task type, epochs, learning rate |
| `model` | `model/` | Architecture (hidden dim, layers, dropout) |
| `loss` | `loss/` | Loss function and component weights |
| `splitter` | `splitter/` | Data splitting strategy |
| `condition` | `condition/` | Perturbation and covariate definitions |
| `ae_geometry` | `ae_geometry/` | AE geometry (`phate_potential`, `ambient_euclidean`, `none`) |
| `trainer` | `trainer/` | PyTorch Lightning trainer settings |

### Available spaces

| Config | Base | Projection | Training dim |
|---|---|---|---|
| `log1p` | log1p + 2000 HVGs | none | 2000 |
| `pca` | log1p + 2000 HVGs | PCA 50 | 50 |
| `ae_latent` | log1p + 2000 HVGs | AE encoder | 128 |
| `orthogonal_lift` | log1p + 2000 HVGs | random orthogonal lift | 4000 |
| `nonlinear_rff_lift` | log1p + 2000 HVGs | random fourier features | 4000 |
| `log1p_all_genes` | log1p + all genes | none | ~20k |
| `pca_all_genes` | log1p + all genes | PCA 50 | 50 |
| `raw_counts` | raw integer counts | none | varies |

Compound spaces like `orthogonal_lift_pca` and `nonlinear_rff_lift_ae_latent` chain multiple projections.

### Experiment presets

Experiment configs override multiple groups at once. Use them as starting points:

```bash
# flow matching in log1p space (no projection)
experiment=sciplex/fm_log1p

# flow matching in PCA space
experiment=sciplex/fm_log1p space=pca

# flow matching in MSE AE latent (d=512)
experiment=sciplex/fm_ae_latent_mse_d512

# flow matching in AE latent space with PHATE geometry
experiment=sciplex/fm_ae_latent_phate

# flow matching on AE latent with geometric regularization
# (built from the hybrid base experiment plus a loss override; see
# scripts/train_geom_reg_sweep.sbatch)
experiment=sciplex/fm_ae_latent_hybrid

# ODE counterpart
experiment=sciplex/ode_ae_latent_mse_d512

# analytical baselines (no_effect, additive, context_mean, perturb_mean,
# decoder, linear, latent_additive)
experiment=sciplex/baseline_no_effect
```

See `src/flatcfm/configs/experiment/sciplex/` for the full list.

### Overriding config values

Any config value can be overridden from the command line:

```bash
# change epochs and learning rate
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/fm_log1p \
  task.epochs=200 \
  task.lr=0.0005 \
  model.hidden_dim=512

# change number of PCA components
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/fm_log1p \
  space=pca \
  space.projections.0.n_components=100
```

## Training

```bash
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/fm_log1p \
  experiment_name=sciplex_fm_hvg \
  trainer.precision=bf16-mixed
```

Output structure:
```
artifacts/runs/sciplex_fm_hvg/20260315_153557/
  checkpoints/best.ckpt    # best model checkpoint
  run_config.yaml           # full resolved hydra config
  run_metadata.json         # feature names, covariate dicts, task metadata
  history.json              # train/val loss curves per epoch
  metrics.csv               # lightning metrics log
```

### Training baselines

Train all baseline methods at once:

```bash
./scripts/train_sciplex_baselines.sh
```

This trains no_effect, additive, context_mean, perturb_mean (1 epoch each), then decoder and linear (100 epochs each), and generates predictions for all.

Set `FORCE_RETRAIN=1` to retrain even if checkpoints exist. Set `FORCE_PREDICT=1` to regenerate predictions.

## Prediction

```bash
# by experiment name (uses most recent run)
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict \
  predict.experiment_name=sciplex_fm_hvg

# by explicit run directory
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict \
  predict.run_dir=artifacts/runs/sciplex_fm_hvg/20260315_153557
```

Predictions are saved to `{run_dir}/predictions/held_out/predictions.h5ad`.

## Evaluation

### Model card

View a structured summary of how a model was trained:

```bash
.venv/bin/python scripts/model_card.py sciplex_fm_deg_ae_latent_phate
```

Shows panels for model identity, space, architecture, training, loss, data split, and (for AEs) schedule and geometry config. Accepts experiment names or full run directory paths.

### CLI evaluation

```bash
# single run
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg

# compare multiple runs side by side
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg sciplex_fm_pca_hvg

# evaluate all runs of an experiment (mean +/- std across runs with matching config)
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg --all-runs

# loss curves only (no metric computation)
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg --losses-only

# skip loss curves
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg --no-losses

# custom metrics and reduction
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg \
  --metrics mean_gene_w1,w2_squared,cosine_log_fc \
  --reduction cell_weighted_mean
```

`run_eval.py` accepts either experiment names (resolves to latest run) or full run directory paths.

### PerturBench evaluation

Models can be trained and evaluated through the [perturbench](https://github.com/altoslabs/perturbench) pipeline using the FlatCFM model wrappers in `src/flatcfm/perturbench/`. This uses perturbench's gene space and metrics (not the FlatCFM evaluation above).

Use the `experiment=sciplex3/<name>` invocation below — these pin `data/evaluation=final_test` so results are evaluated on the test split, matching the PerturBench Table 2 protocol. Invoking via `data=sciplex3 model=<name>` directly falls back to the default evaluation which uses the val split and is **not comparable to Table 2**.

```bash
# flow matching (full gene space)
.venv/bin/python scripts/run_perturbench.py experiment=sciplex3/flatcfm_fm_sciplex3

# flow matching in PCA16 latent
.venv/bin/python scripts/run_perturbench.py experiment=sciplex3/flatcfm_fm_pca16_sciplex3

# neural ODE (full gene space)
.venv/bin/python scripts/run_perturbench.py experiment=sciplex3/flatcfm_ode_sciplex3

# neural ODE in PCA16 latent
.venv/bin/python scripts/run_perturbench.py experiment=sciplex3/flatcfm_ode_pca16_sciplex3

# decoder baseline
.venv/bin/python scripts/run_perturbench.py experiment=sciplex3/flatcfm_decoder_sciplex3

# latent additive baseline (from perturbench)
.venv/bin/python scripts/run_perturbench.py experiment=neurips2025/sciplex3/latent_best_params_sciplex3

# view all perturbench results (bold = best, underline = second best)
.venv/bin/python scripts/view_perturbench.py

# show all metrics including rank metrics
.venv/bin/python scripts/view_perturbench.py --all-metrics
```

Results are saved to perturbench's logs directory and picked up automatically by `view_perturbench.py`.

### Dimensionality reduction comparison

Compare theoretical reconstruction limits of PCA vs autoencoders (MSE, FlatVI, PHATE) at latent dims 16-1024.

```bash
# check which models are trained and which need training
.venv/bin/python scripts/check_dim_sweep.py

# generate training commands for missing models
.venv/bin/python scripts/check_dim_sweep.py --generate

# train a single model (example: MSE AE at dim 64)
.venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/ae_log1p_mse_d512 \
  model.latent_dim=64 \
  experiment_name=sciplex_ae_deg_mse_d64 \
  space.ae_export_artifact_tag=sciplex_ae_deg_mse_d64
```

After training, open `view_theoretical_limits.ipynb` to plot cosine logfc and W2 vs variance explained / latent dim for all methods. The notebook skips missing models, so it can be run at any point during the sweep.

### Sweep tooling

Each major sweep ships with a checker script and an sbatch launcher:

| Sweep | Checker | Launcher | Viewer |
|---|---|---|---|
| FM AE MSE latent dim | `scripts/check_fm_ae_mse_sweep.py` | `scripts/train_ae_sweep.sbatch` | `view_fm_ae_mse_sweep.ipynb` |
| FM PCA dim | `scripts/check_fm_pca_sweep.py` | `scripts/train_fm_pca_allctrl.sbatch` | `view_gen_flows_pca_sweep.ipynb` |
| ODE PCA dim | `scripts/check_ode_pca_sweep.py` | `scripts/train_ode_pca_sweep.sbatch` | (use `view_gen_flows_pca_sweep.ipynb`) |
| Geometric reg weight | `scripts/check_geom_reg_sweep.py` | `scripts/train_geom_reg_sweep.sbatch` (and per-weight `_seeds.sbatch` variants) | `view_geom_reg_sweep.ipynb` |
| Theoretical limits (PCA / AE) | `scripts/check_dim_sweep.py` | `scripts/train_phate_mse_ae_sweep.sbatch` | `view_theoretical_limits.ipynb` |

Run a checker with `--generate` to print the training commands that are still missing.

### Available metrics

| Metric | Direction | Description |
|---|---|---|
| `mean_gene_w1` | lower is better | mean gene-wise 1-Wasserstein distance |
| `w2_squared` | lower is better | 2-Wasserstein distance squared (in PCA space) |
| `cosine_log_fc` | higher is better | cosine similarity of log fold-change vectors |
| `top_k_recall` | higher is better | recall of top-K differentially expressed genes |
| `deg_jaccard` | higher is better | Jaccard similarity of DEG sets |
| `e_distance` | lower is better | energy distance |
| `mmd` | lower is better | maximum mean discrepancy |

## Notebooks

| Notebook | Purpose |
|---|---|
| `view_flow_results.ipynb` | Inspect a single flow matching or ODE run: loss curves, perturbation ranking by W1, five-way distribution overlays for selected genes |
| `view_benchmark_results.ipynb` | Compare multiple models (flow + baselines) via the benchmark suite API: per-group metrics, summary tables, box plots, dose breakdowns, LaTeX results table |
| `view_benchmark_results_cross_ct.ipynb` | Cross-cell-type benchmark comparison |
| `view_autoencoder_results.ipynb` | Inspect trained autoencoder: reconstruction quality, latent space |
| `view_dataset_embeddings.ipynb` | Visualize sciplex embeddings in raw and projected spaces |
| `view_theoretical_limits.ipynb` | Compare PCA vs AE (MSE, PHATE, FlatVI) reconstruction limits across latent dims |
| `view_pca_theoretical_limits.ipynb` | PCA-only reconstruction limit sweep |
| `view_fm_ae_mse_sweep.ipynb` | Flow matching results across MSE AE latent dim sweep |
| `view_gen_flows_pca_sweep.ipynb` | Generative flow results across PCA dim sweep |
| `view_geom_reg_sweep.ipynb` | Geometric regularization weight sweep results |
| `view_toy_results.ipynb` | Toy dataset experiments: compare projection methods and architectures |
| `toy-orthogonal-lift-fm-vs-grae.ipynb` | Flow matching vs GRAE comparison on toy data with orthogonal lifts |

### Running the benchmark notebook

`view_benchmark_results.ipynb` requires predictions to be pre-generated for all models. Typical workflow:

```bash
# 1. train and predict flow model
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/fm_log1p experiment_name=sciplex_fm_hvg trainer.precision=bf16-mixed
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict \
  predict.experiment_name=sciplex_fm_hvg

# 2. train and predict all baselines
./scripts/train_sciplex_baselines.sh

# 3. open the notebook
jupyter lab view_benchmark_results.ipynb
```

## Project layout

```
src/flatcfm/
  configs/           # hydra configuration hierarchy
  data/              # data loading, splitting, space transforms, geometry
    datamodules.py   # SciplexDataModule, ToyDataModule (lightning)
    geometry.py      # TransformPipeline, BaseTransform, ProjectionTransform
    space.py         # pipeline construction, labels, config normalization
    splitters.py     # OOD context splits, stratified sampling
    dataset.py       # CondFMDataset (pytorch)
  modelcore/         # training and prediction entrypoints
    train.py         # hydra training entrypoint
    predict.py       # hydra prediction entrypoint
    models/          # lightning modules (flow_matching, autoencoder, baselines)
    predictors/      # benchmark predictor implementations
  models/            # neural network architectures
    autoencoder.py   # NBAutoEncoder (negative binomial)
    flow.py          # CondFlow, ConditionEncoder
  training/          # loss functions
    losses.py        # composable loss registry
  analysis/          # evaluation and benchmarking
    benchmarking.py  # benchmark suite runner
    benchmarks/      # metric computation, aggregation, metric spaces
    flow_results.py  # load and inspect flow runs
scripts/
  run_eval.py        # CLI evaluation tool
  model_card.py      # structured run summary
  view_perturbench.py # view perturbench results across runs
  run_perturbench.py # train models through perturbench pipeline
  run_status.py      # monitor training runs
  gpu_status.py      # GPU utilization snapshot
  common.sh          # shared bash helpers (latest_run_dir, ensure_train, ensure_predict)
  train_sciplex_baselines.sh  # train all baselines
  train_toy_*.sh     # toy experiment grid sweeps
  check_*_sweep.py   # report which sweep configs are trained vs missing
  train_*_sweep.sbatch # SLURM array launchers for each sweep
  build_drug_embeddings.py  # precompute molformer drug embeddings
  materialize_allctrl_subsample.py  # materialize the all-controls subsample h5ad
artifacts/
  runs/              # training outputs (gitignored)
  models/            # pretrained AE checkpoints
  spaces/            # cached fitted pipelines
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/
```
