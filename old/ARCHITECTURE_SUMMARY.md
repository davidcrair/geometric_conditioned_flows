# FlatCFM Architecture Summary

## Purpose

FlatCFM is a perturbation prediction codebase for single-cell expression data

The main real-data task is

- take a control cell
- condition on a requested perturbation and context
- predict the perturbed cell state distribution
- evaluate those predictions against held-out real perturbed cells

The default real dataset is SciPlex3

The default split is an **underrepresented-context** split rather than a pure unseen-drug split

- the code samples held-out `product_name` values from the chosen test cell type `K562`
- only perturbed K562 cells for those products are marked held out
- the same products can still appear in other cell types during training
- controls are not held out

So the intended generalization problem is

- infer perturbation responses in a context where that perturbation is underrepresented or absent
- use perturbation information seen elsewhere plus context information to extrapolate into the held-out cell type

There is also a toy pipeline with synthetic data that uses the same interfaces

## Main packages

The code is organized into five main layers

- `src/flatcfm/data`
  data loading
  split construction
  condition schema handling
  space transforms
  AE dataloaders
- `src/flatcfm/models`
  neural architectures
  autoencoders
  condition encoder
  flow models
  mean-flow model
- `src/flatcfm/modelcore/models`
  Lightning wrappers around the architectures
  training logic per task
  baselines
- `src/flatcfm/training`
  modular loss functions
- `src/flatcfm/analysis`
  prediction export
  benchmark metrics
  evaluation helpers
  run-loading utilities

## End-to-end data flow

At a high level the system does this

1. load raw `AnnData`
2. build train and evaluation space pipelines
3. encode categorical condition fields into integer tensors
4. create task-specific dataloaders
5. train one of
   - autoencoder
   - flow matching model
   - neural ODE model
   - mean-flow model
   - baseline
6. predict held-out perturbed cells from matched control cells
7. optionally decode back to raw counts and re-encode into a different evaluation space
8. score predictions with native metrics and optionally export in PerturbBench format

## Task definition

### Real task

The default real task is defined by the config stack in `src/flatcfm/configs/train.yaml`

Default pieces

- data: `sciplex`
- splitter: `strict_k562.yaml`
  despite the filename the default policy inside it is `underrepresented_context`
- space: `log1p`
- condition: `sciplex`
- task: `fm`
- model: `fm`
- loss: `fm`

The default prediction request is

- split: `held_out`
- target subset: `perturbed`
- controls source: `all_controls`
- control matching mode: `test_cell_type`

That means the model is evaluated on held-out perturbed cells and gets matched control cells from the test cell type

### Condition schema

The real-data schema comes from `src/flatcfm/configs/condition/sciplex.yaml`

- perturbation source column: `product_dose`
- control column: `vehicle`
- control value: `1`
- sample covariates: `cell_type`
- perturbation covariates: none by default

This creates a condition that is effectively

- requested perturbation identity
- requested cell type context

### Split logic

Split logic lives in `src/flatcfm/data/splitters.py`

Important behavior

- a deterministic manifest chooses held-out products in the test cell type
- `apply_holdout_masks` creates `is_train`, `is_held_out`, `is_ctrl`, and related masks
- in `underrepresented_context` mode only perturbed cells in the held-out cell type are removed from train
- `validate_no_leakage` skips product-overlap checks in `underrepresented_context` mode because overlap across other contexts is intentional

### Prediction target and control selection

Prediction selection lives in `PredictionBuilder` in `src/flatcfm/data/_datamodule_parts.py`

It supports

- split selection: `held_out`, `train`, `val`
- target subsets: `perturbed`, `control`, `all`
- obs filters via exact match and membership filters
- control pools from `all_controls`, `train_controls`, or `val_controls`
- control matching by sample covariates

Control matching is not expression-nearest-neighbor matching

Instead it is round-robin matching inside groups defined by the sample covariates from the schema

With the default SciPlex config that means controls are matched by `cell_type`

## Two kinds of condition encoding

There are two separate condition encoders in the codebase

### 1 data-side categorical encoding

`ConditionEncoder` in `src/flatcfm/data/_datamodule_parts.py` is a preprocessing helper

It does not learn anything

It

- scans `adata.obs`
- builds vocab maps from strings to integer ids
- creates tensors for
  - perturbation ids
  - perturbation covariate ids
  - sample covariate ids

Those integer tensors are what the models receive in `cond_batch`

### 2 model-side neural condition encoder

`ConditionEncoder` in `src/flatcfm/models/flow.py` is the learned neural encoder used by FM ODE and linear baselines

Architecture

- one embedding table for perturbations
- one embedding table per perturbation covariate
- one embedding table per sample covariate
- each embedding is projected with `Linear -> ReLU`
- all projected pieces are concatenated
- a final `Linear -> ReLU -> Linear` MLP produces the condition embedding

Default dimensions from configs

- condition output dim: `128`
- embedding dim: `64`
- per-field projection dim: `64`

For the default SciPlex setup there are two learned inputs

- perturbation embedding
- cell type embedding

## Space system and normalization

The space system is central to the whole repo

It is implemented in

- `src/flatcfm/data/space.py`
- `src/flatcfm/data/geometry.py`

### Base transforms

Every space starts with a base transform

Supported base kinds

- `raw_counts`
- `normalized_log1p`

Both base transforms can operate on

- all genes
- HVGs only

For `normalized_log1p`

- full raw library size is computed from **all genes**
- the selected gene subset is divided by that full library size
- the result is scaled by `target_sum`
- `log1p` is applied

Default real-data base config is

- kind: `normalized_log1p`
- feature set: `hvg`
- number of HVGs: `2000`
- target sum: `10000`

So the model usually trains on log1p-normalized HVGs but still carries the full-cell library size alongside each example

### Projections

After the base transform the pipeline can apply zero or more projections

Supported projections

- `identity`
- `pca`
- `orthogonal_lift`
- `nonlinear_rff_lift`
- `ae_latent`

What they do

- `pca`
  standard PCA projection with inverse transform
- `orthogonal_lift`
  random orthonormal lift into a higher-dimensional ambient space
  exactly invertible back to the original coordinates
- `nonlinear_rff_lift`
  keep the original coordinates and append random Fourier features
  inverse transform just drops the appended RFF coordinates
- `ae_latent`
  use a trained autoencoder as the final projection

### Transform pipeline behavior

`TransformPipeline` does three important things

- `transform`
  raw `AnnData` to model-space matrix plus library size
- `transform_raw`
  raw matrix plus explicit library size to model space
- `inverse_to_raw`
  model-space matrix plus library size back to raw counts

This is how the repo can

- train in one space
- export predictions in another space
- benchmark in a projection-free comparison space

### Training space versus evaluation space

The training space and evaluation space are separate configs

By default evaluation uses `same_as_training`

But that does **not** mean the exact fitted pipeline object is reused

The default evaluation config copies the training space **and refits it on `full_dataset`**

That matters because

- HVG selection can differ
- PCA basis can differ
- fitted pipelines can differ even when the space family name is the same

When the fitted training and evaluation pipeline specs differ the code roundtrips predictions through raw counts before exporting them

### Comparison space used in benchmarking

Benchmarking usually strips away projections

`analysis/benchmarks/_utils.py` builds a comparison pipeline that

- uses the evaluation base transform
- removes all projections
- pins feature names to the training pipeline input features

This gives a common projected-free space for comparing predictions to observations

## Library size handling

Library size handling is explicit throughout the codebase

This is one of the most important implementation details

### Where library size comes from

For transformed data the pipeline always returns

- transformed matrix
- `library_size`

For `normalized_log1p` this `library_size` is the sum of raw counts over the full original cell not just over HVGs

This is intentional

There is even a warning path in `_check_library_size_vs_subset` that detects the bad case where a caller accidentally uses HVG-only sums as the library size

### AE batches

The AE dataloader exposes four related tensors

- `x_raw`
  raw counts for the input feature set
- `x_log_norm`
  `log1p(x_raw / input_library_size * target_sum)` when training in normalized space
- `x_input`
  actual model input
  this may equal `x_log_norm` or another transformed representation
- `lib_size`
  library size used for the count model
- `input_lib_size`
  library size used to reconstruct the input space

In the current datamodule setup both `lib_size` and `input_lib_size` are passed in as the full raw library size

The separation exists so the code can distinguish

- the library size used to parameterize the count model
- the library size used when reconstructing normalized inputs

### Prediction batches

Prediction batches carry

- `x_ctrl`
  matched control cells already transformed into model space
- `cond_batch`
  requested perturbation and context
- `control_library_size`
  raw library size of the matched control cells

That `control_library_size` is what later gets used to decode model outputs back into raw counts

So the default generative interpretation is

- start from a real control cell
- transport it in model space
- decode the predicted perturbed state using that control cell's depth

### Decoding across spaces

`BasePerturbationDataModule.decode_predictions` handles export

If training and evaluation pipelines differ it does

1. inverse the training-space prediction back to raw counts using the provided library size
2. re-transform those raw counts into the evaluation pipeline

This is the key mechanism that keeps library-size semantics consistent across spaces

## Autoencoder architecture

The AE implementation is split across

- `src/flatcfm/modelcore/models/autoencoder.py`
- `src/flatcfm/models/autoencoder.py`

There are two AE families

- `StandardAutoEncoder`
- `NegativeBinomialAutoEncoder`

The default is the negative binomial family

### Shared encoder MLP design

Both AE families use `_build_mlp`

The building block is

- `Linear`
- `LayerNorm`
- `SiLU`

Additional hidden blocks add

- `Linear`
- `LayerNorm`
- `SiLU`
- `Dropout`

Then there is a final output linear layer

Default real-data AE config

- latent dim: `128`
- hidden dim: `256`
- layers: `3`
- dropout: `0.1`

### Standard AE

The standard AE is a plain deterministic encoder-decoder

- encoder: MLP from genes to latent
- decoder: MLP from latent back to genes

Output behavior depends on the input space

- if input space is `raw_counts`
  decoder output is unconstrained
- otherwise
  decoder output is passed through `softplus`

So the standard AE reconstructs the **input space directly**

It does not model count noise explicitly

### Negative binomial AE

The negative binomial AE is the main AE family

Architecture

- encoder: MLP from input features to latent `z`
- decoder hidden tower: MLP from `z` to hidden state
- mean head: linear layer `dec_log_rate`
- dispersion head:
  - default `shared_gene`
    one learned `log_theta` value per gene
  - optional `per_cell_gene`
    linear head from hidden state to genewise dispersion

Default decoder config

- mean head: `per_cell_gene`
- dispersion head: `shared_gene`

### How the NB decoder uses library size

The decoder computes

- `log_rate = dec_log_rate(hidden)`
- `mu = library_size * exp(log_rate)`
- `theta = exp(log_theta)` or `softplus(dec_log_theta(hidden))`

Important consequence

- `mu` is **not** normalized with a softmax
- each gene gets an independent positive rate scaled by the cell depth
- the expected counts across genes do **not** have to sum to the library size

This is a deliberate implementation choice

The test suite explicitly checks that `mu.sum(-1)` is not forced to equal the input library size

### Reconstruction targets

For NB training the model consumes

- `x_input`
  transformed model input
- `x_raw`
  raw counts on the modeled genes
- `library_size`
  full raw library size

Loss is the negative binomial log likelihood of `x_raw` under `(mu theta)`

So the encoder sees normalized or projected inputs but the decoder is trained against count-space targets

### Reconstructing the input space

`NegativeBinomialAutoEncoder.reconstruct_input` does **not** return raw counts unless the input space itself is raw counts

If the input space is `normalized_log1p`

1. decode to `mu` in count space using `library_size`
2. divide by `input_library_size`
3. scale by `target_sum`
4. apply `log1p`

That is why the code keeps both `lib_size` and `input_lib_size`

### Reconstructing raw counts

`reconstruct_counts`

- returns `mu` if `sample=False`
- samples from `NegativeBinomial(total_count=theta probs=mu/(mu+theta))` if `sample=True`

This becomes important when the AE is used as an `ae_latent` projection for downstream FM models

### AE loss terms

The AE wrapper uses a `LossComposer` with three terms

- `recon`
- `distance`
- `pullback`

#### Reconstruction loss

- NB family uses `NBReconLoss`
- standard family uses `MSEReconLoss`

#### Distance preservation loss

`DistancePreservationLoss` compares

- `torch.pdist(z)`
- the provided reference pairwise distances for the same batch

using mean squared error

This is how the code adds geometry supervision to the latent space

#### Pullback isotropy loss

`PullbackIsotropyLoss` tries to make the decoder-induced pullback metric look like `alpha * I`

It does this by

- decoding the current latent point
- computing Fisher-information weights for each gene under the NB decoder
- using JVPs to get Jacobian columns without materializing the full Jacobian
- assembling `G = J^T diag(w) J`
- penalizing `||G - alpha I||^2`

This is a flatness regularizer on the latent geometry

It is present for both AE families because both models include a learned scalar `alpha`

### Geometry targets for AE distance matching

AE geometry is configured separately from the model

Supported modes

- `none`
- `ambient_euclidean`
- `phate_potential`

How those work

- `none`
  no geometry supervision
- `ambient_euclidean`
  use the current training-space representation itself as the geometry embedding
  the dataloader computes within-batch Euclidean distances from those embeddings
- `phate_potential`
  compute PHATE diffusion potential embeddings from the training-space representation
  cache them
  then use within-batch Euclidean distances in diffusion-potential space

So "distance matching" in this repo really means

- provide each AE batch with a reference geometry
- compute pairwise distances inside that geometry
- force latent Euclidean distances to match them

For the common real-data PHATE setting

- normalize and select HVGs first
- compute PHATE diffusion potentials on that normalized space
- train the latent distances to preserve those PHATE distances

### AE schedules

The AE training loop supports single-phase and two-phase schedules

#### Single phase

The model just uses the current loss weights from `loss/ae.yaml` or the experiment override

Examples

- pure reconstruction
- reconstruction plus distance preservation

#### Two phase

The callback in `src/flatcfm/modelcore/callbacks.py` changes both

- which losses are active
- which submodules are trainable

The default two-phase config is

Phase 1

- epochs: `50`
- loss weights: `distance=1 recon=0 pullback=0`
- decoder frozen
- encoder trainable

Phase 2

- epochs: `50`
- loss weights: `recon=1 distance=0 pullback=0`
- encoder frozen
- decoder trainable

Conceptually this means

- first learn a geometry-respecting latent code
- then learn a decoder on top of that fixed code

### AE artifact export and reuse

After AE training the datamodule exports three artifacts

- a checkpoint copy
- a pickled `AELatentProjection`
- a metadata json

The exported projection stores

- the trained AE model object
- input feature names
- latent dimension
- artifact tag

Later a space config can include

- `projections: [{kind: ae_latent artifact_tag: ...}]`

and the pipeline will load that projection as just another transform stage

That is how the repo trains FM models in AE latent space

## Flow matching model

The main predictive model lives in

- `src/flatcfm/modelcore/models/flow_matching.py`
- `src/flatcfm/models/flow.py`

### Architecture

`CondFlow` is a conditional velocity network

Inputs

- current state `x_t`
- scalar time `t`
- condition embedding

Time encoding

- Gaussian Fourier embedding of size `64`

Network

- concatenate `x_t`
- concatenate time embedding
- concatenate condition embedding
- one input linear layer
- a stack of residual linear blocks with `ELU`
- final linear output layer

Output

- predicted velocity vector in the same dimension as the model space

### Training objective

Each batch provides

- a pool of control cells `x_0`
- target perturbed cells `x_1`
- requested condition tensors

The training step does

1. optionally pair controls to perturbed cells with Sinkhorn OT
2. otherwise truncate the random control pool to batch size
3. sample `t ~ Uniform(0 1)`
4. build linear interpolation `x_t = (1-t) x_0 + t x_1`
5. target velocity `v = x_1 - x_0`
6. predict `v_hat = f_theta(x_t t cond)`
7. optimize MSE between `v_hat` and `v`

Optional training details

- `use_ot_coupling`
  use Sinkhorn pairing on the fly
- `ot_pool_multiplier`
  sample more control candidates than targets for OT pairing
- `flow_noise`
  add Gaussian noise to `x_t`

### Prediction

At prediction time the model starts from matched control cells and numerically integrates the ODE

- wrapper class `CondFlowODE`
- `torchdiffeq.odeint`
- default solver `rk4`
- fixed 50-step time grid from `0` to the requested end time

So FM training uses the flow-matching regression objective but inference is done by integrating the learned vector field

## Neural ODE model

The neural ODE model reuses `CondFlow` but changes the training objective

It lives in `src/flatcfm/modelcore/models/neural_ode.py`

Training

- integrate control cells forward with the current vector field
- compare the terminal distribution to perturbed targets
- losses:
  - OT loss
  - density loss
  - energy loss

This is more distributional than pointwise FM

Prediction still uses ODE integration from control cells

## Mean-flow model

The mean-flow implementation is in

- `src/flatcfm/models/mean_flow.py`
- `src/flatcfm/modelcore/models/mean_flow.py`

Architecture

- similar residual MLP style to `CondFlow`
- separate Gaussian Fourier embeddings for `r` and `t`
- predicts an average velocity `u_theta(z_t r t cond)`

Training

- typically OT-pairs controls and perturbed cells with Hungarian assignment
- samples `(r t)` time pairs
- computes a JVP-based identity loss

Prediction

- one-step map `x_hat_1 = x_0 + u_theta(x_0 0 1 cond)`

So this model is an alternative to the FM ODE solve

## Baselines

Baselines live in `src/flatcfm/modelcore/models/baselines.py`

They all operate in the current training space

### Statistical baselines

These do not really train with gradients

They compute statistics once in `on_train_start`

#### No effect

- prediction: `x_control`

#### Additive

- compute one global delta
- `delta = mean(perturbed) - mean(control)`
- prediction: `x_control + delta`

#### Perturb mean

- group perturbed training cells by perturbation and covariates
- store each group mean
- predict the stored mean for that key
- fallback to global perturbed mean

#### Context mean

- group by context only
  not perturbation
- store mean of controls plus perturbed cells within that context
- fallback to global mean

### Learned baselines

#### Linear baseline

- use the neural condition encoder
- project condition embedding directly to an effect vector
- prediction: `x_control + W * cond_embedding`

#### Decoder-only baseline

- build explicit one-hot vectors for perturbation and covariates
- concatenate one-hot condition with `x_control`
- pass through an MLP decoder
- output predicted perturbed state directly

## Dataloaders and training batches

### FM and related tasks

The `CondFMDataset` stores a transformed `AnnData` and splits it into

- control cells
- perturbed cells

The training loader can use `ConditionFirstBatchSampler`

That sampler

- picks a perturbation condition first
- then samples cells from that condition

This increases condition coherence within a batch

The collate function then adds a random pool of control cells

### AE task

The AE task uses `AEBatchDataset`

It batches precomputed matrices instead of single cells

It can carry either

- a full pairwise distance matrix
- or a geometry embedding matrix

In the current datamodule implementations the geometry is provided as embeddings and the dataset computes `torch.pdist` inside each batch

## Prediction export path

Prediction export is handled by

- `src/flatcfm/modelcore/predict.py`
- `src/flatcfm/analysis/prediction_export.py`

The predict command

1. loads the original run config and metadata
2. reconstructs the datamodule and model
3. runs `trainer.predict`
4. asks the datamodule to export predictions into the evaluation space
5. writes `predictions.h5ad`
6. writes prediction request and metadata files
7. optionally converts to PerturbBench-compatible `AnnData`

Important export behavior

- predictions keep `_target_obs_name`
- predictions keep `_control_obs_name`
- predictions record `_prediction_space`
- metadata records whether inverse export was used

## Benchmark metric spaces

Benchmarking is more than just evaluating whatever matrix happened to be predicted

The benchmark stack can build several metric spaces

Supported modes

- `comparison`
  base transform only
  no projections
- `train_base`
  refit the train base transform on the requested fit split
- `train_pca`
  refit train base plus PCA on the requested fit split

This allows the same model outputs to be scored in different analysis spaces

## Metric aggregation modes

Before a metric is computed each group can be represented in different ways

Supported aggregations from `analysis/benchmarks/aggregation.py`

- `none`
  use the full cell distribution
- `average`
  group mean profile
- `var`
  group variance profile
- `scaled`
  z-score against controls then average
- `logfc`
  mean group profile minus mean control profile
- `pca`
  project each cell into PCA space
- `pca_average`
  mean PCA embedding

This is important because some metrics are cell-distribution metrics and some are profile metrics

## Metrics

The metric registry is in `src/flatcfm/analysis/benchmarks/metrics.py`

### Cell distribution metrics

- `mean_gene_w1`
  average 1D Wasserstein distance gene by gene
- `w2_squared`
  exact Wasserstein-2 squared between empirical distributions using POT `emd2`
- `energy_distance`
  energy distance between the two cell sets
- `mmd`
  Gaussian-kernel maximum mean discrepancy

### Profile metrics

- `pearson`
- `cosine`
- `mse`
- `rmse`
- `mae`
- `r2_score`
- `cosine_log_fc`

`cosine_log_fc` expects control data because it compares log fold-change style vectors

### Differential-expression metrics

- `top_k_recall`
- `deg_jaccard`
- `deg_overlap_at_k`

The DE workflow is

1. compare true perturbed cells against controls with Welch t tests
2. apply Benjamini-Hochberg FDR correction
3. rank significant genes by absolute log fold change
4. compare predicted and true DEG sets at `k`

### Metric reduction

Per-group metrics can be reduced to model-level summaries with

- `unweighted_mean`
- `cell_weighted_mean`

## How evaluation groups are formed

By default `Evaluation` groups by

- all sample covariates
- perturbation key

For SciPlex this means

- `cell_type`
- `product_dose`

Benchmark configs can override that and often do

The default benchmark metric config groups by

- `product_dose`

## Benchmark suite architecture

The run-based benchmark suite lives in `src/flatcfm/analysis/benchmarking.py`

It works by

1. loading an anchor run
2. building one or more run-backed predictors
3. generating predictions for a shared prediction request
4. transforming those predictions into one or more metric spaces
5. evaluating them groupwise
6. reducing them into model-level summaries

The implemented predictor kinds are

- `run_fm`
- `run_ode`
- `run_baseline`

So the benchmarking layer compares saved run directories rather than raw model classes

## Default experiment families in the repo

The experiment configs under `src/flatcfm/configs/experiment` cover several model-space combinations

Real-data families include

- log1p FM
- raw-count FM
- PCA-space FM
- AE-latent FM
- AE-latent mean-flow
- log1p neural ODE
- log1p autoencoder with reconstruction only
- log1p autoencoder with PHATE distance matching
- statistical and learned baselines in log1p space
- linear baseline in AE latent space

Toy families include

- identity-space FM
- lifted-space FM
- nonlinear RFF lift experiments
- AE-latent toy experiments

## Important implementation notes

### The repo is space-first

The models are deliberately agnostic to whether they are operating on

- log1p HVGs
- raw counts
- PCA coordinates
- orthogonal lifts
- AE latent codes

That choice is pushed into the transform pipeline

### The AE is the bridge between count modeling and flat latent spaces

When the space uses `ae_latent`

- upstream normalization and feature selection still happen first
- the AE learns a count decoder back to raw space
- downstream FM models operate purely in latent coordinates

This is the central mechanism behind the "flat latent space" idea in this codebase

### Library size is treated as external conditioning for decoding

The latent-space predictive models do not predict library size

Instead they reuse the matched control cell library size when decoding predictions back to raw counts

### Raw-count exports are generally real-valued

Unless `sample_decode=true` for an NB AE export

- inverse transforms usually return expected counts
- those counts are not integer-rounded

That is fine for benchmarking but worth keeping in mind

### The default evaluation space may differ from the training space even when their names match

Because the evaluation pipeline is typically refit on `full_dataset`

This is why the repo has so much explicit roundtripping logic

## If you want one mental model for the whole repo

The cleanest way to think about FlatCFM is

- the `data` layer defines how raw cells become a model-space representation and how conditions become tensors
- the `autoencoder` is an optional learned projection that ties a flat latent space back to raw-count semantics
- the `flow` or `ODE` layers learn a conditional transport from control cells to perturbed cells in that chosen space
- the `analysis` layer decodes predictions back into a common comparison space and scores them with both distributional and profile-level metrics
