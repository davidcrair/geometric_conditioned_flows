# Conditioned Optimal Transport Flows in Geometrically Regularized Latent Space for Single Cell Perturbation Modeling

Senior Thesis Project in Computer Science\\
Author: David Crair\\
Advisor: Smita Krishnaswamy

## setup
Use `uv sync` to install dependencies and create a virutal environment.
Run all jupyter notebooks using this virtual environment.

## repo layout
```
├── data/                        # data loading and preprocessing
│   ├── dataset.py               # dataset classes
│   ├── simulations.py           # toy data generation
│   ├── space.py                 # latent space utilities
│   ├── splitters.py             # train/test splitting
│   └── types.py                 # shared type definitions
├── models/                      # model definitions
│   ├── autoencoder.py           # autoencoder architectures
│   ├── baselines.py             # baseline models
│   ├── flow.py                  # flow matching model
│   ├── mean_flow.py             # mean flow model
│   └── vector_fields.py         # vector field networks
├── training/                    # training loops
│   ├── losses.py                # loss functions
│   ├── trainer_ae.py            # autoencoder trainer
│   ├── trainer_flow_matching.py # flow matching trainer
│   ├── trainer_mean_flow.py     # mean flow trainer
│   └── trainer_neural_ode.py    # neural ODE trainer
├── evaluation/                  # evaluation utilities
│   ├── metrics.py               # evaluation metrics
│   └── plotting.py              # visualization helpers
├── artifacts/                   # saved models and data splits
├── train_*_toy.ipynb            # toy experiment notebooks
├── train_*_sciplex.ipynb        # sci-Plex experiment notebooks
├── benchmark_toy.ipynb          # toy benchmark evaluation
├── benchmark_real.ipynb         # sci-Plex benchmark evaluation
└── pyproject.toml               # project dependencies
```

## running the notebooks

### toy experiments
1. `train_ae_toy.ipynb` - train the autoencoder
2. `train_fm_toy.ipynb` - train flow matching model
3. `train_mf_toy.ipynb` - train mean flow model
4. `train_ode_toy.ipynb` - train neural ODE model
5. `benchmark_toy.ipynb` - evaluate all toy models

steps 2-4 are independent and can be run in any order after step 1\
step 5 requires all models from steps 2-4

### sciplex experiments
1. `train_ode_sciplex.ipynb` - subsample data, create train/test splits, train neural ODE baseline
2. `train_nbae_sciplex.ipynb` - train the negative binomial autoencoder

after steps 1-2, the following can be run in any order:
- `train_fm_sciplex.ipynb` - train flow matching in gene space (requires step 1)
- `train_fm_nbae_sciplex.ipynb` - train flow matching in ae latent space (requires steps 1 and 2)
- `train_ode_nbae_sciplex.ipynb` - train neural ode in ae latent space (requires steps 1 and 2)

finally
- `benchmark_real.ipynb` - evaluate all sciplex models (requires all of the above)
