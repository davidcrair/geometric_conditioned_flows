


**Train flow matching model on toy dataset:**
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=toy/fm_identity`

**Train Neural ODE model on toy dataset:**
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=toy/ode_identity`


**Train autoencoder model without geometric loss:**
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_recon`
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_ae_latent_recon`

**Train autoencoder model with geometric loss:**
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_phate`
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_ae_latent_phate`




HVGs after normalize total and log1p, then PCA:
```bash
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/fm_log1p \
  space=pca \
  evaluation_space=raw_counts \
  experiment_name=sciplex_fm_pca_hvg
```

Train all toy projections for comparision
`./scripts/train_toy_projection_comparison_grid.sh`


1. Train FM in AE latent space (the main model):                                                      
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_ae_latent_recon`

2. Run predictions for FM:                                                                              
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict predict.run_dir=$(ls -d artifacts/runs/sciplex_fm_ae_latent_recon/*/ | tail -1)`

3. (Optional) Train Neural ODE baseline:
`PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ode_log1p`

4. Run predictions for baselines (already trained, just need predictions):
```bash
for baseline in no_effect additive context_mean perturb_mean decoder linear; do
  run_dir=$(ls -d artifacts/runs/sciplex_baseline_${baseline}/*/ | tail -1)
  PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict "predict.run_dir=${run_dir}"
done
```


FM in PCA space with HVG selection:
```bash
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/fm_log1p \
  space=pca \
  experiment_name=sciplex_fm_pca_hvg
```

PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_log1p space=pca  experiment_name=sciplex_fm_pca_hvg



# Rerun stuff, added perturbation covariates.

Train all models                                                                                             
                                                                                                               
# baselines (fast — minutes each)                                                                            
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/baseline_no_effect experiment_name=sciplex_baseline_no_effect trainer.precision=bf16-mixed                                      
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/baseline_additive experiment_name=sciplex_baseline_additive trainer.precision=bf16-mixed                                       
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/baseline_context_mean experiment_name=sciplex_baseline_context_mean trainer.precision=bf16-mixed
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/baseline_perturb_mean experiment_name=sciplex_baseline_perturb_mean trainer.precision=bf16-mixed

# learned baselines (longer — use condition encoder with new dose covariate)
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/baseline_decoder experiment_name=sciplex_baseline_decoder trainer.precision=bf16-mixed
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/baseline_linear experiment_name=sciplex_baseline_linear trainer.precision=bf16-mixed

# main models
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/fm_log1p space=pca experiment_name=sciplex_fm_pca_hvg trainer.precision=bf16-mixed
- [x] PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/ode_log1p

Predict all models

After each training run completes, predict using the latest run dir:

for exp in sciplex_baseline_no_effect sciplex_baseline_additive sciplex_baseline_context_mean sciplex_baseline_perturb_mean sciplex_baseline_decoder sciplex_baseline_linear sciplex_fm_pca_hvg sciplex_ode_log1p; do
  run_dir=$(ls -dt artifacts/runs/$exp/*/ | head -1)
  echo "Predicting $exp -> $run_dir"
  PYTHONPATH=src python -m flatcfm.modelcore.predict predict.run_dir="$run_dir"
done





# PHATE Training
Step 1: Train the PHATE autoencoder

PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_phate

This trains an AE with PHATE distance-preservation geometry (two-phase schedule: 50 epochs
distance-preserving with frozen decoder, 50 epochs reconstruction with frozen encoder). It exports artifacts
tagged sciplex_ae_log1p_phate to:
- artifacts/spaces/sciplex_ae_projection_sciplex_ae_log1p_phate.pkl
- artifacts/models/sciplex_ae_model_sciplex_ae_log1p_phate.ckpt

The config is at src/flatcfm/configs/experiment/sciplex/ae_log1p_phate.yaml.

Step 2: Train FM in the AE latent space

PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_ae_latent_phate

This config (src/flatcfm/configs/experiment/sciplex/fm_ae_latent_phate.yaml) uses:
- Space: ae_latent with artifact_tag: sciplex_ae_log1p_phate — loads the pre-trained AE checkpoint
- Pipeline: raw → log1p(2000 HVGs) → ae_latent — FM operates in the compressed latent dim
- Training: standard flow matching velocity prediction in latent space

Step 3: Generate predictions

PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict predict.run_dir=artifacts/runs/sciplex_fm_ae_latent_phate/20260315_125420

Step 4: Evaluate

PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_ae_latent_phate

Evaluation automatically handles the space mismatch — FM predictions in latent space are decoded back through
  the AE pipeline (latent → inverse AE → base space → comparison space) before computing metrics.






# Train FM in HVG space with log1p normalization
PYTHONPATH=src python -m flatcfm.modelcore.train experiment=sciplex/fm_log1p space=log1p experiment_name=sciplex_fm_hvg trainer.precision=bf16-mixed

PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict predict.run_dir=artifacts/runs/sciplex_fm_hvg/20260315_153557

PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_hvg


# Train FM in PCA space with log1p normalization
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_log1p space=pca experiment_name=sciplex_fm_pca_hvg trainer.precision=bf16-mixed                                               
                                                                                                                
PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_pca_hvg
                                        
PYTHONPATH=src .venv/bin/python scripts/run_eval.py sciplex_fm_pca_hvg






# ODE in HVG space with log1p normalization
.venv/bin/python -m flatcfm.training experiment=sciplex/ode_log1p experiment_name=sciplex_ode_hvg


PYTHONPATH=src .venv/bin/python -m flatcfm.modelcore.predict predict.run_dir=/nfs/roberts/scratch/pi_sk2433/dac227/FlatCFM/artifacts/runs/sciplex_ode_hvg/20260315_195300




.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_baseline_decoder_deg predict.name=train_same_control_rule predict.split=train "+predict.target_filters.obs_equals.cell_type=K562"



# most recent train stuff

Step 1: Train the PHATE AE (on HVG+DEG space — already configured)
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_phate trainer.precision=bf16-mixed

Step 2: Train FM in latent space
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_ae_latent_phate trainer.precision=bf16-mixed

Step 3: Train ODE in latent space
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ode_ae_latent_phate trainer.precision=bf16-mixed

Step 1 must finish first (it exports the AE projection artifact that steps 2 and 3 consume). Steps 2 and 3 are
  independent and can run in parallel.

Then predictions:
.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_deg_ae_latent_phate
.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_ode_deg_ae_latent_phate

.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_deg_ae_latent_phate predict.name=train_same_control_rule predict.split=train "+predict.target_filters.obs_equals.cell_type=K562"

.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_ode_deg_ae_latent_phate predict.name=train_same_control_rule predict.split=train "+predict.target_filters.obs_equals.cell_type=K562"


# Recon only stuff

.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_phate trainer.precision=bf16-mixed



.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_deg_ae_latent_recon predict.name=train_same_control_rule predict.split=train "+predict.target_filters.obs_equals.cell_type=K562"

uv run scripts/run_eval.py sciplex_fm_deg_ae_latent_recon --split train_same_control_rule


**using autoencoder in log1p space with latent_dim=512, no geometric loss, training for 50 epochs with 4 workers**
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_recon trainer.precision=bf16-mixed model.latent_dim=512  experiment_name=sciplex_ae_deg_log1p_recon_d512 task.epochs=50 trainer.num_workers=12




# train FM                                                                                                                                   
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_ae_latent_phate_cosine trainer.precision=bf16-mixed                        
                                                                                                                                               
# predict held-out                                                                                                                           
.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_deg_ae_latent_phate_cosine         

# predict train
.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_deg_ae_latent_phate_cosine predict.name=train_same_control_rule predict.split=train "+predict.target_filters.obs_equals.cell_type=K562"

                                                                                                                                              
# evaluate                                                                                                                                   
uv run scripts/run_eval.py sciplex_fm_deg_ae_latent_phate_cosine

**eval on train same control rule split**
uv run scripts/run_eval.py sciplex_fm_deg_ae_latent_phate_cosine --split train_same_control_rule



# train ODE in PCA latent (50 PCs from DEG+HVG space)
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ode_pca trainer.precision=bf16-mixed

# predict
.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_ode_deg_pca

# evaluate
uv run scripts/run_eval.py sciplex_ode_deg_pca

**eval on train same control rule split**
`.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_ode_deg_pca predict.name=train_same_control_rule predict.split=train "+predict.target_filters.obs_equals.cell_type=K562"`


# FM in PCA latent
# train                                                                                                                                            
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/fm_pca_deg trainer.precision=bf16-mixed                                             
                                                                                                                                                    
# predict held-out                                                                                                                                 
.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_deg_pca                                                           
                                                                                                                                                    
# predict train K562                                                                                                                               
.venv/bin/python -m flatcfm.modelcore.predict predict.experiment_name=sciplex_fm_deg_pca predict.name=train_same_control_rule predict.split=train "+predict.target_filters.obs_equals.cell_type=K562"                                                                                                

# evaluate held-out                                                                                                                                
uv run scripts/run_eval.py sciplex_fm_deg_pca                                                                                                    
                                                                                                                                                    
# evaluate train K562                                                                                                                              
uv run scripts/run_eval.py sciplex_fm_deg_pca --split train_same_control_rule                                                                      
                                                                                    






python -m flatcfm.modelcore.train \
  experiment=sciplex/baseline_decoder \
  hpo=baseline_decoder \
  hydra.sweeper.n_trials=40 \
  hydra.sweeper.n_jobs=4 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  trainer.num_workers=8 \
  trainer.pin_memory=true



# view perturbench results
(.venv) [dac227@a1118u09n01.bouchet FlatCFM]$ `uv run scripts/view_perturbench.py`



Both files are ready. Here's the workflow:

**1. Check status and get training commands:**
```bash
.venv/bin/python scripts/check_dim_sweep.py            # see status table
.venv/bin/python scripts/check_dim_sweep.py --generate  # print commands for missing models
```

**2. Train models** (21 runs: 3 variants x 7 dims). Example:
```bash
.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_mse_d512 model.latent_dim=64 experiment_name=sciplex_ae_deg_mse_d64 space.ae_export_artifact_tag=sciplex_ae_deg_mse_d64
```

**3. View results** — open `view_theoretical_limits.ipynb`. It skips missing models gracefully, so you can run it at any point to see partial results as models finish training.

The notebook produces 4 plots:
- cosine_logfc vs variance explained (all methods, PCA extends further right)
- W2 vs variance explained
- cosine_logfc vs latent dim (direct comparison at 16-1024)
- W2 vs latent dim


.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_mse_d512 model.latent_dim=32 experiment_name=sciplex_ae_deg_mse_d32 space ae_export_artifact_tag=sciplex_ae_deg_mse_d32 trainer.num_workers=12



.venv/bin/python -m flatcfm.modelcore.train experiment=sciplex/ae_log1p_mse_wide model.latent_dim=512 model.hidden_dim=1024 experiment_name=debug_mse_wide_baseline space.ae_export_artifact_tag=debug_mse_wide_baseline trainer.num_workers=12 task.epochs=20  


P=scavenge_gpu
for SP in strict_a549 strict_mcf7; do
  sbatch --partition=$P \
  --export=ALL,EXPERIMENT=sciplex/ae_log1p_mse_d512,SPLITTER=$SP,SKIP_PREDICT=1,TASK_EPOCHS=50 \
  scripts/train_benchmark.sbatch
  done


P=scavenge_gpu

# Phase 1: AE pretraining (3 jobs, wait for these to finish)
for SP in strict_k562 strict_a549 strict_mcf7; do
  sbatch --partition=$P --export=ALL,EXPERIMENT=sciplex/ae_log1p_hybrid_d1e3,SPLITTER=$SP,SKIP_PREDICT=1 \
    scripts/train_benchmark.sbatch
done

# Phase 2: FM + ODE on top of the hybrid AE (6 jobs, after AEs complete)
for SP in strict_k562 strict_a549 strict_mcf7; do
  sbatch --partition=$P --export=ALL,EXPERIMENT=sciplex/fm_ae_latent_hybrid_d1e3,SPLITTER=$SP  scripts/train_benchmark.sbatch
  sbatch --partition=$P --export=ALL,EXPERIMENT=sciplex/ode_ae_latent_hybrid_d1e3,SPLITTER=$SP scripts/train_benchmark.sbatch
done




.venv/bin/python scripts/run_eval.py \
  sciplex_fm_deg_pca_d16 \
  sciplex_fm_deg_pca_d128 \
  sciplex_fm_deg_geom_reg_w0 \
  sciplex_fm_deg_geom_reg_w1



```bash
.venv/bin/python scripts/run_eval.py \
  sciplex_fm_deg_pca_d16 \
  sciplex_fm_deg_pca_d128 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_k562 \
  sciplex_fm_deg_geom_reg_w0 sciplex_fm_deg_geom_reg_w0p001 \
  sciplex_fm_deg_geom_reg_w0p01 sciplex_fm_deg_geom_reg_w0p1 sciplex_fm_deg_geom_reg_w1 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_molformer_k562 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_cell_dropout0p2_k562 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_pert_dropout0p2_k562 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_all_dropout0p2_k562
```


```bash
.venv/bin/python scripts/run_eval.py \
  sciplex_fm_deg_pca_d16 \
  sciplex_fm_deg_pca_d128 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_k562 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_molformer_k562 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_cell_dropout0p2_k562 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_pert_dropout0p2_k562 \
  sciplex_fm_deg_ae_latent_mse_d128_dist_all_dropout0p2_k562
```