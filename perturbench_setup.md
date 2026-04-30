Based on the methodology described in the *PerturBench* paper, **Table 2** presents the results of a **covariate transfer** experiment designed to test out-of-distribution generalization.

Here is the detailed breakdown of the experimental setup that generated the results in Table 2:

### 1. Dataset Selection & Preprocessing

The experiment uses the **Srivatsan20** dataset (same as Sciplex), which maps single-cell transcriptomic responses to **188 chemical perturbations (drugs)** applied across **3 distinct cell lines** (K562, A549, and MCF-7).

* While the original dataset contains responses across four different dosages, the researchers **subsetted the data to only include the highest dose** of each drug. This was done because most of the generative models being benchmarked do not have built-in capacity to model dose-response curves.

From PeturBench paper: "To ensure we are capturing the most biologically relevant features, we subset the expression vectors to highly variable or differentially expressed genes. Specifically, we keep the top 4000 variable genes using the scanpy pp.highly_variable_genes method with flavor=’seurat_v3’. We also keep the top 25 top differentially expressed genes for every perturbation in every unique set of covariates, using scanpy’s tl.rank_genes_groups method with default parameters. For datasets with genetic perturbations, we also ensure that the perturbed gene is included in the feature set as well."

### 2. Task Definition (Covariate Transfer)

The goal of the task is to evaluate whether a model can predict the effect of a drug on a specific cell line when that specific drug-cell combination has never been seen before. The model must "transfer" its understanding of the drug's effect from observed cell lines (the covariates) to the unseen one.

### 3. Data Splitting Strategy

To create this out-of-sample prediction task, the authors used a highly specific splitting mechanism:

* For *each* of the three cell lines, **30% of the perturbations are held out** and allocated strictly for validation and testing.
* The split is engineered to ensure that **any drug held out in one cell line is still observed in the training data of the other two cell lines.** * *Example:* If Drug X is held out for the K562 cell line, the model is still trained on how Drug X affects the A549 and MCF-7 cell lines. At test time, the model is asked to predict what Drug X would do to K562 cells.

### 4. Evaluation Protocol

* **Metrics:** The models' predicted single-cell transcriptomes are compared against the ground-truth held-out cells using a mix of fit metrics (Cosine similarity, MMD in PCA space) and biological rank metrics (LogFC Cosine, LogFC rank, and Differentially Expressed Gene [DEG] recall).
* **Stability Testing:** To account for the stochastic nature of neural network training, the entire setup was run across **5 different random seeds**. The numbers in Table 2 reflect the **mean performance ± one standard deviation**.

### 5. Models Benchmarked

The setup compares a wide array of architectures to see how complexity impacts generalization:

* **Complex/Published Models:** CPA*, SAMS-VAE*, and BioLord*.
* **Ablations/Pre-trained:** CPA without adversarial training (`noAdv`), and models utilizing scGPT foundation embeddings.
* **Simple Baselines:** Latent Additive (LA), simple decoders, and a strictly linear model.



### Latent Distance Error

| Model | Cosine LogFC | Cosine LogFC rank | MMD PCA | DEG recall |
|---|---|---|---|---|
| CPA* | 0.38 ± 6×10⁻³ | 0.15 ± 1×10⁻² | 0.53 ± 4×10⁻³ | 0.007 ± 2×10⁻³ |
| CPA* (noAdv) | 0.40 ± 5×10⁻³ | **0.09 ± 4×10⁻³** | **0.49 ± 1×10⁻²** | 0.004 ± 2×10⁻³ |
| CPA* (scGPT) | 0.39 ± 9×10⁻³ | 0.13 ± 2×10⁻² | - | - |
| SAMS-VAE* | 0.44 ± 1×10⁻³ | 0.17 ± 1×10⁻² | 0.69 ± 1×10⁻² | 0.000 ± 1×10⁻⁴ |
| SAMS-VAE* (S) | **0.53 ± 1×10⁻²** | 0.12 ± 2×10⁻² | 0.79 ± 1×10⁻² | 0.000 ± 0 |
| Biolord* | 0.18 ± 1×10⁻¹ | 0.37 ± 2×10⁻² | 4.3 ± 4×10⁰ | 0.000 ± 1×10⁻⁴ |
| LA | 0.45 ± 2×10⁻³ | 0.13 ± 4×10⁻³ | 2.0 ± 2×10⁻¹ | 0.000 ± 0 |
| LA (scGPT) | 0.50 ± 4×10⁻³ | 0.13 ± 7×10⁻³ | - | - |
| Decoder | 0.35 ± 5×10⁻³ | 0.16 ± 1×10⁻² | 1.9 ± 5×10⁻³ | - |
| Decoder (Cov) | 0.30 ± 1×10⁻² | 0.47 ± 9×10⁻³ | - | - |
| Linear | 0.16 ± 1×10⁻² | 0.28 ± 5×10⁻³ | 0.76 ± 9×10⁻⁴ | 0.004 ± 3×10⁻⁴ |