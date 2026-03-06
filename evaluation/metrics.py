import torch
import ot
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control


def compute_energy_distance(x, y, max_samples=2000):
    """energy distance with subsampling to avoid OOM on large sets"""
    if x.size(0) > max_samples:
        x = x[torch.randperm(x.size(0), device=x.device)[:max_samples]]
    if y.size(0) > max_samples:
        y = y[torch.randperm(y.size(0), device=y.device)[:max_samples]]

    dist_xy = torch.cdist(x, y, p=2)
    dist_xx = torch.cdist(x, x, p=2)
    dist_yy = torch.cdist(y, y, p=2)

    energy_dist = 2 * dist_xy.mean() - dist_xx.mean() - dist_yy.mean()
    return energy_dist.item()


def compute_mmd(x, y, bandwidth=None, max_samples=2000):
    if x.size(0) > max_samples:
        x = x[torch.randperm(x.size(0), device=x.device)[:max_samples]]
    if y.size(0) > max_samples:
        y = y[torch.randperm(y.size(0), device=y.device)[:max_samples]]
    if bandwidth is None:
        # median heuristic
        all_dists = torch.cdist(x, y, p=2).reshape(-1)
        bandwidth = all_dists.median().item()
    bandwidth = max(bandwidth, 1e-8)
    gamma = 1.0 / (2 * bandwidth**2)
    k_xx = torch.exp(-gamma * torch.cdist(x, x, p=2).pow(2)).mean()
    k_yy = torch.exp(-gamma * torch.cdist(y, y, p=2).pow(2)).mean()
    k_xy = torch.exp(-gamma * torch.cdist(x, y, p=2).pow(2)).mean()
    return (k_xx + k_yy - 2 * k_xy).item()


def compute_mse(x, y):
    """mse between per-gene mean expression of predicted and ground-truth cells"""
    return torch.mean((x.mean(0) - y.mean(0)) ** 2).item()


def compute_wasserstein_distance(x, y, max_samples=2000):
    """2nd order wass distance with sampling"""
    if x.size(0) > max_samples:
        x = x[torch.randperm(x.size(0), device=x.device)[:max_samples]]
    if y.size(0) > max_samples:
        y = y[torch.randperm(y.size(0), device=y.device)[:max_samples]]

    # cost is squared euclidean distance
    M = torch.cdist(x, y, p=2).pow(2)

    # solve earth mover's distance (EMD)
    # ot.emd2 expects numpy/float64 for stability in some versions
    M_np = M.detach().cpu().numpy().astype(np.float64)
    a = np.ones(x.size(0)) / x.size(0)
    b = np.ones(y.size(0)) / y.size(0)
    w2_sq = ot.emd2(a, b, M_np)
    return np.sqrt(w2_sq)


def precompute_true_deg_info(
    x_true: np.ndarray,
    x_ctrl: np.ndarray,
    fdr_alpha: float = 0.05,
    min_cells: int = 5,
) -> dict:
    """precompute deg info for a single perturbation's ground truth

    runs welch t-test (pert vs ctrl) applies bh correction then ranks
    significant genes by absolute log fold change (descending)

    Args:
        x_true: ground truth perturbed expression (n_pert, n_genes)
        x_ctrl: control expression (n_ctrl, n_genes)
        fdr_alpha: bh fdr threshold
        min_cells: minimum cells required per group to run the test

    Returns:
        dict with:
            ``ranked_indices``: gene indices sorted by |lfc| descending (sig only)
            ``n_sig``: total number of significant degs (n)
    """
    if x_true.shape[0] < min_cells or x_ctrl.shape[0] < min_cells:
        return {"ranked_indices": np.array([], dtype=int), "n_sig": 0}

    _, pvals = stats.ttest_ind(x_true, x_ctrl, axis=0, equal_var=False)
    pvals = np.nan_to_num(pvals, nan=1.0)
    adj_pvals = false_discovery_control(pvals, method="bh")
    sig_mask = adj_pvals < fdr_alpha

    lfc = x_true.mean(axis=0) - x_ctrl.mean(axis=0)
    sig_indices = np.where(sig_mask)[0]

    if len(sig_indices) == 0:
        return {"ranked_indices": np.array([], dtype=int), "n_sig": 0}

    order = np.argsort(-np.abs(lfc[sig_indices]))
    ranked_indices = sig_indices[order]
    return {"ranked_indices": ranked_indices, "n_sig": len(ranked_indices)}


def compute_deg_overlap(
    x_pred: np.ndarray,
    x_ctrl: np.ndarray,
    true_deg_info: dict,
    ks: list | None = None,
) -> dict:
    """compute deg overlap between predicted and true deg sets at various k

    for the predicted set genes are ranked by absolute log fold change vs
    control (no fdr filtering) for the true set only fdr-significant genes
    (precomputed in ``true_deg_info``) are used

    the overlap fraction is ``|true_top_k ∩ pred_top_k| / k`` when
    ``k = None`` (the ``DEG@N`` variant) ``k`` is set to the total number
    of significant degs in the true set

    Args:
        x_pred: predicted expression (n_pred, n_genes)
        x_ctrl: control expression (n_ctrl, n_genes)
        true_deg_info: output of :func:`precompute_true_deg_info`
        ks: top-k cutoffs use ``None`` as an entry for k=n defaults to
            ``[50, 100, 200, None]``

    Returns:
        dict mapping label -> overlap fraction e.g.
        ``{"DEG@50": 0.4, "DEG@100": 0.5, "DEG@200": 0.55, "DEG@N": 0.6}``
    """
    if ks is None:
        ks = [50, 100, 200, None]

    true_ranked = true_deg_info["ranked_indices"]
    n_sig = true_deg_info["n_sig"]

    lfc_pred = x_pred.mean(axis=0) - x_ctrl.mean(axis=0)
    pred_ranked_all = np.argsort(-np.abs(lfc_pred))  # all genes, desc |LFC|

    overlaps = {}
    for k in ks:
        actual_k = n_sig if k is None else k
        label = "DEG@N" if k is None else f"DEG@{k}"

        if actual_k == 0 or n_sig == 0:
            overlaps[label] = 0.0
            continue

        true_top_k = set(true_ranked[:actual_k])
        pred_top_k = set(pred_ranked_all[:actual_k])
        overlaps[label] = len(true_top_k & pred_top_k) / actual_k

    return overlaps


def compute_perturbation_discrimination(
    pert_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    """perturbation discrimination score

    measures whether the model's predictions preserve relative differences between perturbations
    for each perturbation i we count how many other perturbations j have a ground-truth profile closer to g_i
    than the model's own prediction p_i using manhattan distance

    Args:
        pert_profiles: dict mapping perturbation label to a tuple of
            (predicted_mean_profile ground_truth_mean_profile) each a
            1-d numpy array of shape (n_genes,)
    """
    labels = list(pert_profiles.keys())
    T = len(labels)
    if T < 2:
        return 0.0

    preds = np.stack([pert_profiles[l][0] for l in labels])  # (T, G)
    gts = np.stack([pert_profiles[l][1] for l in labels])  # (T, G)

    # gt_dists[i, j] = Manhattan distance d(g_i, g_j)
    gt_dists = np.abs(gts[:, None, :] - gts[None, :, :]).sum(axis=2)  # (T, T)

    # Distance from each prediction to its own ground truth
    pred_gt_dists = np.abs(preds - gts).sum(axis=1)  # (T,)

    pdis_per_pert = np.zeros(T)
    for i in range(T):
        others_closer = 0
        for j in range(T):
            if j == i:
                continue
            if gt_dists[j, i] < pred_gt_dists[i]:
                others_closer += 1
        pdis_per_pert[i] = others_closer / (T - 1)

    pdis = pdis_per_pert.mean()
    return float(1.0 - 2.0 * pdis)


def compute_cosine_log_fc(x_pred, x_true, x_ctrl, eps=1e-8):
    """cosine similarity between log fold changes of predicted and true data"""

    # compute means
    mu_pred = x_pred.mean(0)
    mu_true = x_true.mean(0)
    mu_ctrl = x_ctrl.mean(0)

    delta_pred = mu_pred - mu_ctrl
    delta_true = mu_true - mu_ctrl

    if torch.norm(delta_pred) < 1e-9 or torch.norm(delta_true) < 1e-9:
        return 0.0

    cos = torch.nn.functional.cosine_similarity(delta_pred.unsqueeze(0), delta_true.unsqueeze(0))
    return cos.item()
