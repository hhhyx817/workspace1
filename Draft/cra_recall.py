import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

# -----------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.1  # Desired risk level


# -----------------------------------------------------------------------------------
# Step1: 求 s(Xi,j): the weight of the j-th pixel of the i-th image 
# -----------------------------------------------------------------------------------
def compute_cra_score(test_prob_batch: torch.Tensor) -> torch.Tensor:  # 对于所在图片计算的, 代表在所在图片中的地位
    """
    test_prob_batch: probability of forgery
    target class: null(not forged)
    Computes CRA conformity scores according to Equation (4) from CRA paper:
    s(Xi, j) = cumulative_sum(ordered probabilities <= p̂j(Xi)) / total_probability
    """
    B, H, W = test_prob_batch.shape
    scores = torch.zeros_like(test_prob_batch) 
    target_null_prob_batch = 1 - test_prob_batch  

    for b in range(B):
        null_probs = target_null_prob_batch[b].view(-1)                # target prob: [0.1, 0.2, 0.8, 0.7, 0.3]
        sorted_probs, sorted_idx = torch.sort(null_probs)              # sorted: [0.1, 0.2, 0.3, 0.7, 0.8]
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)           # cumulative: [0.1, 0.3, 0.6, 1.3, 2.1]
        total_prob = cumulative_probs[-1]                              # total: 2.1

        # Inverse mapping to original order
        ranks = torch.argsort(sorted_idx)
        cra_scores = cumulative_probs[ranks] / total_prob
        scores[b] = cra_scores.view(H, W)

    # score increases as null probability increases, 所以之后选择 score 较小的划分为 forged
    return scores # shape: (B, H, W)


# -----------------------------------------------------------------------------------
# Step 2: Calibrate Threshold (Weighted Quantile)
# -----------------------------------------------------------------------------------
'''注意一下: 这里的 τ 是一个全局的阈值, 不是每张图像都有自己的 τ_i, 而是所有图像共享同一个 τ
using the calibration set(here has nothing to do with D_test'''
def calibrate_threshold_global(cal_loader: DataLoader, alpha: float = ALPHA) -> float:
    n = len(cal_loader.dataset)  # 图像数量
    
    # 2.1) 分别收集每张图的 (target!) null-class pixel 的 CRA scores
    image_scores = []
    for batch in cal_loader:
        scores = compute_cra_score(batch["mask_binary"].to(device)).view(batch["mask"].shape[0], -1)
        masks  = batch["mask"].to(device).view(batch["mask"].shape[0], -1)
        
        for si, mi in zip(scores, masks): # null-class
            null_scores_i = si[mi == 0]
            image_scores.append(null_scores_i) # image_scores[i]:  null_scores_i，长度 |Y_i|
            '''image_scores = [tensor([0.10, 0.20]),  # 第 1 张图
                        tensor([0.30, 0.60, 0.70, 0.90]),        # 第 2 张图
                        tensor([0.05, 0.15, 0.25, 0.40, 0.55, 0.95])  # 第 3 张图]'''

    # 2.2） 枚举并排列 τ：取所有 image_scores 里独特的 s 值, 依次测 R(τ)，第一个 ≤ α 就是 τ′
    all_scores = torch.unique(torch.cat(image_scores))
    all_scores, _ = torch.sort(all_scores)       
    
    # 2.3） 1/n * 第 i 张图像中 's_i < τ'(被误选)的比例, 取B: float = 1.0
    def R(tau: float) -> float:
        upper_bound = 1.0
        total_frac = sum((si < tau).float().mean().item() for si in image_scores)
        return total_frac / (n + 1) + upper_bound / (n + 1)

    # 2.4） 二分查找第一个(即最小的)满足 R(τ) ≤ α 的 τ
    lo = 0 # index of first element
    hi = len(all_scores) - 1
    best = all_scores[hi].item()
    while lo <= hi:
        mid = (lo+hi)//2
        tau_mid = all_scores[mid].item()
        if R(tau_mid) <= alpha:
            best = tau_mid
            hi = mid-1
        else:
            lo = mid+1
    best_tau = best
    return best_tau
   
   
# -----------------------------------------------------------------------------------
# Step 3: Find the threshold α_i for each image_i using τ
# -----------------------------------------------------------------------------------
def compute_adaptive_alpha(test_prob_batch: torch.Tensor,
                        cra_scores: torch.Tensor,
                        tau: float) -> torch.Tensor:
    """
    Args:
      test_prob_batch: Tensor of shape (B, H, W), forgery probabilities p̂_j(X_i).
      cra_scores:       Tensor of shape (B, H, W), the s(X_i,j) scores.
      tau:              scalar threshold τ'.
    Returns:
      alpha_primes: Tensor of shape (B,), where
      alpha_primes[i] = 1 - (sum_{j: s_ij >= tau} p̂_j) / (sum_j p̂_j).
    """
    B, H, W = test_prob_batch.shape
    alpha_primes = torch.zeros(B, device=test_prob_batch.device, dtype=test_prob_batch.dtype)
    target_null_prob_batch = 1 - test_prob_batch
    
    for i in range(B):
        
        p = target_null_prob_batch[i].view(-1)    # forgery probabilities
        s = cra_scores[i].view(-1)                # CRA scores
        # test_prob_batch[i]: 在 batch 维上取第 i 张图, 得到 (H, W) 的 Tensor; H*W 对应公式中对像素 

        # numerator: sum of p_j over those pixels *not* selected as null (i.e. s >= tau)
        selected_mass = p[s >= tau].sum()
        total_mass    = p.sum().clamp(min=1e-12)

        alpha_primes[i] = 1.0 - selected_mass / total_mass
    return alpha_primes


# -----------------------------------------------------------------------------------
# Step 4: Prediction Set Generation
# -----------------------------------------------------------------------------------
def select_prediction_set_batch(test_prob_batch: torch.Tensor,
                   alpha_primes: torch.Tensor) -> torch.Tensor:
    """
    求出了 batch 中每张图像的 null-class 预测集 Ĉ(X_i, α′_i)
    注意了: 按照 CRA paper, 预测集是指 (目标类) null-class 的像素点, 目标是控制 recall rate of null-class pixels
    Args:
      test_prob_batch: Tensor (B, H, W) of null-class probabilities p̂_j(X_i).
      alpha_primes:    Tensor (B,) per-image α′_i.
    Returns:
      selection_masks: BoolTensor (B, H, W), True for j ∈ Ĉ(X_i,α′_i).
    """
    B, H, W = test_prob_batch.shape
    selection_masks = torch.zeros((B, H * W), dtype=torch.bool, 
                                  device=test_prob_batch.device)
    target_null_prob_batch = 1 - test_prob_batch   # (B, H, W)

    for i in range(B):
        p = target_null_prob_batch[i].view(-1)     # (N_i,) null-class probs
        alpha_i = alpha_primes[i].item()           # scalar

        vals, idx = torch.sort(p, descending=True) # in descending order
        total_mass    = p.sum()                    # how much mass we need to collect
        required_mass = (1 - alpha_i) * total_mass 

        # Sort and Compute cumulative mass until it reaches (1−𝛼𝑖′)×total mass
        cumsum = torch.cumsum(vals, dim=0)
        k = torch.searchsorted(cumsum, required_mass, right=False).item() # smallest k s.t. cumsum[k] ≥ required_mass        
        selection_masks[i, idx[:k+1]] = True # "descending" so select the left(top) k+1 pixels

    # reshape back to (B, H, W)
    return selection_masks.view(B, H, W)


# -----------------------------------------------------------------------------------
# Step 5: CRA pipeline
# -----------------------------------------------------------------------------------
def cra_on_test(cal_loader: DataLoader,
                test_loader: DataLoader,
                alpha: float = ALPHA) -> torch.LongTensor:
    """
    Run full CRA on the entire test set.
    Args:
      cal_loader:  DataLoader for calibration set
      test_loader: DataLoader for test set, each batch is dict with 'mask_binary' of shape (B,H,W)
      alpha:       target FDR level
    Returns:
      all_masks:   LongTensor of shape (N, H, W), where N is total
                   number of test images. 1 = predicted null-class,(target)
                   0 = predicted forged.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5.1) 计算全局阈值 τ′
    tau = calibrate_threshold_global(cal_loader, alpha)

    all_masks = []
    # 5.2) 遍历 test_loader
    for batch in tqdm(test_loader, desc="Running CRA on test set"):
        # a) 取出模型概率并移动到 device
        prob_batch = batch["mask_binary"].to(device)  # shape (B,H,W)

        # b) 计算 CRA scores
        cra_scores = compute_cra_score(prob_batch)

        # c) 计算每张图的 α′_i
        alpha_primes = compute_adaptive_alpha(prob_batch, cra_scores, tau)

        # d) 生成 null-class 预测集掩码 (BoolTensor)
        sel_mask = select_prediction_set_batch(prob_batch, alpha_primes)
        # 转成 0/1 的 LongTensor
        all_masks.append(sel_mask.long())

    # 5.3) 拼接所有 batch，返回 (N, H, W)
    return torch.cat(all_masks, dim=0)
'''例子: 
all_pred_masks = cra_on_test(cal_loader, test_loader, alpha=0.1)
# all_pred_masks.shape == (N, H, W)
# all_pred_masks[i,j,k] == 1 表示第 i 张图上 (j,k) 像素被预测为 null-class'''
