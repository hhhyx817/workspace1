import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # ← 必须在最前面！

import torch
device = torch.device("cuda:0") # 对应可见的第一块

import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.1   # Target FDR level
TAU = 0.5     # Threshold tau for estimating null proportion
EPS = 1e-6    # Numerical stability for division


# -----------------------------------------------------------------------------------------------------------------------
# Step 1: Estimate q_hat 
# -----------------------------------------------------------------------------------------------------------------------
# TODO wight 依据无 calibration 时 predictor 的准确程度来判断
def compute_qhat_local_maxpool(prob_matrix: torch.Tensor,weight_self: float,weight_local: float,
                               kernel_size: int = 5,eps: float = 1e-6) -> torch.Tensor:

    device = prob_matrix.device
    N, H, W = prob_matrix.shape

    # 增加 channel 维度
    prob = prob_matrix.unsqueeze(1)  # (N, 1, H, W)

    # 负最大池化等价于最小池化
    pad = kernel_size // 2
    prob_local_max = F.max_pool2d(
        prob,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )  # (N, 1, H, W)

    prob_local_max = prob_local_max.squeeze(1)

    q_hat = 1.0 - (weight_self * prob_matrix + weight_local * prob_local_max)
    q_hat = torch.clamp(q_hat, min=eps, max=1.0)
    return q_hat

"""
def compute_qhat_local(prob_matrix: torch.Tensor, kernel_size: int) -> torch.Tensor:
    
    device = prob_matrix.device
    N, H, W = prob_matrix.shape

    # 增加 channel 维度以便使用 max_pool2d
    prob = prob_matrix.unsqueeze(1)  # (N, 1, H, W)

    # 计算局部最大
    pad = kernel_size // 2
    prob_local_max = F.max_pool2d(
        prob,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )  # (N, 1, H, W)

    # 恢复形状
    prob_local_max = prob_local_max.squeeze(1)  # (N, H, W)

    # q_hat = 1 - local_max
    q_hat = 1.0 - prob_local_max

    # 截断到 [eps, 1.0]
    eps = 1e-6
    q_hat = torch.clamp(q_hat, min=eps, max=1.0)

    return q_hat
"""

# -----------------------------------------------------------------------------------------------------------------------
# Step 2: p_value
# -----------------------------------------------------------------------------------------------------------------------
def compute_null_pixel_pvalues_gpu(cal_loader, test_prob_batch, device=device):
    """ 
    基于右尾分布的像素级 conformal p-values
    输入：
        cal_loader: 全体校准集 DataLoader, 每个batch是 dict {"mask_binary", "mask", ...}
        test_prob_batch: (B, 256, 256) Tensor  
                    这个变量表示的是测试集像素的 mask_binary 变量
                    这里不用 test_loader 是因为有几个 batch 并不符合要求
                    在 evaluate 函数中使用 test_groundtruth_batch
    输出：
        pval_matrix: (B, 256, 256) """

    # 1. 从校准集loader中提取数据
    X_all = []
    Y_all = []

    for batch in tqdm(cal_loader, desc="Collecting calibration set"):
        X_all.append(batch["mask_binary"].float().to(device, non_blocking=True))   # 模型预测
        Y_all.append(batch["mask"].float().to(device, non_blocking=True))          # 真实标签

    X_cal_tensor = torch.cat(X_all, dim=0)  # (n, 256, 256)
    Y_cal_tensor = torch.cat(Y_all, dim=0)  # (n, 256, 256)

    # nonconformity scores, 只计算 null 类型
    null_mask = (Y_cal_tensor == 0)                        # (n, 256, 256) with 1/0
    V_cal = X_cal_tensor[null_mask]                        # 自动展平
    V_sorted, _ = torch.sort(V_cal)

    # 2. 测试集的 V 就是 mask_binary 本身(因为和null类的差距即 prob - 0), flatten
    V_test_flat = test_prob_batch.float().to(device, non_blocking=True).view(-1)  # (B * 256 * 256,)

    # 3. 正确的右尾 p-value 计算（大V → 小p）
    n_cal = V_sorted.numel()

    idx_gt = torch.searchsorted(V_sorted, V_test_flat, right=True)   # index of first V_sorted > V_test
    idx_eq = torch.searchsorted(V_sorted, V_test_flat, right=False)  # index of first V_sorted ≥ V_test

    n_greater = n_cal - idx_gt
    n_equal = idx_gt - idx_eq

    U = torch.rand_like(n_greater, dtype=torch.float32, device=V_sorted.device)
    pvals_flat = (n_greater.float() + U * (n_equal.float() + 1)) / (n_cal + 1)

    B = test_prob_batch.shape[0]
    pval_matrix = pvals_flat.view(B, 256, 256)

    return pval_matrix


# -----------------------------------------------------------------------------------------------------------------------
# Step 3: SABHA selection - GPU版
# -----------------------------------------------------------------------------------------------------------------------
# pvals 即 N*256*256 matrix of pixel-wise p-value, 这个在 distribution_test 中求过了
def sabha_selection_gpu(pvals: torch.Tensor, q_hat: torch.Tensor, alpha: float = ALPHA):
    """
    输入:
        pvals: (B,256,256) tensor
        q_hat: (B,256,256) tensor
    输出:
        selected_mask: (B,256,256) tensor, dtype=torch.bool (或 0/1 int型)
    """
    original_shape = pvals.shape  # 保存原shape (N,256,256)

    # 1. flatten到一维
    pvals = pvals.view(-1)
    q_hat = q_hat.view(-1)
    
    # 2. 调整p-value
    pvals_adj = pvals * q_hat  #注意了 p*q 之后两边同除以 q_hat
    pvals_adj = torch.clamp(pvals_adj, max=1.0)
    pvals_adj_sorted, sort_idx = torch.sort(pvals_adj)  # 从小到大升序排列

    n = pvals_adj.numel()
    print(n)
    thresholds = (torch.arange(1, n + 1, device=pvals.device).float() / n) * alpha

    # 3. 找满足的索引
    below = (pvals_adj_sorted <= thresholds)
    if below.any():
        k_max = torch.nonzero(below, as_tuple=False).max()
        selected_flat_idx = sort_idx[:k_max + 1]
    else:
        selected_flat_idx = torch.tensor([], dtype=torch.long, device=pvals.device)

    # 4. 构建 (N*256*256,) 的 mask
    selected_mask_flat = torch.zeros(n, dtype=torch.bool, device=pvals.device)
    if selected_flat_idx.numel() > 0:
        selected_mask_flat[selected_flat_idx] = True
        
    selected_mask = selected_mask_flat.view(original_shape)

    return selected_mask


# -----------------------------------------------------------------------------------------------------------------------
# Step4: 整体pipeline封装
# -----------------------------------------------------------------------------------------------------------------------
# mask_binary 指的是 概率矩阵
def sabha_pipeline_gpu(cal_loader: DataLoader,
                       test_prob_batch: torch.Tensor,
                       alpha: float = ALPHA, device = device ):
    
    pval_matrix = compute_null_pixel_pvalues_gpu(cal_loader, test_prob_batch, device=device).to(device)
    q_hat = compute_qhat_local_maxpool(test_prob_batch,0,1,5,1e-6).to(device)
    selected_mask = sabha_selection_gpu(pval_matrix, q_hat, alpha)
    
    # print("  • mean(q_hat):", q_hat.mean().item())
    # print("  • mean(pvals):", pval_matrix.mean().item())

    return selected_mask

