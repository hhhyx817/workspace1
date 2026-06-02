import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # ← 必须在最前面！

import torch
device = torch.device("cuda:0") # 对应可见的第一块

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import numpy as np
from typing import Tuple

# -----------------------------------------------------------------------------------
# Config
'''
预测集里的是 forgery(target/positive), prob大 ——> p-value 大 ——> s 大, s(X_i,j) ≥ τ* 则预测为 forgery
目标是: 真正 forgery / 预测为 forgery >= 0.9 ——> 选中了且真的 forgery TP → 相当于正确拒绝；选中了但其实是 null FP → 错误拒绝 

'''
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.1  # Desired risk level, 1 -1 precision <= ALPHA


# -----------------------------------------------------------------------------------
# Step1: 求 s(Xi,j): the weight of the j-th pixel of the i-th image 
# -----------------------------------------------------------------------------------
def compute_cra_score(test_prob_batch: torch.Tensor) -> torch.Tensor:  # 对于所在图片计算的, 代表在所在图片中的地位
    """
    test_prob_batch: probability of forgery
    target/positive class: forged
    Computes CRA conformity scores according to Equation (4) from CRA paper:
    s(Xi, j) = cumulative_sum(ordered probabilities <= p̂j(Xi)) / total_probability
    """
    B, H, W = test_prob_batch.shape
    scores = torch.zeros_like(test_prob_batch) 
    target_prob_batch = test_prob_batch  

    for b in range(B):
        null_probs = target_prob_batch[b].view(-1)                     # target prob: [0.1, 0.2, 0.8, 0.7, 0.3]
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
    
    # 2.1) 分别收集每张图所有像素的 CRA scores 和对应的真实标签
    all_scores = []
    all_labels = []

    for batch in cal_loader:
        scores = compute_cra_score(batch["mask_binary"].to(device)).view(batch["mask"].shape[0], -1)
        masks  = batch["mask"].to(device).view(batch["mask"].shape[0], -1)
        
        all_scores.append(scores)
        all_labels.append(masks)
        
    scores_tensor = torch.cat(all_scores, dim=0).view(-1)
    labels_tensor = torch.cat(all_labels, dim=0).view(-1)

    # 2.2) 枚举所有唯一的 s 值（作为候选的 best τ）
    candidate_taus = torch.unique(scores_tensor)
    candidate_taus, _ = torch.sort(candidate_taus, descending=False)  # 由小到大排序(看之后二分)
    
    # 2.3) 定义 calibration set 中的的误检比例 R(τ)
    def R_precision(tau: float) -> float:
        selected = scores_tensor >= tau                                  # 认为 target/positive 的像素, 这里取 forged 
        false_positive = ((labels_tensor == 0) & selected).float().sum() # 真实 null-class
        total_selected = selected.float().sum()
        return false_positive / (total_selected + 1e-8)  # 防止除以零

    # 2.4) 二分查找第一个满足 R(τ) ≤ α 的 τ
    lo, hi = 0, len(candidate_taus) - 1
    best_tau = candidate_taus[-1].item()
    while lo <= hi:
        mid = (lo + hi) // 2
        tau_mid = candidate_taus[mid].item()
        if R_precision(tau_mid) <= alpha:
            best_tau = tau_mid
            hi = mid - 1
        else:
            lo = mid + 1

    return best_tau


# -----------------------------------------------------------------------------------
# Step 3: CRA pipeline: Precision-based Conformal Risk Assessment
# -----------------------------------------------------------------------------------
def cra_pipeline_precision(cal_loader: DataLoader,
                           test_loader: DataLoader,
                           alpha: float = ALPHA) -> torch.LongTensor:
    """
    Precision‐based CRA pipeline using helper functions:
      1) 全局校准 τ*（误检率控制）
      2) 计算 CRA scores
      3) 逐批计算 per‐image α*_i（误检率）
      4) 调用 compute_prediction_set_batch 构造每批的预测集
      5) 合并所有批次并返回 ——————> 不能乱 shuffle! 否则拼接会导致对应关系错误！
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 5.1) 全局校准 τ*
    tau_star = calibrate_threshold_global(cal_loader, alpha)

    all_masks = []
    for batch in tqdm(test_loader, desc="Adaptive P-CRA on test set"):
        prob_batch = batch["mask_binary"].to(device)      # (B,H,W)

        # 5.2) CRA scores
        cra_scores = compute_cra_score(prob_batch)        # (B,H,W)

        # 5.3) 直接用  s(X_i,j) ≥ τ*  选 forged 像素
        sel_masks = (cra_scores >= tau_star)              # BoolTensor (B,H,W)
        all_masks.append(sel_masks.long())

    # 5.4) 合并返回所有图像的预测掩码 (N, H, W)
    return torch.cat(all_masks, dim=0)



#---------------------------------------------------------------------------------------------------------------------------
# 概率校准（Isotonic Regression）, 校准完的放在 CRA 就是 ccra了
# --------------------------------------------------------------------------------------------------------------------------
class _CalibratedDataset(Dataset):
    """
    Wraps an existing dataset that returns dicts with keys
    "mask_binary", "mask", etc., and replaces "mask_binary"
    with its calibrated version via an Isotonic model.
    """
    def __init__(self, base_dataset, iso_model, device):
        self.base = base_dataset
        self.iso   = iso_model
        self.device = device

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        # 原 prob
        p = sample["mask_binary"].to(self.device)           # Tensor (H,W)
        flat = p.view(-1).cpu().numpy()                     # numpy 1d
        # 校准后
        p_cal = self.iso.predict(flat).reshape(p.shape)     
        sample["mask_binary"] = torch.tensor(
            p_cal, device=self.device, dtype=p.dtype
        )
        return sample

def calibrate_loaders(cal_loader: DataLoader,
                      test_loader: DataLoader) -> Tuple[DataLoader, DataLoader]:
    """
    1) 在 cal_loader 上收集 (p, y) 拟合一次 isotonic regression
    2) 返回两个新的 DataLoader：cal_loader 和 test_loader
       都会在 __getitem__ 里对 mask_binary 做相同的校准
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A) fit isotonic on entire cal_loader
    p_list, y_list = [], []
    for batch in cal_loader:
        p_list.append(batch["mask_binary"].view(-1).cpu().numpy())
        y_list.append(batch["mask"].view(-1).cpu().numpy())
    p_all = np.concatenate(p_list)
    y_all = np.concatenate(y_list)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_all, y_all)

    # B) wrap datasets
    cal_ds_new  = _CalibratedDataset(cal_loader.dataset, iso, device)
    test_ds_new = _CalibratedDataset(test_loader.dataset, iso, device)

    # C) build new loaders (preserve batch_size)
    cal_loader_new = DataLoader(
        cal_ds_new,
        batch_size=cal_loader.batch_size,
        shuffle=False,
        num_workers=getattr(cal_loader, "num_workers", 0),
        pin_memory=getattr(cal_loader, "pin_memory", False),
    )
    test_loader_new = DataLoader(
        test_ds_new,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=getattr(test_loader, "num_workers", 0),
        pin_memory=getattr(test_loader, "pin_memory", False),
    )
    return cal_loader_new, test_loader_new



# -----------------------------------------------------------------------------------
# Step 5': CCRA pipeline: Precision-based Conformal Risk Assessment
# -----------------------------------------------------------------------------------
def ccra_pipeline_precision(cal_loader: DataLoader,
                           test_loader: DataLoader,
                           alpha: float = ALPHA) -> torch.LongTensor:
    cal_loader_new, test_loader_new = calibrate_loaders(cal_loader, test_loader)
    return cra_pipeline_precision(cal_loader_new, test_loader_new, alpha)
