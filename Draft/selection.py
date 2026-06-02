import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import torch
# --------------------------------------------------------------------------------------------------------------

def compute_conformal_pvalues_batch_torch(X_cal: torch.Tensor,
                                          Y_cal: torch.Tensor,
                                          X_test: torch.Tensor,
                                          c_const: float,
                                          batch_size: int = 64,
                                          cal_chunk_size: int = 100) -> torch.Tensor:

    device = X_cal.device

    # 将校准集中所有像素 flatten，然后筛选出 Y=1 的像素位置
    mask_flat = (Y_cal == 1).reshape(-1)                   # shape: (n_cal * H * W,)
    X_cal_flat = X_cal.reshape(-1)
    Y_cal_flat = Y_cal.reshape(-1)

    X_filtered = X_cal_flat[mask_flat]                    # 只取 Y=1 的像素
    Y_filtered = Y_cal_flat[mask_flat]
    V_cal_flat = (X_filtered - Y_filtered).abs().to(device)  # shape: (N_selected,)

    # 测试集展平
    n_test, H, W = X_test.shape
    V_test_flat = (X_test - c_const).abs().reshape(-1).to(device)

    total_cal = V_cal_flat.numel()
    total_test = V_test_flat.numel()

    p_flat = torch.zeros_like(V_test_flat, dtype=torch.float32, device=device)

    for i in range(0, total_test, batch_size):
        end = min(i + batch_size, total_test)
        v_test_batch = V_test_flat[i:end].unsqueeze(1)      # shape: (B, 1)
        B = v_test_batch.shape[0]

        less_total = torch.zeros((B,), dtype=torch.float32, device=device)
        equal_total = torch.zeros((B,), dtype=torch.float32, device=device)

        for j in range(0, total_cal, cal_chunk_size):
            cal_chunk = V_cal_flat[j:j + cal_chunk_size]    # shape: (C,)
            cal_chunk = cal_chunk.unsqueeze(0)              # shape: (1, C)

            less = (cal_chunk < v_test_batch).sum(dim=1).float()     # shape: (B,)
            equal = (cal_chunk == v_test_batch).sum(dim=1).float()   # shape: (B,)

            less_total += less
            equal_total += equal

        U = torch.rand(size=(B,), device=device)
        p_flat[i:end] = (less_total + U * equal_total) / (total_cal + 1)

    return p_flat.reshape(n_test, H, W)




def BH_selection(p_values: torch.Tensor, q: float) -> torch.Tensor:
    """    
    参数:
        - p_values: shape (n_test, H, W) 的 p-value 张量，必须在 GPU 或 CPU 上
        - q: BH FDR 控制目标 (例如 0.1)

    返回:
        - selection_matrix: shape (n_test, H, W), 元素为 0/1, 表示是否选择
    """
    device = p_values.device
    p_flat = p_values.reshape(-1)     # shape: (M,)
    m = p_flat.numel()

    sorted_p, sorted_idx = torch.sort(p_flat)  # sorted_p: (M,), sorted_idx: (M,)

    thresholds = q * torch.arange(1, m + 1, device=device) / m  # shape: (M,)

    below = sorted_p <= thresholds  # shape: (M,), bool tensor
    if not torch.any(below):
        return torch.zeros_like(p_values, dtype=torch.bool)

    k = torch.nonzero(below, as_tuple=False)[-1].item() + 1
    threshold = sorted_p[k - 1]

    selection_matrix = (p_values <= threshold).to(dtype=torch.uint8)  # 或 torch.bool
    return selection_matrix


