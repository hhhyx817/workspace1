import torch
import numpy as np
from sklearn import metrics

# ----------------------------------------------------------------------------------------------------------------
import torch

def evaluate_with_selection(selection_matrix: torch.Tensor,
                            test_groundtruth_batch: torch.Tensor):
    """
    参数:
        - selection_matrix: (n_test, H, W) 的 uint8/bool tensor, 被拒绝null hypothesis 的像素位置
        - test_groundtruth_batch: (B, 256, 256) Tensor, 和 SABH_selection 输入的 test_prob_batch 对应
                                这个变量表示的是测试集像素的 mask 变量
    """
    test_groundtruth_batch = test_groundtruth_batch.to(selection_matrix.device)

    selection_matrix = selection_matrix.bool()
    test_groundtruth_batch = test_groundtruth_batch.bool()

    # 只考虑 selection == 1 的地方
    selected_positions = selection_matrix
    
    correct_reject = selected_positions & (test_groundtruth_batch == 1) 
    correct_reject_count = correct_reject.sum().item()
    wrong_reject = selected_positions & (test_groundtruth_batch == 0)
    wrong_reject_count = wrong_reject.sum().item()

    if (wrong_reject_count == 0) and (correct_reject_count == 0):
        FDR = -1.0
    else:
        FDR = wrong_reject_count / (correct_reject_count + wrong_reject_count)

    return correct_reject_count, wrong_reject_count, FDR




# ----------------------------------------------------------------------------------------------------------------
# TODO 最后去区分一下这个, HIFI用了两种方式去判断 positivity, 随时函数什么的要不要改？
'''
pred_mask_score: 
    Per-pixel forgery probability score matrix, similar to `mask_binary` but calculated differently:
    - Value range: [0,1] (continuous values).
    - **If `args.loss_type == 'dm'`**:
        - Computed using `LOSS_MAP.dist`, representing the model's forgery score for each pixel.
        - `dm` (Deep Metric Loss): Suitable for **fine-grained algorithms and localization tasks**, focusing on feature representation of forgery regions.
    - **If `args.loss_type == 'ce'`**:
        - Directly equal to `mask_binary` (i.e., the forgery probability output by the model).
        - `ce` (Cross Entropy Loss): Suitable for **classification tasks**, outputting the **forgery probability distribution**.
'''
# ----------------------------------------------------------------------------------------------------------------

'''
    TP, FP, TN, FN = 0, 0, 0, 0
    total_f1, total_auc = 0.0, 0.0
    total_count = 0

    idx_counter = 0

    for batch in test_loader:
        masks = batch["mask"].to(device)                # (B, H, W), ground truth
        mask_binarys = batch["mask_binary"].to(device)  # (B, H, W), predicted probability ∈ [0,1]
        batch_size = masks.shape[0]

        preds = selection_matrix[idx_counter:idx_counter+batch_size].to(device).bool()  # (B, H, W)
        idx_counter += batch_size

        # Flatten
        pred_flat = preds.view(batch_size, -1).float()
        mask_flat = masks.view(batch_size, -1).float()
        prob_flat = mask_binarys.view(batch_size, -1).float()

        # 统计 TP, FP, TN, FN
        TP += ((pred_flat == 1) & (mask_flat == 1)).sum().item()
        FP += ((pred_flat == 1) & (mask_flat == 0)).sum().item()
        TN += ((pred_flat == 0) & (mask_flat == 0)).sum().item()
        FN += ((pred_flat == 0) & (mask_flat == 1)).sum().item()

        # 计算 F1（flipped, macro）
        tp = (pred_flat * mask_flat).sum(dim=1)
        fp = (pred_flat * (1 - mask_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * mask_flat).sum(dim=1)

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        total_f1 += f1.sum().item()
        total_count += batch_size

        # AUC（近似法：Batched ROC AUC，方向翻转处理）
        for i in range(batch_size):
            y_true = mask_flat[i]
            y_score = prob_flat[i]
            if torch.unique(y_true).numel() > 1:
                # Rank-based AUC approximation via concordant pairs
                y_true = y_true.bool()
                pos = y_score[y_true]
                neg = y_score[~y_true]
                if pos.numel() > 0 and neg.numel() > 0:
                    # Broadcasting: pos > neg: (P, N) → bool
                    auc_val = (pos[:, None] > neg[None, :]).float().mean().item()
                    auc_val = auc_val if auc_val > 0.5 else 1 - auc_val
                else:
                    auc_val = 0.0
            else:
                auc_val = 0.0
            total_auc += auc_val
            
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = total_f1 / total_count
    auc = total_auc / total_count

    return precision, f1, auc, TP, FP, TN, FN
'''
