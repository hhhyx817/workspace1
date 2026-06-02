from SABHA_algorithm.evaluate import evaluate_with_selection # (lower_bounds, test_loader, device='cuda'):
from Data.load_data import CP_BaseData, ValColumbia, ValCasia, ValCoverage, ValIMD2020, ValNIST16
from Data.load_data import dataset_loader1, cal_dataset_loader2, test_dataset_loader2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # ← 必须在最前面！

import torch
device = torch.device("cuda:0") # 对应可见的第一块

import gc
import argparse
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from SABHA_algorithm.SABH_selection import sabha_pipeline_gpu
from CRA_compare.cra_precision import ccra_pipeline_precision, cra_pipeline_precision
import os

# -----------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------
ALPHA = 0.1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_experiments = 1
user_seed = None

print("Torch sees", torch.cuda.device_count(), "GPU(s)")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# -----------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------
for experiment_id in range(num_experiments):
    '''
    batch = {
    "image":       Tensor of shape [B, C, H, W],
    "mask":        Tensor of shape [B, H, W],   # ← 真实
    "mask_binary": Tensor of shape [B, H, W]    # ← 模型预测（概率）
    }
    '''
    gc.collect() 
    torch.cuda.empty_cache()  
    
    for val_tag in [2]: # [0, 1, 2, 3, 4]:
        FDR_global_cra = 0
        FDR_global_ccra = 0
        FDR_global_bh = 0
        cal_loader, test_loader, data_label = dataset_loader1(val_tag, 16, 6043,user_seed=42) # cal_batch_size test_batch_size
        # -----------------------------------------------------------------------------------
        # Step1: CRA selection
        # -----------------------------------------------------------------------------------
        total_correct_rejects_cra = 0
        total_wrong_rejects_cra = 0
        
        # 1.1) reject
        selected_mask = cra_pipeline_precision(cal_loader, test_loader, ALPHA)
        selected_mask = selected_mask.to(device).bool()

        # 1.2）groundtruth
        all_groundtruth_masks = []
        for batch in test_loader:
            test_groundtruth_batch = batch["mask"].to(device) 
            all_groundtruth_masks.append(test_groundtruth_batch)
            
        all_groundtruth_masks = torch.cat(all_groundtruth_masks, dim=0)  # 合并成 (N, H, W)
        all_groundtruth_masks = all_groundtruth_masks.bool()             # 转为布尔类型
        
        # 1.3）选中了且真的 forgery TP → 等价于正确拒绝；选中了但其实是 null FP → 等价于错误拒绝
        correct_mask = selected_mask & all_groundtruth_masks      
        wrong_mask   = selected_mask & (~all_groundtruth_masks)   

        
        # 1.4） 计数
        total_P = all_groundtruth_masks.sum().item() # recall 这个只要一个地方算就可以了
         
        total_correct_rejects_cra = correct_mask.sum().item()
        total_wrong_rejects_cra   = wrong_mask.sum().item()
        FDR_global_cra = total_wrong_rejects_cra / (total_correct_rejects_cra + total_wrong_rejects_cra)
        
        # -----------------------------------------------------------------------------------
        # Step1': CCRA selection
        # -----------------------------------------------------------------------------------
        total_correct_rejects_ccra = 0
        total_wrong_rejects_ccra = 0
        
        # 1.1') reject
        selected_mask_ = ccra_pipeline_precision(cal_loader, test_loader, ALPHA)
        selected_mask_ = selected_mask_.to(device).bool()

        # 1.2'）groundtruth
        all_groundtruth_masks_ = []
        for batch in test_loader:
            test_groundtruth_batch_ = batch["mask"].to(device) 
            all_groundtruth_masks_.append(test_groundtruth_batch_)
            
        all_groundtruth_masks_ = torch.cat(all_groundtruth_masks_, dim=0)  # 合并成 (N, H, W)
        all_groundtruth_masks_ = all_groundtruth_masks_.bool()             # 转为布尔类型
        
        # 1.3'）选中了且真的 forgery → 正确拒绝；选中了但其实是 null → 错误拒绝
        correct_mask_ = selected_mask_ & all_groundtruth_masks_      
        wrong_mask_   = selected_mask_ & (~all_groundtruth_masks_)   

        # 1.4'） 计数
        total_correct_rejects_ccra = correct_mask_.sum().item()
        total_wrong_rejects_ccra   = wrong_mask_.sum().item()
        FDR_global_ccra = total_wrong_rejects_ccra / (total_correct_rejects_ccra + total_wrong_rejects_ccra)
        
        # -----------------------------------------------------------------------------------
        # Step2: BH selection
        # -----------------------------------------------------------------------------------
        total_correct_rejects_bh = 0
        total_wrong_rejects_bh = 0
        
        for batch_idx, batch in enumerate(test_loader):
            test_prob_batch        = batch["mask_binary"]
            test_groundtruth_batch = batch["mask"]

            # ① SABHA 选点
            selected_mask = sabha_pipeline_gpu(cal_loader, test_prob_batch, ALPHA, device = device) # (B,256,256)matrix
   
            # ② 统计 TP / FP 以及 batch-level FDR
            tp_cnt, fp_cnt, FDR_batch = evaluate_with_selection(
                selected_mask,
                test_groundtruth_batch
            )

            # ③ 计算 batch-level power
            positives_batch = test_groundtruth_batch.sum().item()
            power_batch     = tp_cnt / max(positives_batch, 1)

            # ④ 打印
            print(f"[SABHA][Batch {batch_idx:3d}] "
            f"FDR: {FDR_batch:.4f} | power: {power_batch:.4f} "
            f"| TP: {tp_cnt} | FP: {fp_cnt} | P: {tp_cnt + fp_cnt}")

            # ⑤ 全局累计（保持原来的三行不变）
            total_correct_rejects_bh += tp_cnt
            total_wrong_rejects_bh   += fp_cnt

            
        # -----------------------------------------------------------------------------------
        # FDR over the whole dataset
        # -----------------------------------------------------------------------------------
        total_selected_bh = total_correct_rejects_bh + total_wrong_rejects_bh
        FDR_global_bh = total_wrong_rejects_bh / max(total_selected_bh, 1)
        power_global_bh = total_correct_rejects_bh / total_P
        print(f"[SABHA] Total Rejects: {total_selected_bh}")
        # print(f"[SABHA] Total Correct Rejects: {total_correct_rejects_bh}")
        print(f"[SABHA] Total Wrong Rejects:    {total_wrong_rejects_bh}")
        print(f"[SABHA] Global FDR:             {FDR_global_bh:.4f}")
        print(f"[SABHA] Global power:           {power_global_bh:.4f}")

        total_selected_cra = total_correct_rejects_cra + total_wrong_rejects_cra
        FDR_global_cra = total_wrong_rejects_cra / max(total_selected_cra, 1) 
        recall_global_cra = total_correct_rejects_cra / total_P
        print(f"[CRA]   Total P:                {total_selected_cra}")  
        # print(f"\n[CRA]   Total Correct Rejects: {total_correct_rejects_cra}")
        print(f"[CRA]   Total FP:               {total_wrong_rejects_cra}")
        print(f"[CRA]   Global 1-precision:     {FDR_global_cra:.4f}")
        print(f"[CRA]   Global recall:          {recall_global_cra:.4f}")

        total_selected_ccra = total_correct_rejects_ccra + total_wrong_rejects_ccra
        FDR_global_ccra = total_wrong_rejects_ccra / max(total_selected_ccra, 1) 
        recall_global_ccra = total_correct_rejects_ccra / total_P 
        print(f"[CCRA]   Total P:               {total_selected_ccra}")   
        # print(f"\n[CCRA]   Total Correct Rejects: {total_correct_rejects_ccra}")
        print(f"[CCRA]   Total FP:              {total_wrong_rejects_ccra}")
        print(f"[CCRA]   Global 1-precision:    {FDR_global_ccra:.4f}")
        print(f"[CCRA]   Global recall:         {recall_global_ccra:.4f}")
            
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    
    f1_list = [], auc_list = []
    precision_list = [], recall_list = []
    c_const = 1.00  q = 0.1  gamma = 0.5  cal_tag = 0
    significance = 0.1
    ncm_scores = []

    data_columbia = ValColumbia()
    data_coverage = ValCoverage()
    data_casia = ValCasia()
    data_NIST16 = ValNIST16()
    data_IMD202 = ValIMD2020()
    #------------------------------------------------------------------------------------------------------
    # select cal_data
    #------------------------------------------------------------------------------------------------------
    X_all, Y_all = [], []
    combined_dataset = ConcatDataset([data_columbia, data_coverage, data_casia, data_NIST16, data_IMD202])
    cal_loader = cal_dataset_loader2(combined_dataset, batch_size)
    # cal_loader, test_loader, data_label = cal_dataset_loader1(cal_tag, batch_size)
    
    for batch in cal_loader:                   # accelerate the data processing (reading)
        X_all.append(batch["mask_binary"])     # probability of positivity
        Y_all.append(batch["mask"])            # groudtruth 0/1
        
    X_cal_tensor = torch.cat(X_all, dim=0)  # X_all: batch_size*H*W ; X_tensor: size_of_full_set(sum of batch_size) *H*W
    Y_cal_tensor = torch.cat(Y_all, dim=0)  # X_all contains matrices with size [batch_size,H,W], concat along dim[0] = batch
   
    true_positive_mask = (X_cal_tensor >= 0.5) & (Y_cal_tensor == 1)
    true_positive_indices = torch.nonzero(true_positive_mask, as_tuple=False) 
    
    #------------------------------------------------------------------------------------------------------
    # test data
    #------------------------------------------------------------------------------------------------------
    for val_tag in [3]: # [0, 1, 2, 3, 4]:
        val_data_loader, data_label = test_dataset_loader2(val_tag, batch_size)
        print(f"calibrate on the dataset: {data_label}.")
        
        X_test_list = []
       
        for batch in val_data_loader:
            x= batch["mask_binary"]         # shape: (1, H, W)
            X_test_list.append(x)            # 不需要 reshape

        X_test_tensor = torch.cat(X_test_list, dim=0)  # shape: (N, H, W)

        
        p_value_test = compute_conformal_pvalues_batch_torch(X_cal_tensor, Y_cal_tensor, X_test_tensor, c_const, batch_size=64,
                                        cal_chunk_size = 100) 
        selection_matrix = BH_selection(p_value_test, q)

        precision, f1, auc, TP, FP, TN, FN \
                = evaluate_with_selection(
                  selection_matrix, val_data_loader, device='cuda')
                
        print("min p-value:", p_value_test.min().item())
        print("selection_matrix positive count:", selection_matrix.sum().item())

                
        if experiment_id != num_experiments: # the result of the val_tag D_test in id-th experiments
            f1_list.append(f1)
            auc_list.append(auc)
            precision_list.append(precision)
       
        print(f"{experiment_id + 1}, significance: {significance:.2f}\n"
              f"F1 Score: {f1:.2f}, AUC: {auc:.2f}, precison: {precision:.2f}\n"
              f"[TP, FP, TN, FN] = [{TP:.2f}, {FP:.2f}, {TN:.2f}, {FN:.2f}\n]")
        
'''        
#------------------------------------------------------------------------------------------------------          
# var & std
#------------------------------------------------------------------------------------------------------  
mean_precision, var_precision, std_precision = np.mean(precision_list), np.var(precision_list), np.std(precision_list)
mean_f1, var_f1, std_f1 = np.mean(f1_list), np.var(f1_list), np.std(f1_list)
mean_auc, var_auc, std_auc = np.mean(auc_list), np.var(auc_list), np.std(auc_list)

print("\n=============================================")
print(f"Precision -> mean: {mean_precision:.4f}, variance: {var_precision:.4f}, std: {std_precision:.4f}")
# print(f"Recall  -> mean: {mean_recall:.4f},  variance: {var_recall:.4f},  std: {std_recall:.4f}")
print(f"F1  -> mean: {mean_f1:.4f},  variance: {var_f1:.4f},  std: {std_f1:.4f}")
print(f"AUC -> mean: {mean_auc:.4f}, viriance: {var_auc:.4f}, std: {std_auc:.4f}")
print("===============================================")
'''


"""