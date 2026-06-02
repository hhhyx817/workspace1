import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # ← 必须在最前面！

import torch
device = torch.device("cuda:0") # 对应可见的第一块


import random, gc, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_1samp
from torch.utils.data import DataLoader
from scipy.stats import uniform
from Data.load_data import dataset_loader1
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import norm, shapiro
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ----------------------------------------
# 配置
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_CAL_PIXELS = 200 * 256 * 256      # 全量提取校准集像素
ALPHA_IMG      = 0.05                 # KS 显著性水平
CAL_BATCH      = 16                 # 校准阶段 batch_size
TEST_BATCH     = 1420                # 测试阶段 batch_size
SAVE_DIR       = "/data/home/yunxin/Forgery_CP/Data/shift_test_casia"  # 统一保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

PIXELS_PER_MB = 1024 * 1024 / 4  # float32 4字节
MAX_GPU_MEM_MB = 24564 * 0.7
MAX_PIXELS_TEST = int(MAX_GPU_MEM_MB * PIXELS_PER_MB)

EPS = 1e-6    # 全局clip eps 
N_SAMPLE_DRAW = 10000  # 后续画图和独立性检测抽样数

# ----------------------------------------
# 校准集：提取所有 nonconformity scores，并排序
# ----------------------------------------
def build_calibration_scores_gpu(cal_loader, device=device):
    scores = []

    for batch in tqdm(cal_loader, desc="Collect calibration"):
        p = batch["mask_binary"].float().to(device, non_blocking=True)
        y = batch["mask"].float().to(device, non_blocking=True)
        v = torch.abs(p - y).view(-1)  # 直接flatten一维
        scores.append(v)

    V_all = torch.cat(scores, dim=0)  # dim=0拼接像素
    V_sorted, _ = torch.sort(V_all)
    return V_sorted, V_sorted.shape[0]

# ----------------------------------------
# p-value计算（在GPU）
# ----------------------------------------
def pixel_pvalues_gpu(V_sorted, V_test_flat):
    n_cal = V_sorted.numel()
    idx_lt = torch.searchsorted(V_sorted, V_test_flat, right=False)
    idx_le = torch.searchsorted(V_sorted, V_test_flat, right=True)

    n_less = idx_lt
    n_eq = idx_le - idx_lt

    U = torch.rand_like(n_less, dtype=torch.float32, device=V_sorted.device)

    pvals = (n_less.float() + U * (n_eq.float() + 1)) / (n_cal + 1)
    return pvals

# ----------------------------------------
# 主流程
# ----------------------------------------
for val_tag in [2]:
    cal_loader, test_loader, data_label = dataset_loader1(
                                        val_tag, cal_batch_size=CAL_BATCH, test_batch_size=TEST_BATCH, user_seed=42)
    V_sorted, n_cal = build_calibration_scores_gpu(cal_loader)
    torch.cuda.empty_cache(); gc.collect()

    print(f"\n Dataset {val_tag} ")

    p_values_all_batches = []
    batch_idx = 0

    for batch in tqdm(test_loader, desc=f"Testing dataset {data_label}"):
        p_imgs = batch["mask_binary"].float().to(device, non_blocking=True)
        y_imgs = batch["mask"].float().to(device, non_blocking=True)
        '''
        V_imgs = torch.abs(p_imgs - y_imgs).view(-1)  # Flatten [B,256,256]
        '''
        # 先计算nonconformity scores
        V_imgs = torch.abs(p_imgs - y_imgs)
        # 筛选null区域, bool 索引自动展平, 不用 view(-1)了
        null_mask = (y_imgs == 0)  
        V_null = V_imgs[null_mask] 

        '''
        num_pixels = V_imgs.shape[0]
        if num_pixels > MAX_PIXELS_TEST:
            selected_idx = torch.randperm(num_pixels, device=device)[:MAX_PIXELS_TEST]
            V_sampled = V_imgs[selected_idx]
        else:
            V_sampled = V_imgs
        '''    
        num_null_pixels = V_null.shape[0]
        if num_null_pixels > MAX_PIXELS_TEST:
            selected_idx = torch.randperm(num_null_pixels, device=device)[:MAX_PIXELS_TEST]
            V_sampled = V_null[selected_idx]
        else:
            V_sampled = V_null


        pvals = pixel_pvalues_gpu(V_sorted, V_sampled)
        pvals_cpu = pvals.detach().cpu()

        pvals_np = pvals_cpu.numpy()

        # 5) KS检验
        _, p_ks = ks_1samp(pvals_np, uniform.cdf) # , method='asymp'
        p_values_all_batches.append(p_ks)
        print(f"Batch {batch_idx} p-value: {p_ks:.5f} " + ("[SHIFT]" if p_ks < ALPHA_IMG else "[OK]"))

        # 6) 检查相关性 + 正态性
        pvals_clip = np.clip(pvals_np, EPS, 1 - EPS)
        z_scores = norm.ppf(pvals_clip)

        # 抽样用于画图 + 检验
        if len(pvals_np) > N_SAMPLE_DRAW:
            idx_sample = np.random.choice(len(pvals_np), N_SAMPLE_DRAW, replace=False)
            pvals_np_sample = pvals_np[idx_sample]
            z_scores_sample = z_scores[idx_sample]
        else:
            pvals_np_sample = pvals_np
            z_scores_sample = z_scores

        # 做Ljung-Box检验（只对sample）
        lb_test = acorr_ljungbox(pvals_np_sample, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].values[0]
        print(f"Batch {batch_idx} Ljung-Box p-value: {lb_pvalue:.5f} " + ("[Not Independent!]" if lb_pvalue < ALPHA_IMG else "[Independent]"))

        # ----------------------------------------
        # 画图
        # ----------------------------------------
        if p_ks < ALPHA_IMG:
            mean_pval = pvals_np.mean()
            q1 = np.percentile(pvals_np, 25)
            q3 = np.percentile(pvals_np, 75)

            pvals_sorted = np.sort(pvals_np)
            ecdf = np.linspace(0, 1, len(pvals_sorted))

            kde = gaussian_kde(pvals_np)
            x_pdf = np.linspace(0, 1, 200)
            y_pdf = kde(x_pdf)

            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()

            # 子图1: p-value散点图
            axs[0].scatter(range(len(pvals_np)), pvals_np, s=0.01, color='black', alpha=0.7, label='p-values scatter')
            axs[0].axhline(y=mean_pval, color='g', linestyle='--', label=f'Mean={mean_pval:.4f}')
            axs[0].axhline(y=q1, color='blue', linestyle='-.', label=f'Q1 (25%)={q1:.4f}')
            axs[0].axhline(y=q3, color='purple', linestyle='-.', label=f'Q3 (75%)={q3:.4f}')
            axs[0].set_title(f'(1) p-value Scatter Plot (Batch {batch_idx})')
            axs[0].legend()
            axs[0].grid(True)

            # 子图2: 经验CDF
            axs[1].plot(pvals_sorted, ecdf, color='red', linewidth=1.5, label='Empirical CDF')
            axs[1].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Ideal Uniform')
            axs[1].set_title('(2) Empirical CDF')
            axs[1].legend()
            axs[1].grid(True)

            # 子图3: 经验PDF
            axs[2].plot(x_pdf, y_pdf, color='blue', linewidth=1.5, label='Estimated PDF')
            axs[2].axhline(y=1.0, color='gray', linestyle='--', label='Uniform Density=1')
            axs[2].set_title('(3) Estimated PDF')
            axs[2].legend()
            axs[2].grid(True)

            # 子图4: ACF自相关图（画sample）
            plot_acf(pvals_np_sample, lags=40, ax=axs[3])
            axs[3].set_title('(4) p-value ACF (sampled)')

            # 子图5: z_scores直方图
            axs[4].hist(z_scores_sample, bins=100, density=True, color='skyblue', edgecolor='black')
            x = np.linspace(-4, 4, 200)
            axs[4].plot(x, norm.pdf(x), 'r--', label='Standard Normal PDF')
            axs[4].set_title('(5) Histogram of Normal Scores (sampled)')
            axs[4].legend()
            axs[4].grid(True)

            # 子图6: QQ图
            stats.probplot(z_scores_sample, dist="norm", plot=axs[5])
            axs[5].set_title('(6) QQ Plot of Normal Scores (sampled)')

            plt.tight_layout()

            save_path = os.path.join(SAVE_DIR, f'pvalue_scatter_cdf_pdf_val{val_tag}_batch{batch_idx}.png')
            plt.savefig(save_path)
            plt.close()

        batch_idx += 1
