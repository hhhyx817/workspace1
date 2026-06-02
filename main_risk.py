# main_risk_control.py
import gc, torch
from tqdm import tqdm
from Data.load_data import dataset_loader1          # 你项目里已定义
from SABHA_algorithm.SABH_selection import sabha_pipeline_gpu
from SABHA_algorithm.evaluate import evaluate_with_selection
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------------
# Hyper-params
# -----------------------------------------------------------------------------------
ALPHA   = 0.10        # target FDR
GAMMA   = 0.05        # target tail-prob
LAM_MAX = 15          # λ searching upper bound
MC_RUNS = 100         # Monte-Carlo 
VAL_TAG = 2           # 与 main.py 保持一致 (0–4 任意选一个)
CAL_BS, TEST_BS = 16, 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------------------------------------------
# Monte Carlo: 进行一次, 执行 SABH 计算 FDR over the whole D_test
# -----------------------------------------------------------------------------------
@torch.no_grad()
def run_once(cal_loader, test_loader, alpha_eff):
    """一次 Monte-Carlo: 返回全局 FDR (所有 batch 合并)."""
    total_corr = 0
    total_wrong = 0
    for batch in test_loader:
        sel = sabha_pipeline_gpu(cal_loader,
                                 batch['mask_binary'].to(device),
                                 alpha=alpha_eff, device=device)
        c, w, _ = evaluate_with_selection(sel, batch['mask'].to(device))
        total_corr  += c
        total_wrong += w
    denom = total_corr + total_wrong
    return 0.0 if denom == 0 else total_wrong / denom


# -----------------------------------------------------------------------------------
# λ-grid search: 返回满足 tail-控制的最小 λ̂ 
# -----------------------------------------------------------------------------------
'''
alpha_eff = ALPHA / λ
重复 MC_RUNS 次, 记录 1(FDR > α) ——> 计算违反概率 p_hat ——> 计算 bootstrap CI(上下界)
若 p^ ≤ γ,返回当前 λ; 否则尝试更保守的 λ
'''
'''
此处 pi 是近似单调的, 因为 Li decrease as λ increase required by <conformal risk control>
但是 pi 是 Monte Carlo 近似的, calibration set 在变化 所以不能用 lamba 的二分 
'''
def estimate_p_hat(lam, all_results):
    alpha_eff = ALPHA / lam
    violations = []
    for mc in tqdm(range(MC_RUNS), desc=f'λ={lam:.2f}'):
        cal_loader, test_loader, _ = dataset_loader1(VAL_TAG, CAL_BS, TEST_BS, user_seed=None)
        fdr = run_once(cal_loader, test_loader, alpha_eff)
        violations.append(int(fdr > ALPHA))
        torch.cuda.empty_cache(); gc.collect()
    all_results[lam] = violations
    return np.mean(violations)

def search_lambda_three_stage():
    all_results = {}  # {λ: [0, 1, 1, 0, ...]} Monte Carlo结果
    best_lambda = None

    # ------------------ Stage 1: coarse search (step=1.0) ------------------
    print("🔍 Stage 1: coarse search (step = 1.0)")
    coarse_range = np.arange(1.0, LAM_MAX + 1, 1.0)
    for lam in coarse_range:
        alpha_eff = ALPHA / lam
        violations = []

        for mc in tqdm(range(MC_RUNS), desc=f'[Coarse] λ={lam:.2f}'):
            cal_loader, test_loader, _ = dataset_loader1(VAL_TAG, CAL_BS, TEST_BS, user_seed=None)
            fdr = run_once(cal_loader, test_loader, alpha_eff)
            violations.append(int(fdr > ALPHA))
            torch.cuda.empty_cache(); gc.collect()

        all_results[lam] = violations
        p_hat = np.mean(violations)
        lower, upper = bootstrap_ci(violations)
        print(f'λ={lam:.2f}  α_eff={alpha_eff:.3f}  p̂={p_hat:.3f}  CI: [{lower:.3f}, {upper:.3f}]')

        if p_hat <= GAMMA:
            best_lambda = lam
            break

    if best_lambda is None:
        print("❗Coarse search failed, returning most conservative λ")
        visualize_tail_probs(all_results)
        return max(coarse_range)

    # ------------------ Stage 2: fine search (step=0.05) ------------------
    print("🔍 Stage 2: fine search (step = 0.05)")
    fine_range = np.arange(best_lambda - 1.0, best_lambda + 0.05, 0.05)
    for lam in fine_range:
        if lam in all_results: continue  # 避免重复计算
        alpha_eff = ALPHA / lam
        violations = []

        for mc in tqdm(range(MC_RUNS), desc=f'[Fine] λ={lam:.2f}'):
            cal_loader, test_loader, _ = dataset_loader1(VAL_TAG, CAL_BS, TEST_BS, user_seed=None)
            fdr = run_once(cal_loader, test_loader, alpha_eff)
            violations.append(int(fdr > ALPHA))
            torch.cuda.empty_cache(); gc.collect()

        all_results[lam] = violations
        p_hat = np.mean(violations)
        lower, upper = bootstrap_ci(violations)
        print(f'λ={lam:.2f}  α_eff={alpha_eff:.3f}  p̂={p_hat:.3f}  CI: [{lower:.3f}, {upper:.3f}]')

        if p_hat <= GAMMA:
            best_lambda = lam
            break

    # ------------------ Stage 3: ultra-fine search (step=0.01) ------------------
    print("🔍 Stage 3: ultra-fine search (step = 0.01)")
    ultra_range = np.arange(best_lambda - 0.05, best_lambda + 0.01, 0.01)
    for lam in ultra_range:
        if lam in all_results: continue
        alpha_eff = ALPHA / lam
        violations = []

        for mc in tqdm(range(MC_RUNS), desc=f'[Ultra] λ={lam:.2f}'):
            cal_loader, test_loader, _ = dataset_loader1(VAL_TAG, CAL_BS, TEST_BS, user_seed=None)
            fdr = run_once(cal_loader, test_loader, alpha_eff)
            violations.append(int(fdr > ALPHA))
            torch.cuda.empty_cache(); gc.collect()

        all_results[lam] = violations
        p_hat = np.mean(violations)
        lower, upper = bootstrap_ci(violations)
        print(f'λ={lam:.2f}  α_eff={alpha_eff:.3f}  p̂={p_hat:.3f}  CI: [{lower:.3f}, {upper:.3f}]')

        if p_hat <= GAMMA:
            best_lambda = lam
            break

    visualize_tail_probs(all_results, chosen_lambda=best_lambda)
    return best_lambda



# -----------------------------------------------------------------------------------
# bootstrap CI + 可视化 
# -----------------------------------------------------------------------------------
def bootstrap_ci(bin_array, B=1000, ci=95):
    """输入: 0/1 数组 → 返回 bootstrap CI"""
    n = len(bin_array)
    samples = np.random.choice(bin_array, size=(B, n), replace=True)
    means = samples.mean(axis=1)
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


def visualize_tail_probs(all_results, chosen_lambda=None):
    """画出每个 λ 对应的 tail violation probability（以及 CI），并标出最终 λ̂"""
    lambdas = []
    p_hats = []
    ci_lowers = []
    ci_uppers = []

    for lam, bool_list in sorted(all_results.items()):
        arr = np.array(bool_list, dtype=int)
        p_hat = arr.mean()
        lower, upper = bootstrap_ci(arr)
        lambdas.append(lam)
        p_hats.append(p_hat)
        ci_lowers.append(lower)
        ci_uppers.append(upper)

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, p_hats, marker='o', label='p̂ (violation rate)')
    plt.fill_between(lambdas, ci_lowers, ci_uppers, color='gray', alpha=0.3, label='95% CI')
    plt.axhline(GAMMA, color='red', linestyle='--', label=f'γ = {GAMMA}')

    if chosen_lambda is not None:
        plt.axvline(chosen_lambda, color='blue', linestyle='--', label=f'chosen λ̂ = {chosen_lambda:.2f}')
        plt.text(chosen_lambda + 0.1, GAMMA + 0.02, f'λ̂ = {chosen_lambda:.2f}', color='blue')

    plt.xticks(lambdas)
    plt.xlabel('λ')
    plt.ylabel('P(FDR > α)')
    plt.title('Tail Probability Curve: λ ↦ p̂')
    plt.legend()
    plt.tight_layout()
    plt.savefig('risk_control_summary.png')
    print("📊 Visualization saved as 'risk_control_summary.png'.")
    
    '''
    # 保存成 csv
    import pandas as pd
    df = pd.DataFrame({
        'lambda': lambdas,
        'p_hat': p_hats,
        'CI_lower': ci_lowers,
        'CI_upper': ci_uppers
    })
    df.to_csv('/data/home/yunxin/Forgery_CP/lambda_risk_curve.csv', index=False)
    print("📄 CSV saved as 'lambda_risk_curve.csv'.")
    '''


# -----------------------------------------------------------------------------------
# final evaluation
# -----------------------------------------------------------------------------------
def final_test(lam_star):
    alpha_eff = ALPHA / lam_star
    cal_loader, test_loader, label = dataset_loader1(VAL_TAG, CAL_BS, TEST_BS, user_seed=None)
    fdr = run_once(cal_loader, test_loader, alpha_eff)

    print('\n===========  FINAL REPORT  ===========')
    print(f'Dataset tag         : {label}')
    print(f'λ̂ (chosen)         : {lam_star}')
    print(f'α_eff (used in SABHA): {alpha_eff:.4f}')
    print(f'Observed global FDR : {fdr:.4f}')
    print('======================================\n')

# -----------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    lam_hat = search_lambda_three_stage()
    final_test(lam_hat)

