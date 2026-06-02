import numpy as np
from typing import Tuple, Dict
from sklearn.cluster import MiniBatchKMeans
from tqdm import trange

# ------------------------------------------------------------------
# quantile-based partition: 
# Hierarchical clustering不合适, 对时间内存要求更高/ 等概率间隔分层也不行,希望每一层内数量尽量均衡
# ------------------------------------------------------------------
def partition(test_probs: np.ndarray, n_partitions: int):
    B, H, W = test_probs.shape
    probs_flat = test_probs.flatten()

    # 计算 quantile 边界, [0, 0.2, 0.3, 0.4] median = avg([0.2,0.3])
    quantile_edges = np.quantile(probs_flat, np.linspace(0, 1, n_partitions + 1))

    # digitize 返回 bin 的编号（1 开始）
    partition_ids = np.digitize(probs_flat, quantile_edges, right=False) - 1
    partition_ids = np.clip(partition_ids, 0, n_partitions - 1)  # ensure ids are within bounds
    partition_map = partition_ids.reshape(B, H, W)

    return partition_map
    

''' 没用上, 放 pipeline 里效果不大
# ------------------------------------------------------------------
# clustering: 唯一指标是概率
# ------------------------------------------------------------------
def Kclustering(n_clusters, batch_size, test_probs: np.ndarray) -> np.ndarray:
    B, H, W = test_probs.shape
    probs_flat = test_probs.reshape(-1, 1)  # shape: (N, 1)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
    
    # kmeans.fit_predict生成簇的id
    cluster_ids_flat = kmeans.fit_predict(probs_flat)  # shape: (B*H*W,)
    cluster_id_map = cluster_ids_flat.reshape(B, H, W)
    
    return cluster_id_map # shape: (B, H, W)
'''

# ------------------------------------------------------------------
# Compute cluster-level p-values using conformal calibration
# ------------------------------------------------------------------
def compute_cluster_pvals(cal_probs: np.ndarray,
                          test_cluster_probs: np.ndarray) -> np.ndarray:
    """
    For each cluster, compute p-value as right-tail conformal p-value of its average test prob
    against distribution of calibration cluster probabilities.
    通过conformal 算法, 用校准集概率分布为每个簇的平均概率计算p值
    """
    # 校准集概率排序
    cal_probs_flat = cal_probs.reshape(-1)
    V_sorted = np.sort(cal_probs_flat)
    n_cal = len(V_sorted)
    
    # compute p-values per cluster
    V_test = test_cluster_probs
    idx_gt = np.searchsorted(V_sorted, V_test, side='right')
    idx_eq = np.searchsorted(V_sorted, V_test, side='left')
    n_greater = n_cal - idx_gt
    n_equal = idx_gt - idx_eq
    U = np.random.rand(len(V_test)).astype(np.float32)
    pvals = (n_greater + U * (n_equal + 1)) / (n_cal + 1)
    return pvals


# ------------------------------------------------------------------
# Compute cluster-level π0 using calibration labels
# ------------------------------------------------------------------
def estimate_pi0_from_cal(cal_labels: np.ndarray, 
                          cali_cluster_id_map: np.ndarray) -> np.ndarray:
    '''
    将测试集的聚类函数作用于校准集, 分为 0,1,..., C-1 类, 在校准集中求出每个簇的 π
    测试集的簇 0,1,..., C-1 也对应相同的 π
    此处 cali_cluster_id_map 是校准集的聚类结果, 具体聚类过程见 Kclustering
    '''
    C = cali_cluster_id_map.max() + 1
    pi0 = np.ones(C, dtype=np.float32)

    for c in range(C):
        mask = (cali_cluster_id_map == c)
        if mask.sum() == 0:
            continue
        labels = cal_labels[mask]
        pi0[c] = np.mean(labels == 0)  # null (真实) 的比例

    return pi0

# ------------------------------------------------------------------
# SABHA procedure
# ------------------------------------------------------------------
def sabha(pvals: np.ndarray,
            pi0: np.ndarray,
            alpha: float = 0.1) -> np.ndarray:

    m = len(pvals)
    # Sort p-values
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    pi0_sorted = pi0[order]

    # Compute adaptive thresholds
    thresholds = (np.arange(1, m + 1) / m) * alpha / pi0_sorted

    # Find largest k where p_(k) <= threshold_k
    below = p_sorted <= thresholds
    if np.any(below):
        k_max = np.nonzero(below)[0].max()
        reject_idx = order[:k_max + 1]
    else:
        reject_idx = np.array([], dtype=int)

    # Initialize rejection array
    reject = np.zeros(m, dtype=bool)
    reject[reject_idx] = True

    return reject # 与 pvals 同形状 的 NumPy 布尔数组


# ------------------------------------------------------------------
# 修正聚类所导致的误差, associated with clustering std
# ------------------------------------------------------------------
def adjust_pi0_by_cluster_error(pi0_raw: np.ndarray,
                                cal_probs: np.ndarray,
                                cluster_id_map: np.ndarray,
                                gamma: np.ndarray) -> np.ndarray:
    """
    cal_probs: 校准集像素概率 (N_cal, H, W)
    cluster_id_map: 测试集的聚类 ID 映射 (N_cal, H, W)
    gamma: 调整系数
    """
    C = pi0_raw.shape[0]
    pi0_adj = np.copy(pi0_raw)

    for c in range(C):
        mask = (cluster_id_map == c)
        if np.sum(mask) == 0:
            continue
        std = np.std(cal_probs[mask])
        pi0_adj[c] = min(1.0, pi0_raw[c] + gamma[c] * std)

        # TODO 求 gamma
    return pi0_adj


# ------------------------------------------------------------------
# pipeline
# ------------------------------------------------------------------
def sabha_cluster_pipeline(
    test_probs: np.ndarray,                # shape: (B, H, W)
    test_labels: np.ndarray,               # shape: (B, H, W), 0(真实), 1(伪造)
    cal_probs: np.ndarray,                 # shape: (N_cal, H, W)
    cal_labels: np.ndarray,                # shape: (N_cal, H, W), 0 (真实), 1 (伪造)
    n_partitions: int,                     # 层数
    clusters_per_partition: list,          # 每层聚类数, 长度为 n_partitions 的列表
    alpha: float,                          # 目标 FDR
    gamma: np.ndarray,                          # pi0 修正强度
    batch_size: int                        # KMeans 小批量大小
) -> dict:
    """
    统计簇级别的 wrong reject, reject 累加成 total wrong reject, total reject
        {   'FDR': float,
            'total_wrong_rejects': int,
            'total_rejects': int,
            'power': float
        }
    """
    assert len(clusters_per_partition) == n_partitions, "clusters_per_partition 长度应与 n_partitions 一致"

    B, H, W = test_probs.shape
    partition_map = partition(test_probs, n_partitions)

    # Step 1: 测试集分层聚类
    cluster_id_map = np.zeros((B, H, W), dtype=np.int32)
    cluster_base = 0
    for part_id in range(n_partitions):
        mask = (partition_map == part_id)
        probs_flat = test_probs[mask].reshape(-1, 1)
        n_clusters = clusters_per_partition[part_id]
        if len(probs_flat) == 0:
            continue
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
        cluster_ids = kmeans.fit_predict(probs_flat)
        cluster_id_map[mask] = cluster_ids + cluster_base
        cluster_base += n_clusters

    C = cluster_id_map.max() + 1

    # Step 2: 每个簇的平均概率（用于计算 p 值）
    test_cluster_probs = np.zeros(C, dtype=np.float32)
    for c in range(C):
        m = (cluster_id_map == c)
        if m.sum() == 0:
            continue
        test_cluster_probs[c] = test_probs[m].mean()

    # Step 3: p 值计算
    cluster_pvals = compute_cluster_pvals(cal_probs, test_cluster_probs)

    # Step 4: 校准集分层聚类
    cal_partition_map = partition(cal_probs, n_partitions)
    cal_cluster_id_map = np.zeros_like(cal_partition_map, dtype=np.int32)
    cluster_base = 0
    for part_id in range(n_partitions):
        mask = (cal_partition_map == part_id)
        probs_flat = cal_probs[mask].reshape(-1, 1)
        n_clusters = clusters_per_partition[part_id]
        if len(probs_flat) == 0:
            continue
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
        cluster_ids = kmeans.fit_predict(probs_flat)
        cal_cluster_id_map[mask] = cluster_ids + cluster_base
        cluster_base += n_clusters

    # Step 5: π₀ 估计和修正
    pi0_raw = estimate_pi0_from_cal(cal_labels, cal_cluster_id_map)
    pi0_adj = adjust_pi0_by_cluster_error(pi0_raw, cal_probs, cal_cluster_id_map, gamma)

    # Step 6: SABHA 选择（得到每个簇是否reject）
    reject_clusters = sabha(cluster_pvals, pi0_adj, alpha)   # shape: (C, )

    # Step 7: 按簇统计 total_rejects / total_wrong_rejects / power
    total_rejects = 0
    total_wrong_rejects = 0
    total_true_1 = 0
    total_reject_true_1 = 0

    for c in range(C):
        mask = (cluster_id_map == c)
        if mask.sum() == 0:
            continue
        cluster_label = test_labels[mask]
        if reject_clusters[c]:
            total_rejects += mask.sum()  # <--- 统计被拒绝簇所有像素数
            n_wrong = np.sum(cluster_label == 0)
            n_true_1 = np.sum(cluster_label == 1)
            total_wrong_rejects += n_wrong
            total_reject_true_1 += n_true_1
        # 真实值1, correct reject  用来求 power 的
        total_true_1 += np.sum(cluster_label == 1)

    FDR = total_wrong_rejects / max(total_rejects, 1)
    power = total_reject_true_1 / max(total_true_1, 1)

    return {
        'FDR': FDR,
        'total_wrong_rejects': total_wrong_rejects,
        'total_rejects': total_rejects,
        'power': power
    }

# ------------------------------------------------------------------
# TODO 
def find_optimal_gamma(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_cluster_id_map: np.ndarray,
    gamma_list: list,
    alpha: float,
    delta: float = 0.05
):
    """
    用 Bernstein 不等式控制 FDR 上界，选择满足条件的最小 gamma
    """
    clusters = np.unique(cal_cluster_id_map)
    n_c = {c: (cal_cluster_id_map == c).sum() for c in clusters}

    results = []

    for gamma in gamma_list:
        fdr_upper = 0
        for c in clusters:
            mask = (cal_cluster_id_map == c)
            if mask.sum() == 0:
                continue
            pi_hat = np.mean(cal_labels[mask] == 0)  # 估计真实比例
            sigma_hat = np.sqrt(pi_hat * (1 - pi_hat))  # Bernoulli std

            term = -n_c[c] * (gamma**2 * sigma_hat**2) / (2 * sigma_hat**2 + 2 * gamma * sigma_hat / 3 + 1e-8)
            fdr_upper += np.exp(term)

        if fdr_upper <= delta:
            results.append((gamma, fdr_upper))

    if not results:
        raise ValueError("No gamma satisfies Bernstein union bound for FDR control")

    gamma_best = min(results, key=lambda x: x[0])[0]
    return gamma_best
