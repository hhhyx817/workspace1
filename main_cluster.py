import gc
import argparse
import numpy as np
from SABHA_algorithm.SABH_cluster_selection import sabha_cluster_pipeline
from Data.load_data import CP_BaseData, ValColumbia, ValCasia, ValCoverage, ValIMD2020, ValNIST16
from Data.load_data import dataset_loader1, cal_dataset_loader2, test_dataset_loader2
import os
import torch
import time


ALPHA            = 0.1
PARTITIONS       = 8
CLUSTERS_PER_PAR = [75] * PARTITIONS
GAMMA_CORRECTION = np.ones(sum(CLUSTERS_PER_PAR), dtype=np.float32)
BATCH_SIZE_KMEAN = 10_000
USER_SEED        = 42
VAL_TAGS         = [2]

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch sees", torch.cuda.device_count(), "GPU(s)")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

for val_tag in VAL_TAGS:
    print(f"\n==========  Validation tag {val_tag}  ==========")

    cal_loader, test_loader, _ = dataset_loader1(
        val_tag, cal_batch_size=16, test_batch_size=6043, user_seed=USER_SEED
    )

    test_probs = torch.cat([b["mask_binary"] for b in test_loader]).cpu().numpy()
    cal_probs  = torch.cat([b["mask_binary"] for b in cal_loader]).cpu().numpy()
    cal_labels = torch.cat([b["mask"]        for b in cal_loader]).cpu().numpy()
    all_gt     = torch.cat([b["mask"]        for b in test_loader]).cpu().numpy()

    # =================== Clustering‑SABHA ==================
    start = time.time()
    results = sabha_cluster_pipeline(
        test_probs=test_probs,
        test_labels=all_gt,
        cal_probs=cal_probs,
        cal_labels=cal_labels,
        n_partitions=PARTITIONS,
        clusters_per_partition=CLUSTERS_PER_PAR,
        alpha=ALPHA,
        gamma=GAMMA_CORRECTION,
        batch_size=BATCH_SIZE_KMEAN,
    )
    elapsed = time.time() - start
    print(f"[Clust‑SABHA] Pipeline finished in {elapsed:.2f}s")

    print("\n================= [Summary: SABHA Pipeline] =================")
    print(f"像素级 FDR  : {results['FDR']:.4f}")
    print(f"Power      : {results['power']:.4f}")
    print(f"total wrong rejects : {results['total_wrong_rejects']}")
    print(f"total rejects      : {results['total_rejects']}")
    print("=============================================================\n")
