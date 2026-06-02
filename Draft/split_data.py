import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader
import torch
import math
import hashlib

# 本来在 Data 里面, 把内容合并到 load_data.py 了, 估计没用了

# ---------------------------------------------------------------------------------------------------------------------
# 数据集内部抽取, 没意外的话用的是这个  D_cal, D_test = D\D_cal
#----------------------------------------------------------------------------------------------------------------------

def split_dataset_cal(dataset, calibration_ratio):
    # 1. 自动根据 dataset 的长度生成一个固定随机种子 对于这个函数 只要 dataset 没变, 输出的划分就不会变。
    dataset_id = str(len(dataset)).encode('utf-8')
    seed = int(hashlib.sha256(dataset_id).hexdigest(), 16) % (2**32)

    # 2. 创建局部随机生成器
    rng = np.random.default_rng(seed)

    # 3. 打乱并划分
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    # 4. 至少抽一张
    cal_index = max(1, math.ceil(calibration_ratio * len(indices)))

    cal_indices = indices[:cal_index]
    test_indices = indices[cal_index:]

    return cal_indices, test_indices


# ---------------------------------------------------------------------------------------------------------------------
# 数据集合并之后混合抽取  D_cal, D_test = D\D_cal
#----------------------------------------------------------------------------------------------------------------------
def split_dataset_test(combined_dataset, sub_dataset, calibration_ratio): 
    cal_indices, test_indices = split_dataset_cal(combined_dataset,calibration_ratio) # 合并后的数据集中的indices
    
    start_index, end_index = find_subset_indices(combined_dataset=combined_dataset, sub_dataset=sub_dataset)
    indices = np.arange(start_index, end_index + 1) # [start_index, start_index + 1, ..., end_index]
    
    test_indices = np.intersect1d(test_indices, indices) - start_index

    return test_indices


def find_subset_indices(combined_dataset, sub_dataset):
    N, n = len(combined_dataset), len(sub_dataset)
    '''
    在 dataset 的定义中, return {"image_name": img_name, "mask": mask, "mask_binary": mask_binary}
    即我定义的 dataset 在比较的时候, 本质是字典在比较
    '''
    for i in range(N - n + 1):
        match = True
        for j in range(n):
            if combined_dataset[i + j]['image_name'] != sub_dataset[j]['image_name']:
                match = False
                break
        if match:
            return i, i + n - 1
    return None





'''
def split_dataset_test(dataset1, dataset2, dataset3, dataset4, dataset5, calibration_ratio=0.2, random_seed=4): # 大数据集0.2, 小数据集0.7
    np.random.seed(random_seed)
    
    combined_dataset = ConcatDataset([dataset1,dataset2,dataset3, dataset4, dataset5])
    cal_indices, test_indices = split_dataset_cal(combined_dataset,calibration_ratio=0.2,random_seed=4) 


    lengths = [len(dataset1),len(dataset2),len(dataset3),len(dataset4),len(dataset5)] # [3, 4, 2, 5, 6]
    indices_cumsum = np.cumsum([0] + lengths)                                         # [0, 3, 7, 9, 14, 20]
    indices1 = np.arange(indices_cumsum[0], indices_cumsum[1])                        # [first_index, last_index]=[0, 1, 2]
    indices2 = np.arange(indices_cumsum[1], indices_cumsum[2])                        # [3, 4, 5, 6]
    indices3 = np.arange(indices_cumsum[2], indices_cumsum[3])
    indices4 = np.arange(indices_cumsum[3], indices_cumsum[4])
    indices5 = np.arange(indices_cumsum[4], indices_cumsum[5])
    
                                                                    
    test_indices1 = np.intersect1d(test_indices, indices1) - indices1[0]  # 在 dataset1 的index, 不是 combined_dataset的位置
    test_indices2 = np.intersect1d(test_indices, indices2) - indices1[1]  # [3,6,8,10] intersect [3,4,5,6] = [3,6]
    test_indices3 = np.intersect1d(test_indices, indices3) - indices1[2] 
    test_indices4 = np.intersect1d(test_indices, indices4) - indices1[3] 
    test_indices5 = np.intersect1d(test_indices, indices5) - indices1[4] 

    return cal_indices, test_indices1, test_indices2, test_indices3, test_indices4, test_indices5
'''



    

