import numpy as np

# 读取文件
data = np.load('/data/home/yunxin/Forgery_CP/Data/output_data/casia/Sp_D_CND_A_pla0005_pla0023_0281_mask_binary.npy')

print(data[:5])

# 或简单统计信息
print("最小值:", data.min())
print("最大值:", data.max())
print("均值:", data.mean())


